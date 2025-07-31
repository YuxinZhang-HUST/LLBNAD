import csv
import random
import os
import copy
import glob
import shutil
import datetime

import pandas as pd
import tabulate
import torch
from util.util import makedirs, log_cfg, able, log_msg, get_log_terms, update_log_term
from util.net import trans_state_dict, print_networks, get_timepc, reduce_tensor
from util.net import get_loss_scaler, get_autocast, distribute_bn
from optim.scheduler import get_scheduler
from data import get_loader, get_split_loader
from model import get_model
from optim import get_optim
from loss import get_loss_terms
from util.metric import get_evaluator
from timm.data import Mixup
from util.vis import vis_rgb_gt_amp
from thop import profile
import numpy as np
from torch.nn.parallel import DistributedDataParallel as NativeDDP

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model as ApexSyncBN
except:
    from timm.layers.norm_act import convert_sync_batchnorm as ApexSyncBN
from timm.layers.norm_act import convert_sync_batchnorm as TIMMSyncBN
from timm.utils import dispatch_clip_grad
from util.util import save_metric
import time
# from . import TRAINER
from util.registry import Registry
from data import get_dataset
TRANSFORMS = Registry('Transforms')
DATA = Registry('Data')
from torch.utils.data import DataLoader, Subset
import glob
import importlib

from data.dataset_info import *
from data.sampler import *
files = glob.glob('data/[!_]*.py')
for file in files:
    model_lib = importlib.import_module(file.split('.')[0].replace('/', '.'))

from data.utils import get_transforms
from util.compute_am import compute_discrepancy_map, maximum_as_anomaly_score
from ._base_trainer import BaseTrainer
from . import TRAINER
import torch
from torch.nn import functional as F

import os
import copy
import glob
import shutil
import datetime
import tabulate
import torch
from util.util import makedirs, log_cfg, able, log_msg, get_log_terms, update_log_term
from util.net import trans_state_dict, print_networks, get_timepc, reduce_tensor
from util.vis import vis_rgb_gt_amp
from thop import profile
import numpy as np
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from model import get_model


@TRAINER.register_module
class DinomalyTrainer():
	def __init__(self, cfg):
		self.cfg = cfg
		self.master, self.logger, self.writer = cfg.master, cfg.logger, cfg.writer
		self.local_rank, self.rank, self.world_size = cfg.local_rank, cfg.rank, cfg.world_size
		log_msg(self.logger, '==> Running Trainer: {}'.format(cfg.trainer.name))
		# =========> dataset <=================================
		# cfg.logdir_train, cfg.logdir_test = f'{cfg.logdir}/show_train', f'{cfg.logdir}/show_test'
		# makedirs([cfg.logdir_train, cfg.logdir_test], exist_ok=True)
		log_msg(self.logger, "==> Loading training dataset: {}".format(cfg.train_data.type))
		log_msg(self.logger, "==> Loading testing dataset: {}".format(cfg.test_data.type))

		self.train_loader, self.sub_train_loaders, self.test_train_loaders, self.test_loader, self.train_set, self.test_set= get_split_loader(cfg)
		# for info in self.train_set.verbose_info:
		# 	log_msg(self.logger, "==> Dataset info: {}".format(info))
		for info in self.test_set.verbose_info:
			log_msg(self.logger, "==> Dataset info: {}".format(info))

		cfg.train_data.train_size, cfg.test_data.test_size = len(self.train_loader), len(self.test_loader)
		cfg.train_data.train_length, cfg.test_data.test_length = self.train_loader.dataset.length, self.test_loader.dataset.length

		self.test_cls_names = self.test_loader.dataset.cls_names
		self.train_cls_names = self.train_loader.dataset.cls_names
		self.all_cls_names = self.test_loader.dataset.all_cls_names

		# =========> model <=================================
		log_msg(self.logger, '==> Using GPU: {} for Training'.format(list(range(cfg.world_size))))
		log_msg(self.logger, '==> Building model')

		if self.cfg.mode in ['test']:
			if not (cfg.model.kwargs['checkpoint_path'] or cfg.trainer.resume_dir):
				cfg.model.kwargs['checkpoint_path'] = f'{self.cfg.logdir}/{self.train_cls_names}_ckpt.pth'
				cfg.model.kwargs['strict'] = False
				log_msg(self.logger, f"==> Automatically Generate checkpoint: {cfg.model.kwargs['checkpoint_path']}")

		self.net = get_model(cfg.model)
		self.net.to('cuda:{}'.format(cfg.local_rank))
		self.net.eval()
		log_msg(self.logger, f"==> Load checkpoint: {cfg.model.kwargs['checkpoint_path']}") if cfg.model.kwargs[
			'checkpoint_path'] else None
		# print_networks([self.net], torch.randn(self.cfg.fvcore_b, self.cfg.fvcore_c, self.cfg.size, self.cfg.size).cuda(), self.logger) if self.cfg.fvcore_is else None

		### Others
		log_msg(self.logger, '==> Creating optimizer')

		self.optim = get_optim(cfg.optim.kwargs, self.net, lr=cfg.optim.lr)
		self.amp_autocast = get_autocast(cfg.trainer.scaler)
		self.loss_scaler = get_loss_scaler(cfg.trainer.scaler)
		self.loss_terms = get_loss_terms(cfg.loss.loss_terms, device='cuda:{}'.format(cfg.local_rank))

		self.scheduler = get_scheduler(cfg, self.optim)
		self.evaluator = get_evaluator(cfg.evaluator)
		self.metrics = self.evaluator.metrics

		cfg.trainer.metric_recorder = dict()
		self.selected_data_recoder = dict()
		for idx, cls_name in enumerate(self.test_cls_names):
			for metric in self.metrics:
				cfg.trainer.metric_recorder.update({f'{metric}_{cls_name}': []})
				if idx == len(self.test_cls_names) - 1 and len(self.test_cls_names) > 1:
					cfg.trainer.metric_recorder.update({f'{metric}_Avg': []})
		self.metric_recorder = cfg.trainer.metric_recorder


		self.iter, self.epoch = cfg.trainer.iter, cfg.trainer.epoch
		self.iter_full, self.epoch_full = cfg.trainer.iter_full, cfg.trainer.epoch_full
		self.iteration = cfg.iteration		# 固定stage1迭代次数
		if cfg.trainer.resume_dir:
			state_dict = torch.load(cfg.model.kwargs['checkpoint_path'], map_location='cpu')
			self.optim.load_state_dict(state_dict['optimizer'])
			self.scheduler.load_state_dict(state_dict['scheduler'])
			self.loss_scaler.load_state_dict(state_dict['scaler']) if self.loss_scaler else None
			self.cfg.task_start_time = get_timepc() - state_dict['total_time']

		tmp_dir = f'{cfg.trainer.checkpoint}/tmp'
		tem_i = 0
		while os.path.exists(f'{tmp_dir}/{tem_i}'):
			tem_i += 1
		self.tmp_dir = f'{tmp_dir}/{tem_i}'
		log_cfg(self.cfg)

	def pre_train(self):
		# self.complexity_analysis((1,3,256,256))
		pass

	def pre_test(self):
		pass

	def set_input(self, inputs, **kwargs):
		self.imgs = inputs['img'].cuda()
		self.imgs_mask = inputs['img_mask'].cuda()

		self.bs = self.imgs.shape[0]

		### Note: necessray for evaluations and visualizations
		self.cls_name = inputs['cls_name']
		self.anomaly = inputs['anomaly']
		self.img_path = inputs['img_path']

	def forward(self, **kwargs):
		self.en, self.de = self.net(self.imgs)

	def compute_loss(self, **kwargs):
		dinomaly_loss = self.loss_terms['dinomaly_loss'](self.en, self.de)
		loss_log = {'dinomaly_loss': dinomaly_loss}
		return dinomaly_loss, loss_log

	def compute_anomaly_scores(self, max_ratio = 0):
		anomaly_map, anomaly_map_list = compute_discrepancy_map(self.en, self.de,
														[self.imgs.shape[2], self.imgs.shape[3]], use_cos=True,
														uni_am=False, amap_mode='add', gaussian_sigma=4)
		anomaly_score = maximum_as_anomaly_score(anomaly_map, max_ratio)
		return anomaly_map, anomaly_score

	def get_train_sampler(self):
		train_set, test_set = get_dataset(self.cfg)
		if self.cfg.train_data.sampler.name == 'naive':
			train_sampler = None
		elif self.cfg.train_data.sampler.name == 'balanced':
			train_sampler = SAMPLER.get(self.cfg.train_data.sampler.name, None)

			if train_sampler:
				train_sampler = train_sampler(batch_size=self.cfg.trainer.data.batch_size_per_gpu, dataset=train_set)
		else:
			raise NotImplementedError
		return train_sampler

	# several ways to select data for stage2, including select top t% the lowest average anomaly score
	def get_selected_loader(self, model_result):
		# calculate the average anomaly score of each subset tested in all submodels.
		anomaly_scores_mean = []
		for subset_idx in range(self.cfg.subset_num):
			scores = []
			for model_idx in range(self.cfg.subset_num):
				scores.append(model_result[model_idx][subset_idx]['anomaly_scores'])
			scores = np.array(scores)
			subset_mean = np.mean(scores, axis=0)
			anomaly_scores_mean.append(subset_mean)

		# concat anomaly_scores_mean and anomalys of all subsets.
		all_scores = np.concatenate(anomaly_scores_mean)
		all_anomalys = np.concatenate(
			[model_result[0][subset_idx]['anomalys'] for subset_idx in range(self.cfg.subset_num)])

		# calculate the maximum×var of anomaly score of each subset tested in all submodels.
		# anomaly_scores_max_var = []
		# for subset_idx in range(self.cfg.subset_num):
		# 	scores = []
		# 	for model_idx in range(self.cfg.subset_num):
		# 		scores.append(model_result[model_idx][subset_idx]['anomaly_scores'])
		# 	scores = np.array(scores)
		# 	subset_max = np.max(scores, axis=0)
		# 	subset_var = np.var(scores, axis=0)
		# 	subset_max_var = subset_max * subset_var
		# 	anomaly_scores_max_var.append(subset_max_var)
		#
		# # concat anomaly_scores_max_var and anomalys of all subsets.
		# all_scores = np.concatenate(anomaly_scores_max_var)
		# all_anomalys = np.concatenate(
		# 	[model_result[0][subset_idx]['anomalys'] for subset_idx in range(self.cfg.subset_num)])

		# calculate the maximum anomaly score of each subset tested in all submodels.
		# anomaly_scores_max = []
		# for subset_idx in range(self.cfg.subset_num):
		# 	scores = []
		# 	for model_idx in range(self.cfg.subset_num):
		# 		scores.append(model_result[model_idx][subset_idx]['anomaly_scores'])
		# 	scores = np.array(scores)
		# 	subset_max = np.max(scores, axis=0)
		# 	anomaly_scores_max.append(subset_max)
		#
		# # concat anomaly_scores_max and anomalys of all subsets.
		# all_scores = np.concatenate(anomaly_scores_max)
		# all_anomalys = np.concatenate(
		# 	[model_result[0][subset_idx]['anomalys'] for subset_idx in range(self.cfg.subset_num)])

		num_top = int(len(all_scores) * self.cfg.threshold)
		top_indices = np.argsort(all_scores)[:num_top]
		anomaly_num = np.sum(all_anomalys[top_indices])
		anomaly_ratio = anomaly_num / num_top if num_top > 0 else 0

		print(f"In the first {self.cfg.threshold} of the image, anomaly_num：{anomaly_num}，anomaly_ratio: {anomaly_ratio}")

		self.selected_data_recoder[self.train_cls_names[0]] = {
			'noisy_ratio': self.cfg.noisy_ratio,
			'threshold': self.cfg.threshold,
			'anomaly_num': anomaly_num,
			'anomaly_ratio': anomaly_ratio
		}

		self.save_selected_data_to_csv()
		# creat new dataset for stage2
		dataset_selected = Subset(self.train_set, top_indices)
		train_sampler = self.get_train_sampler()

		if train_sampler:
			train_loader_selected = torch.utils.data.DataLoader(dataset=dataset_selected,
																batch_sampler=train_sampler,
																num_workers=self.cfg.trainer.data.num_workers_per_gpu,
																pin_memory=self.cfg.trainer.data.pin_memory,
																persistent_workers=self.cfg.trainer.data.persistent_workers)
		else:
			train_loader_selected = torch.utils.data.DataLoader(dataset=dataset_selected,
																batch_size=self.cfg.trainer.data.batch_size_per_gpu,
																shuffle=True,
																sampler=train_sampler,
																num_workers=self.cfg.trainer.data.num_workers_per_gpu,
																pin_memory=self.cfg.trainer.data.pin_memory,
																drop_last=self.cfg.trainer.data.drop_last,
																persistent_workers=self.cfg.trainer.data.persistent_workers)

		return train_loader_selected

	# save information of the selected data for stage2
	def save_selected_data_to_csv(self):
		path = f'{self.cfg.logdir}/{self.cfg.model.name}_{self.cfg.train_data.name}_{self.cfg.train_data.mode}_t{self.cfg.threshold}_num{self.cfg.subset_num}_meanscores_selected.csv'
		header = ['class_name', 'noisy_ratio', 'threshold', 'anomaly_num', 'anomaly_ratio']
		file_exists = os.path.exists(path)
		with open(path, 'a', newline='') as csvfile:
			writer = csv.DictWriter(csvfile, fieldnames=header)
			if not file_exists:
				writer.writeheader()
			for cls_name, data in self.selected_data_recoder.items():
				row = {
					'class_name': cls_name,
					'noisy_ratio': data['noisy_ratio'],
					'threshold': data['threshold'],
					'anomaly_num': data['anomaly_num'],
					'anomaly_ratio': data['anomaly_ratio']
				}
				writer.writerow(row)

	# save information of all the data tested by each submodel in stage1
	def save_submodel_result(self, ith_model, submodel_result):
		save_dir = os.path.join(self.cfg.logdir,"stage1_results",self.train_cls_names[0])
		save_path = os.path.join(save_dir, f"{ith_model}.csv")
		os.makedirs(save_dir, exist_ok=True)
		all_data = []

		for ith_subset, result in enumerate(submodel_result):
			data = {
				"ith_subset": [ith_subset] * len(result["cls_names"]),
				"cls_names": result["cls_names"],
				"anomalys": result["anomalys"],
				"anomaly_scores": result["anomaly_scores"],
				"img_paths": result["img_paths"]
			}
			all_data.append(pd.DataFrame(data))
		combined_df = pd.concat(all_data, ignore_index=True)
		combined_df.to_csv(save_path, index=False)

	def train(self):
		model_result = []
		self.epoch_full = self.cfg.stage1_epoch_full
		for ith_model, sub_train_loader in enumerate(self.sub_train_loaders):
			self.cfg.train_data.train_size = len(sub_train_loader)
			self.net = get_model(self.cfg.model)
			self.net.to('cuda:{}'.format(self.cfg.local_rank))
			self.optim = get_optim(self.cfg.optim.kwargs, self.net, lr=self.cfg.optim.lr)
			self.scheduler = get_scheduler(self.cfg, self.optim)
			self.reset(isTrain=True)
			self.pre_train()
			train_length = len(sub_train_loader)
			train_loader = iter(sub_train_loader)
			self.iter = 0
			self.epoch = 0
			# while self.epoch < self.epoch_full and self.iter < self.iter_full:
			while self.iter < self.iteration:
				self.scheduler_step(self.iter)
				# ---------- data ----------
				t1 = get_timepc()
				self.iter += 1
				train_data = next(train_loader)

				self.set_input(train_data, train=True)
				t2 = get_timepc()
				update_log_term(self.log_terms.get('data_t'), t2 - t1, 1, self.master)
				# ---------- optimization ----------
				self.optimize_parameters()
				t3 = get_timepc()
				update_log_term(self.log_terms.get('optim_t'), t3 - t2, 1, self.master)
				update_log_term(self.log_terms.get('batch_t'), t3 - t1, 1, self.master)
				# ---------- log ----------
				# if self.master:
				# 	if self.iter % self.cfg.logging.train_log_per == 0:
				# 		msg = able(self.progress.get_msg(self.iter, self.iter_full, self.iter / train_length,
				# 										 self.iter_full / train_length), self.master, None)
				# 		log_msg(self.logger, msg)
				# 		if self.writer:
				# 			for k, v in self.log_terms.items():
				# 				self.writer.add_scalar(f'Train/{k}', v.val, self.iter)
				# 			self.writer.flush()
				# if self.iter % self.cfg.logging.train_reset_log_per == 0:
				# 	self.reset(isTrain=True)
				# ---------- update train_loader ----------
				if self.iter % train_length  == 0:
					self.epoch += 1
					self.optim.sync_lookahead() if hasattr(self.optim, 'sync_lookahead') else None

				# 	if self.epoch >= self.cfg.trainer.test_start_epoch or self.epoch % self.cfg.trainer.test_per_epoch == 0:
				# 		if self.epoch + self.cfg.trainer.test_per_epoch > self.epoch_full:  # last epoch
				# 			vis = True
				# 		else:
				# 			vis = False
				# 		self.test(vis)
				# 	else:
				# 		self.test_ghost()
				# 	self.cfg.total_time = get_timepc() - self.cfg.task_start_time
				# 	total_time_str = str(datetime.timedelta(seconds=int(self.cfg.total_time)))
				# 	eta_time_str = str(
				# 		datetime.timedelta(seconds=int(self.cfg.total_time / self.epoch * (self.epoch_full - self.epoch))))
				# 	log_msg(self.logger,
				# 			f'==> Total time: {total_time_str}\t Eta: {eta_time_str} \tLogged in \'{self.cfg.logdir}\'')
				# 	self.save_checkpoint()
					self.reset(isTrain=True)
					train_loader = iter(sub_train_loader)
			# finished training, test all the subsets
			submodel_result = self.test_submodel(vis=True)
			model_result.append(submodel_result)
			self.save_submodel_result(ith_model, submodel_result)
			print(f"{ith_model}trained, saved")
		# initialize model for stage 2
		self.net = get_model(self.cfg.model)
		self.net.to('cuda:{}'.format(self.cfg.local_rank))
		train_selected_loader = self.get_selected_loader(model_result)
		self.cfg.train_data.train_size = len(train_selected_loader)
		train_length = len(train_selected_loader)
		self.net = get_model(self.cfg.model)
		self.net.to('cuda:{}'.format(self.cfg.local_rank))
		# initialize optimizer
		self.optim = get_optim(self.cfg.optim.kwargs, self.net, lr=self.cfg.optim.lr)
		# initialize scheduler
		self.scheduler = get_scheduler(self.cfg, self.optim)
		self.iter = 0
		self.epoch = 0
		self.epoch_full = self.cfg.stage2_epoch_full
		train_loader = iter(train_selected_loader)
		while self.epoch < self.epoch_full and self.iter < self.iter_full:
			self.scheduler_step(self.iter)
			# ---------- data ----------
			t1 = get_timepc()
			self.iter += 1
			train_data = next(train_loader)

			self.set_input(train_data, train=True)
			t2 = get_timepc()
			update_log_term(self.log_terms.get('data_t'), t2 - t1, 1, self.master)
			# ---------- optimization ----------
			self.optimize_parameters()
			t3 = get_timepc()
			update_log_term(self.log_terms.get('optim_t'), t3 - t2, 1, self.master)
			update_log_term(self.log_terms.get('batch_t'), t3 - t1, 1, self.master)
			# ---------- log ----------
			if self.master:
				if self.iter % self.cfg.logging.train_log_per == 0:
					msg = able(self.progress.get_msg(self.iter, self.iter_full, self.iter / train_length,
													 self.iter_full / train_length), self.master, None)
					log_msg(self.logger, msg)
					if self.writer:
						for k, v in self.log_terms.items():
							self.writer.add_scalar(f'Train/{k}', v.val, self.iter)
						self.writer.flush()
			if self.iter % self.cfg.logging.train_reset_log_per == 0:
				self.reset(isTrain=True)
			# ---------- update train_loader ----------
			if self.iter % train_length == 0:
				self.epoch += 1
				self.optim.sync_lookahead() if hasattr(self.optim, 'sync_lookahead') else None

				if self.epoch >= self.cfg.trainer.test_start_epoch or self.epoch % self.cfg.trainer.test_per_epoch == 0:
					if self.epoch + self.cfg.trainer.test_per_epoch > self.epoch_full:  # last epoch
						vis = True
					else:
						vis = False
					self.test(vis)
				else:
					self.test_ghost()
				self.cfg.total_time = get_timepc() - self.cfg.task_start_time
				total_time_str = str(datetime.timedelta(seconds=int(self.cfg.total_time)))
				eta_time_str = str(
					datetime.timedelta(seconds=int(self.cfg.total_time / self.epoch * (self.epoch_full - self.epoch))))
				log_msg(self.logger,
						f'==> Total time: {total_time_str}\t Eta: {eta_time_str} \tLogged in \'{self.cfg.logdir}\'')
				self.save_checkpoint()
				self.reset(isTrain=True)
				train_loader = iter(train_selected_loader)

		self._finish()

	@torch.no_grad()
	def test_ghost(self):
		for idx, cls_name in enumerate(self.test_cls_names):
			for metric in self.metrics:
				self.metric_recorder[f'{metric}_{cls_name}'].append(0)
				if idx == len(self.test_cls_names) - 1 and len(self.test_cls_names) > 1:
					self.metric_recorder[f'{metric}_Avg'].append(0)

	def save_scores(self, results, N):

		anomaly_maps = copy.deepcopy(results['anomaly_maps'])
		imgs_masks = copy.deepcopy(results['imgs_masks'])

		# 将 anomaly_maps 和 imgs_masks 展平，便于操作
		anomaly_maps_flat = anomaly_maps.flatten()
		imgs_masks_flat = imgs_masks.flatten()

		non_nan_mask = ~np.isnan(anomaly_maps_flat)

		# 使用布尔掩码提取 anomaly_maps_flat 和 imgs_masks_flat 中对应的非 NaN 值
		anomaly_maps_flat = anomaly_maps_flat[non_nan_mask]
		imgs_masks_flat = imgs_masks_flat[non_nan_mask]

		# anomaly_maps_flat = (anomaly_maps_flat-np.min(anomaly_maps_flat))/(np.max(anomaly_maps_flat)-np.min(anomaly_maps_flat))
		# 正常像素的异常分值（imgs_masks > 0.5）
		normal_scores = anomaly_maps_flat[imgs_masks_flat < 0.5]

		# 异常像素的异常分值（imgs_masks <= 0.5）
		abnormal_scores = anomaly_maps_flat[imgs_masks_flat >= 0.5]

		# 分别从正常和异常分值中随机采样 N 个值

		sampled_normal_scores = np.random.choice(normal_scores, min(N, normal_scores.shape[0]), replace=False)
		sampled_abnormal_scores = np.random.choice(abnormal_scores, min(N, abnormal_scores.shape[0]), replace=False)

		normal_scores_path = f'{self.cfg.logdir}/{self.test_cls_names}_normal_scores.npy'
		abnormal_scores_path = f'{self.cfg.logdir}/{self.test_cls_names}_abnormal_scores.npy'

		np.save(normal_scores_path, sampled_normal_scores)
		np.save(abnormal_scores_path, sampled_abnormal_scores)

		print(f"Saved normal scores to {normal_scores_path}")
		print(f"Saved abnormal scores to {abnormal_scores_path}")

	@torch.no_grad()
	def test_submodel(self, vis=True):
		self.pre_test()
		if self.master:
			if os.path.exists(self.tmp_dir):
				shutil.rmtree(self.tmp_dir)
			os.makedirs(self.tmp_dir, exist_ok=True)
		self.reset(isTrain=False)
		submodel_result = []

		for ith_subset, test_train_loader in enumerate(self.test_train_loaders):
			imgs_masks, anomaly_maps, cls_names, anomalys, anomaly_scores, img_paths = [], [], [], [], [], []
			batch_idx = 0
			test_length = len(test_train_loader)
			test_loader = iter(test_train_loader)
			while batch_idx < test_length:
				t1 = get_timepc()
				batch_idx += 1
				test_data = next(test_loader)
				self.set_input(test_data, train=False)
				self.forward(train=False)

				# loss, loss_log = self.compute_loss(train=False)
				# for k, v in loss_log.items():
				#     update_log_term(self.log_terms.get(k),
				#                     reduce_tensor(v, self.world_size).clone().detach().item(),
				#                     1, self.master)

				anomaly_map, anomaly_score = self.compute_anomaly_scores(self.cfg.max_ratio)
				self.imgs_mask[self.imgs_mask > 0.5], self.imgs_mask[self.imgs_mask <= 0.5] = 1, 0
				# if self.cfg.vis and vis:
				# 	if self.cfg.vis_dir is not None:
				# 		root_out = self.cfg.vis_dir
				# 	else:
				# 		root_out = self.cfg.logdir
				# 	vis_rgb_gt_amp(self.img_path, self.imgs, self.imgs_mask.cpu().numpy().astype(int),
				# 				   anomaly_map, self.cfg.model.name, root_out)

				imgs_masks.append(self.imgs_mask.cpu().numpy().astype(int))
				anomaly_maps.append(anomaly_map)
				anomaly_scores.append(anomaly_score)
				cls_names.append(np.array(self.cls_name))
				anomalys.append(self.anomaly.cpu().numpy().astype(int))
				img_paths.append(np.array(self.img_path))
				t2 = get_timepc()
				update_log_term(self.log_terms.get('batch_t'), t2 - t1, 1, self.master)
				print(f'\r{batch_idx}/{test_length}', end='') if self.master else None
				# ---------- log ----------
				if batch_idx % self.cfg.logging.test_log_per == 0 or batch_idx == test_length:
					msg = able(self.progress.get_msg(batch_idx, test_length, 0, 0, prefix=f'Test'), self.master, None)
					log_msg(self.logger, msg)
			results = dict(imgs_masks = imgs_masks, anomaly_maps = anomaly_maps, cls_names = cls_names,
					    anomalys = anomalys, anomaly_scores = anomaly_scores, img_paths = img_paths)
			results = {k: np.concatenate(v, axis=0) for k, v in results.items()}
			submodel_result.append(results)
		return submodel_result


		# self.save_scores(results, N=10000)

		# msg = {}
		# for idx, cls_name in enumerate(self.test_cls_names):
		# 	metric_results = self.evaluator.run(results, cls_name, self.logger)
		# 	msg['Name'] = msg.get('Name', [])
		# 	msg['Name'].append(cls_name)
		# 	avg_act = True if len(self.test_cls_names) > 1 and idx == len(self.test_cls_names) - 1 else False
		# 	msg['Name'].append('Avg') if avg_act else None
		#
		# 	for metric in self.metrics:
		# 		metric_result = metric_results[metric] * 100
		# 		self.metric_recorder[f'{metric}_{cls_name}'].append(metric_result)
		# 		if self.writer:
		# 			self.writer.add_scalar(f'Test/{metric}_{cls_name}', metric_result, self.iter)
		# 			self.writer.flush()
		# 		max_metric = max(self.metric_recorder[f'{metric}_{cls_name}'])
		# 		max_metric_idx = self.metric_recorder[f'{metric}_{cls_name}'].index(max_metric) + 1
		# 		msg[metric] = msg.get(metric, [])
		# 		msg[metric].append(metric_result)
		# 		msg[f'{metric} (Max)'] = msg.get(f'{metric} (Max)', [])
		# 		msg[f'{metric} (Max)'].append(f'{max_metric:.3f} ({max_metric_idx:<3d} epoch)')
		# 		if avg_act:
		# 			metric_result_avg = sum(msg[metric]) / len(msg[metric])
		# 			self.metric_recorder[f'{metric}_Avg'].append(metric_result_avg)
		# 			max_metric = max(self.metric_recorder[f'{metric}_Avg'])
		# 			max_metric_idx = self.metric_recorder[f'{metric}_Avg'].index(max_metric) + 1
		# 			msg[metric].append(metric_result_avg)
		# 			msg[f'{metric} (Max)'].append(f'{max_metric:.3f} ({max_metric_idx:<3d} epoch)')
		#
		# msg = tabulate.tabulate(msg, headers='keys', tablefmt="pipe", floatfmt='.3f', numalign="center",
		# 						stralign="center", )
		# log_msg(self.logger, f'\n{msg}')

	def save_checkpoint(self):
		if self.master:
			checkpoint_info = {'net': trans_state_dict(self.net.get_learnable_params(), dist=False),
							   'optimizer': self.optim.state_dict(),
							   'scheduler': self.scheduler.state_dict(),
							   'scaler': self.loss_scaler.state_dict() if self.loss_scaler else None,
							   'iter': self.iter,
							   'epoch': self.epoch,
							   'metric_recorder': self.metric_recorder,
							   'total_time': self.cfg.total_time}
			save_path = f'{self.cfg.logdir}/{self.train_cls_names}_ckpt.pth'
			torch.save(checkpoint_info, save_path)
			torch.save(checkpoint_info['net'], f'{self.cfg.logdir}/{self.train_cls_names}_net.pth')
			if self.epoch % self.cfg.trainer.test_per_epoch == 0:
				torch.save(checkpoint_info['net'], f'{self.cfg.logdir}/{self.train_cls_names}_net_{self.epoch}.pth')

	def run(self):
		log_msg(self.logger,
				f'==> Starting {self.cfg.mode}ing')
		if self.cfg.mode in ['train']:
			self.train()
		elif self.cfg.mode in ['test']:
			self.test()
		else:
			raise NotImplementedError

	def reset(self, isTrain=True):
		self.net.train(mode=isTrain)
		self.log_terms, self.progress = get_log_terms(
			able(self.cfg.logging.log_terms_train, isTrain, self.cfg.logging.log_terms_test),
			default_prefix=('Train' if isTrain else 'Test'))

	def scheduler_step(self, step):
		self.scheduler.step(step)
		update_log_term(self.log_terms.get('lr'), self.optim.param_groups[0]["lr"], 1, self.master)

	def backward_term(self, loss_term, optim):
		optim.zero_grad()
		if self.loss_scaler:
			self.loss_scaler(loss_term, optim, clip_grad=self.cfg.loss.clip_grad, parameters=self.net.parameters(),
							 create_graph=self.cfg.loss.create_graph)
		else:
			loss_term.backward(retain_graph=self.cfg.loss.retain_graph)
			if self.cfg.loss.clip_grad is not None:
				dispatch_clip_grad(self.net.parameters(), value=self.cfg.loss.clip_grad)
			optim.step()

	def optimize_parameters(self):
		with self.amp_autocast():
			self.forward(train=True)
			loss, loss_log = self.compute_loss(train=True)
			# print(f"Loss value: {loss.item()}")

		self.backward_term(loss, self.optim)

		for k, v in loss_log.items():
			update_log_term(self.log_terms.get(k),
							reduce_tensor(v, self.world_size).clone().detach().item(),
							1, self.master)

	def _finish(self):
		log_msg(self.logger, 'finish training')
		self._save_metrics()

	def _save_metrics(self):
		## reorganize metric recorder
		metric_last = dict()
		metric_best = dict()

		for idx, cls_name in enumerate(self.test_cls_names):
			metric_last[cls_name] = dict()
			metric_best[cls_name] = dict()

			for metric in self.metrics:
				metric_last[cls_name][metric] = self.metric_recorder[f'{metric}_{cls_name}'][-1]
				metric_best[cls_name][metric] = max(self.metric_recorder[f'{metric}_{cls_name}'])

		metric_last_csv_path = f'{self.cfg.logdir}/{self.cfg.model.name}_{self.cfg.test_data.name}_{self.cfg.train_data.mode}_t{self.cfg.threshold}_num{self.cfg.subset_num}_meanscores_last.csv'
		metric_best_csv_path = f'{self.cfg.logdir}/{self.cfg.model.name}_{self.cfg.test_data.name}_{self.cfg.train_data.mode}_t{self.cfg.threshold}_num{self.cfg.subset_num}_meanscores_best.csv'

		for idx, cls_name in enumerate(self.test_cls_names):
			save_metric(metric_last[cls_name], self.all_cls_names, cls_name, metric_last_csv_path)
			save_metric(metric_best[cls_name], self.all_cls_names, cls_name, metric_best_csv_path)

	@torch.no_grad()
	def test_ghost(self):
		for idx, cls_name in enumerate(self.test_cls_names):
			for metric in self.metrics:
				self.metric_recorder[f'{metric}_{cls_name}'].append(0)
				if idx == len(self.test_cls_names) - 1 and len(self.test_cls_names) > 1:
					self.metric_recorder[f'{metric}_Avg'].append(0)

	@torch.no_grad()
	def test(self, vis=True):
		self.pre_test()

		if self.master:
			if os.path.exists(self.tmp_dir):
				shutil.rmtree(self.tmp_dir)
			os.makedirs(self.tmp_dir, exist_ok=True)
		self.reset(isTrain=False)
		imgs_masks, anomaly_maps, cls_names, anomalys, anomaly_scores, img_names = [], [], [], [], [], []
		batch_idx = 0
		test_length = self.cfg.test_data.test_size
		test_loader = iter(self.test_loader)
		while batch_idx < test_length:
			t1 = get_timepc()
			batch_idx += 1
			test_data = next(test_loader)

			self.set_input(test_data, train=False)
			self.forward(train=False)

			# loss, loss_log = self.compute_loss(train=False)
			# for k, v in loss_log.items():
			#     update_log_term(self.log_terms.get(k),
			#                     reduce_tensor(v, self.world_size).clone().detach().item(),
			#                     1, self.master)

			anomaly_map, anomaly_score = self.compute_anomaly_scores()
			self.imgs_mask[self.imgs_mask > 0.5], self.imgs_mask[self.imgs_mask <= 0.5] = 1, 0
			if self.cfg.vis and vis:
				if self.cfg.vis_dir is not None:
					root_out = self.cfg.vis_dir
				else:
					root_out = self.cfg.logdir
				vis_rgb_gt_amp(self.img_path, self.imgs, self.imgs_mask.cpu().numpy().astype(int),
							   anomaly_map, self.cfg.model.name, root_out)

			imgs_masks.append(self.imgs_mask.cpu().numpy().astype(int))
			anomaly_maps.append(anomaly_map)
			anomaly_scores.append(anomaly_score)
			cls_names.append(np.array(self.cls_name))
			anomalys.append(self.anomaly.cpu().numpy().astype(int))
			t2 = get_timepc()
			update_log_term(self.log_terms.get('batch_t'), t2 - t1, 1, self.master)
			print(f'\r{batch_idx}/{test_length}', end='') if self.master else None
			# ---------- log ----------
			if batch_idx % self.cfg.logging.test_log_per == 0 or batch_idx == test_length:
				msg = able(self.progress.get_msg(batch_idx, test_length, 0, 0, prefix=f'Test'), self.master, None)
				log_msg(self.logger, msg)

		results = dict(imgs_masks=imgs_masks, anomaly_maps=anomaly_maps,
					   cls_names=cls_names, anomalys=anomalys, anomaly_scores=anomaly_scores)

		results = {k: np.concatenate(v, axis=0) for k, v in results.items()}

		# self.save_scores(results, N=10000)

		msg = {}
		for idx, cls_name in enumerate(self.test_cls_names):
			metric_results = self.evaluator.run(results, cls_name, self.logger)
			msg['Name'] = msg.get('Name', [])
			msg['Name'].append(cls_name)
			avg_act = True if len(self.test_cls_names) > 1 and idx == len(self.test_cls_names) - 1 else False
			msg['Name'].append('Avg') if avg_act else None

			for metric in self.metrics:
				metric_result = metric_results[metric] * 100
				self.metric_recorder[f'{metric}_{cls_name}'].append(metric_result)
				if self.writer:
					self.writer.add_scalar(f'Test/{metric}_{cls_name}', metric_result, self.iter)
					self.writer.flush()
				max_metric = max(self.metric_recorder[f'{metric}_{cls_name}'])
				max_metric_idx = self.metric_recorder[f'{metric}_{cls_name}'].index(max_metric) + 1
				msg[metric] = msg.get(metric, [])
				msg[metric].append(metric_result)
				msg[f'{metric} (Max)'] = msg.get(f'{metric} (Max)', [])
				msg[f'{metric} (Max)'].append(f'{max_metric:.3f} ({max_metric_idx:<3d} epoch)')
				if avg_act:
					metric_result_avg = sum(msg[metric]) / len(msg[metric])
					self.metric_recorder[f'{metric}_Avg'].append(metric_result_avg)
					max_metric = max(self.metric_recorder[f'{metric}_Avg'])
					max_metric_idx = self.metric_recorder[f'{metric}_Avg'].index(max_metric) + 1
					msg[metric].append(metric_result_avg)
					msg[f'{metric} (Max)'].append(f'{max_metric:.3f} ({max_metric_idx:<3d} epoch)')

		msg = tabulate.tabulate(msg, headers='keys', tablefmt="pipe", floatfmt='.3f', numalign="center",
								stralign="center", )
		log_msg(self.logger, f'\n{msg}')
