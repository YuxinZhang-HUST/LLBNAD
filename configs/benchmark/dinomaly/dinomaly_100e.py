from argparse import Namespace
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD
import torchvision.transforms.functional as F
from configs.__base__ import *

class cfg(cfg_common, cfg_dataset_default, cfg_model_dinomaly):
    def __init__(self):
        # super(cfg, self).__init__()
        cfg_common.__init__(self)  # initialize evaluator,optim,trainer,loss,logging
        cfg_dataset_default.__init__(self)  # initialize dateset初始化
        cfg_model_dinomaly.__init__(self)  # initialize model初始化

        self.vis = True
        self.fvcore_b = 1
        self.fvcore_c = 3
        self.seed = 42

        self.stage1_epoch_full = 2
        self.stage2_epoch_full = 2
        self.epoch_full = 2
        self.warmup_epochs = 0
        self.test_start_epoch = 2
        self.test_per_epoch = self.epoch_full // 1  # self.test_per_epoch = self.epoch_full // 20
        self.batch_train = 16  # official 16
        self.batch_test_per = 16
        self.iteration = 10       # 1000

        self.lr = 0.005 * self.batch_train / 16
        self.weight_decay = 0.05
        self.max_ratio = 0.01       # anomaly_score, def maximum_as_anomaly_score(anomaly_map, max_ratio=0.):
        self.threshold = 0.2
        self.subset_num = 3
        self.noisy_ratio = 20
        ###### Train DATA
        self.train_data.type = 'DefaultAD'
        self.train_data.name = 'realiad'  # mvtec, visa
        self.train_data.mode = f'fuiad{self.noisy_ratio}'
        self.train_data.cls_names = ['audiojack']
        self.train_data.anomaly_generator.name = 'white_noise'
        self.train_data.anomaly_generator.enable = False  ### TODO
        self.train_data.anomaly_generator.kwargs = dict()
        self.train_data.sampler.name = 'naive'
        self.train_data.sampler.kwargs = dict()
        ###### Test DATA
        self.test_data.type = 'DefaultAD'
        self.test_data.name = 'realiad'  # mvtec, visa
        self.test_data.mode = f'fuiad{self.noisy_ratio}'
        self.test_data.cls_names = ['audiojack']

        self.size = 392

        self.model.name = 'dinomaly'
        self.model.kwargs = dict(pretrained=False, checkpoint_path='', strict=True,
                                 encoder_name='dinov2reg_vit_base_14')

        self.optim.lr = self.lr
        self.optim.kwargs = dict(name='adam', betas=(0.5, 0.999))

        # ==> trainer
        self.trainer.name = 'DinomalyTrainer'
        self.trainer.logdir_sub = f'{self.train_data.mode}_threshold{self.threshold}_subsetnum{self.subset_num}'##############保存路径改这里#####################
        self.trainer.resume_dir = ''
        self.trainer.epoch_full = self.epoch_full
        self.trainer.scheduler_kwargs = dict(
            name='step', lr_noise=None, noise_pct=0.67, noise_std=1.0, noise_seed=42, lr_min=self.lr / 1e2,
            warmup_lr=self.lr / 1e3, warmup_iters=-1, cooldown_iters=0, warmup_epochs=self.warmup_epochs,
            cooldown_epochs=0, use_iters=True,
            patience_iters=0, patience_epochs=0, decay_iters=0, decay_epochs=int(self.epoch_full * 0.8),
            cycle_decay=0.1, decay_rate=0.1)

        self.trainer.test_start_epoch = self.test_start_epoch
        self.trainer.test_per_epoch = self.test_per_epoch

        self.trainer.data.batch_size = self.batch_train
        self.trainer.data.batch_size_per_gpu_test = self.batch_test_per

        self.train_data.train_transforms = [
            dict(type='Resize', size=(self.size, self.size), interpolation=F.InterpolationMode.BILINEAR),
            dict(type='CenterCrop', size=(self.size, self.size)),
            dict(type='ToTensor'),
            dict(type='Normalize', mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, inplace=True),
        ]
        self.train_data.test_transforms = self.train_data.train_transforms
        self.train_data.target_transforms = [
            dict(type='Resize', size=(self.size, self.size), interpolation=F.InterpolationMode.BILINEAR),
            dict(type='CenterCrop', size=(self.size, self.size)),
            dict(type='ToTensor'),
        ]

        self.test_data.train_transforms = [
            dict(type='Resize', size=(self.size, self.size), interpolation=F.InterpolationMode.BILINEAR),
            dict(type='CenterCrop', size=(self.size, self.size)),
            dict(type='ToTensor'),
            dict(type='Normalize', mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, inplace=True),
        ]
        self.test_data.test_transforms = self.test_data.train_transforms
        self.test_data.target_transforms = [
            dict(type='Resize', size=(self.size, self.size), interpolation=F.InterpolationMode.BILINEAR),
            dict(type='CenterCrop', size=(self.size, self.size)),
            dict(type='ToTensor'),
        ]

        # ==> loss
        self.loss.loss_terms = [
            dict(type='DinomalyLoss', name='dinomaly_loss', p=0.9, factor=0.),
        ]

        # ==> logging
        self.logging.log_terms_train = [
            dict(name='batch_t', fmt=':>5.3f', add_name='avg'),
            dict(name='data_t', fmt=':>5.3f'),
            dict(name='optim_t', fmt=':>5.3f'),
            dict(name='lr', fmt=':>7.6f'),
            dict(name='dinomaly_loss', suffixes=[''], fmt=':>5.3f', add_name='avg'),
        ]
        self.logging.log_terms_test = [
            dict(name='batch_t', fmt=':>5.3f', add_name='avg'),
            dict(name='dinomaly_loss', suffixes=[''], fmt=':>5.3f', add_name='avg'),
        ]

