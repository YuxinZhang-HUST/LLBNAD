import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import argparse
from configs import get_cfg
from util.net import init_training
from util.util import run_pre, init_checkpoint
from trainer import get_trainer
import warnings
warnings.filterwarnings("ignore")

### I have condensed the original ADer repo a lot and improve some parts for easier developments, but I only implement two methods, and two datasets currently.
### More baselinea are encouraged to be added. Firstly, the remaining baselines in ADer can be a good start for practicing.


### TODO: the saving habits for images, logs, tensorboards, ckpts, etc., need to be further adjusted according to our final goal...
### TODO: add more anomaly synthetic strategies to ./data -- CDO, Draem, etc.
### TODO: think about the dataset file for our dataset -- need to including different settings, and should also adapt the training/testing file
### TODO: to be compatible to the multi-view/illumination setup
### TODO: prepare scripts for ease of analyzing results -- for csvs, and visualizations. Just adapt my original scripts according to the new dir organizations

### Very important -- especially for the junior students
### Note: to add a new method, you need to follow the following steps:
### 1. add a model file to ./model, refer to rd.py
### 2.check if there are any specific functions needed to be defined, e.g.,
###### -- data-related should be put into ./data
###### -- loss-related should be put into ./loss
### 3.add the corresponding trainer to ./trainer, refer to ./trainer/rd_trainer.py. there are very few functions needed to be defined.
### 4.add a specific config file to ./benchmark/method, refer to configs/benchmark/devnet_256_100e.py
### 5.extract the default method config from (4) and write a default config file to ./configs/__base__/cfg_model_{method}.py
### for ease of reutilization...

### Note: You are encouraged to write detailed and official comments when reading the code. If you are not able to be official enough,
### you can take the assistance of ChatGPT. With detailed comments, in the future, we can simply write a detailed document for our repo.



def main():
	parser = argparse.ArgumentParser()

	### unsupervised
	parser.add_argument('-c', '--cfg_path', default='configs/benchmark/dinomaly/dinomaly_f20_t40_num5_iter1000.py')
	parser.add_argument('-m', '--mode', default='train', choices=['train', 'test'])
	parser.add_argument('--sleep', type=int, default=-1)
	parser.add_argument('--memory', type=int, default=-1)
	parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training')
	parser.add_argument('--logger_rank', default=0, type=int, help='GPU id to use.')
	parser.add_argument('opts', help='path.key=value', default=None, nargs=argparse.REMAINDER,)
	# parser.add_argument('--subset_num', type=int, default=5)
	# parser.add_argument('--noisy_ratio', type=float, default=0.1)

	cfg_terminal = parser.parse_args()
	cfg = get_cfg(cfg_terminal)
	run_pre(cfg)
	init_training(cfg)
	init_checkpoint(cfg)
	trainer = get_trainer(cfg)
	trainer.run()

	del trainer

if __name__ == '__main__':
	import torch
	# torch.autograd.set_detect_anomaly(True)
	main()
