import os
import json
import pandas as pd
import random
from ..dataset_info import *


class RealCaseSolver(object):
    CLSNAMES = CLASS_NAMES['real_case']

    def __init__(self, root=get_data_root('real_case'), train_normal_ratio=0.8):
        self.root = root
        self.meta_path = f'{root}/meta.json'
        self.phases = ['train', 'test']
        self.train_normal_ratio = train_normal_ratio

    def run(self):
        info = self.generate_meta_info()
        split_meta(info, self.root)

    def generate_meta_info(self):
        info = {phase: {} for phase in self.phases}
        for cls_name in self.CLSNAMES:
            normal_paths = os.listdir(os.path.join(self.root, cls_name, 'normal'))
            abnormal_paths = os.listdir(os.path.join(self.root, cls_name, 'abnormal'))
            mask_paths = os.listdir(os.path.join(self.root, cls_name, 'mask'))

            random.shuffle(normal_paths)

            # 计算分割点
            split_point = int(len(normal_paths) * self.train_normal_ratio)

            # 分割列表
            train_normal_paths = normal_paths[:split_point]
            test_normal_paths = normal_paths[split_point:]

            # train phase
            cls_info = []

            for train_normal_path in train_normal_paths:
                info_img = dict(
                    img_path=f'{cls_name}/normal/{train_normal_path}',
                    mask_path='',
                    cls_name=cls_name,
                    specie_name='',
                    anomaly=0,
                )
                cls_info.append(info_img)

            info['train'][cls_name] = cls_info

            # test phase
            # normal
            cls_info = []

            for test_normal_path in test_normal_paths:
                info_img = dict(
                    img_path=f'{cls_name}/normal/{test_normal_path}',
                    mask_path='',
                    cls_name=cls_name,
                    specie_name='',
                    anomaly=0,
                )
                cls_info.append(info_img)

            # anomaly
            for abnormal_path, mask_path in zip(abnormal_paths, mask_paths):
                info_img = dict(
                    img_path=f'{cls_name}/abnormal/{abnormal_path}',
                    mask_path=f'{cls_name}/mask/{mask_path}',
                    cls_name=cls_name,
                    specie_name='',
                    anomaly=1,
                )
                cls_info.append(info_img)

            info['test'][cls_name] = cls_info

        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")

        return info
