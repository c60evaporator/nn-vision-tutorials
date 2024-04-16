#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
from datetime import datetime

from yolox.exp import Exp as MyExp

NUM_EPOCHS = 300  # number of epochs
OUTPUT_DIR = '/scripts/examples/object_detection/results/yolox' # Training result output dir
MODEL_TYPE = 'YOLOX-s'  # Please change the model name(See https://github.com/Megvii-BaseDetection/YOLOX/tree/main?tab=readme-ov-file#benchmark)
NUM_CLASSES = 71  # Please input the number of classes
SEED = None  # Random seed. If you specify the seed, it can slow down your training considerably. To avoid this, you should use None
DATA_DIR = '/scripts/examples/object_detection/datasets/mini-coco128'
TRAIN_DIR = 'train2017'  # Train directory name
VAL_DIR = 'val2017'  # Validation directory name
TRAIN_ANN = 'instances_train2017.json'  # Train annotation file name
VAL_ANN = 'instances_val2017.json'  # Validation annotation file name
TEST_ANN = '/'
EXP_NAME = f'{datetime.now().strftime("%Y%m%d%H%M%S")}_{os.path.basename(DATA_DIR)}_{MODEL_TYPE}'
FREEZE_BACKBONE = True

MODEL_PARAMS = {  # Please modify the parameters based on the current YOLOX
    'YOLOX-Nano': {
        'depth': 0.33,
        'width': 0.25
    },
    'YOLOX-Tiny': {
        'depth': 0.33,
        'width': 0.375
    },
    'YOLOX-s': {
        'depth': 0.33,
        'width': 0.5
    },
    'YOLOX-m': {
        'depth': 0.67,
        'width': 0.75
    },
    'YOLOX-l': {
        'depth': 1.0,
        'width': 1.0
    },
    'YOLOX-x': {
        'depth': 1.33,
        'width': 1.25
    },
    'YOLOX-Darknet53': {
        'depth': 1.0,
        'width': 1.0
    }
}


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.output_dir = OUTPUT_DIR
        # ---------------- model config ---------------- #
        self.num_classes = NUM_CLASSES
        self.depth = MODEL_PARAMS[MODEL_TYPE]['depth']
        self.width = MODEL_PARAMS[MODEL_TYPE]['width']
        # ---------------- dataloader config ---------------- #
        self.data_dir = DATA_DIR
        self.train_ann = TRAIN_ANN
        self.val_ann = VAL_ANN
        self.test_ann = TEST_ANN
        # --------------- transform config ----------------- #
        # --------------  training config --------------------- #
        self.max_epoch = NUM_EPOCHS
        self.exp_name = EXP_NAME
        # -----------------  testing config ------------------ #
        if SEED is not None:
            self.seed = SEED
    
    def get_model(self):
        from yolox.utils import freeze_module
        model = super().get_model()
        if FREEZE_BACKBONE:
            print('The backbone is freezed')
            freeze_module(model.backbone)
        return model
    
    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        """
        Get dataset according to cache and cache_type parameters.
        Args:
            cache (bool): Whether to cache imgs to ram or disk.
            cache_type (str, optional): Defaults to "ram".
                "ram" : Caching imgs to ram for fast training.
                "disk": Caching imgs to disk for fast training.
        """
        from yolox.data import COCODataset, TrainTransform

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name=TRAIN_DIR,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob
            ),
            cache=cache,
            cache_type=cache_type,
        )
    
    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform
        testdev = kwargs.get("testdev", False)
        legacy = kwargs.get("legacy", False)

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann if not testdev else self.test_ann,
            name=VAL_DIR,
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )
