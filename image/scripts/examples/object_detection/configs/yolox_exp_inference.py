from yolox.exp import Exp as MyExp

MODEL_TYPE = 'YOLOX-s'  # Please change the model name(See https://github.com/Megvii-BaseDetection/YOLOX/tree/main?tab=readme-ov-file#benchmark)
NUM_CLASSES = 71  # Please input the number of classes
SEED = None  # Random seed. If you specify the seed, it can slow down your training considerably. To avoid this, you should use None
CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45

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
        # ---------------- model config ---------------- #
        self.num_classes = NUM_CLASSES
        self.depth = MODEL_PARAMS[MODEL_TYPE]['depth']
        self.width = MODEL_PARAMS[MODEL_TYPE]['width']
        # ---------------- dataloader config ---------------- #
        # --------------- transform config ----------------- #
        # --------------  training config --------------------- #
        # -----------------  testing config ------------------ #
        self.test_conf = CONF_THRESHOLD
        self.nmsthre = NMS_THRESHOLD
        if SEED is not None:
            self.seed = SEED
