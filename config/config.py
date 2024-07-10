from easydict import EasyDict as edict

config = edict()
config.NUM_WORKERS = 1
config.OUTPUT_DIR = './experiments/'
config.SEED = 12345

config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = False    # True leads to unexpected errors and slows training and inference  

config.DATASET = edict()
config.DATASET.ROOT = './dataset/data/proceed/'

config.SPLIT = edict()
config.SPLIT.ROOT = './dataset/data'

config.SPLIT_eva = edict()
config.SPLIT_eva.ROOT = './dataset/data'

config.DATASET_eva = edict()
config.DATASET_eva.ROOT = './dataset/data/proceed'


config.MODEL = edict()
config.MODEL.STAGES = 5
config.MODEL.MIDCHANNEL = 16


config.TRAIN = edict()
config.TRAIN.LR = 1e-2
config.TRAIN.LAMDA = 0.015
config.TRAIN.WEIGHT_DECAY = 3e-5
config.TRAIN.BATCH_SIZE = 32
config.TRAIN.PATCH_SIZE = [224, 224]
config.TRAIN.NUM_BATCHES = 250
config.TRAIN.PARALLEL = True
config.TRAIN.DEVICES = [0]