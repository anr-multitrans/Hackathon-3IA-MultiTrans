from easydict import EasyDict as edict


__C                                             = edict()
# Consumers can get config by: from lstm_config import cfg

cfg                                             = __C

# DATASET options
__C.DATASET                                     = edict()

__C.DATASET.TRAIN_DATA_FOLDER_PATH              = "./../data/train"
__C.DATASET.VAL_DATA_FOLDER_PATH                = "./../data/valid"
__C.DATASET.TRAIN_PROPORTION                    = 0.9
__C.DATASET.TRAINING_BATCH_SIZE                 = 16
__C.DATASET.VALIDATION_BATCH_SIZE               = 16
__C.DATASET.IMAGE_FOLDER_PATH                   = "./../data/try_5/images/"
__C.DATASET.DEPTH_FOLDER_PATH                   = "./../data/try_5/depths/"  
__C.DATASET.COMMAND_FILE_PATH                   = "./../data/try_5/commands.json"  
__C.DATASET.LIDAR_FILE_PATH                     = "./../data/try_5/lidars.json"  
__C.DATASET.GOLD_DATA_PATH                      = "./../data/try_5/data_try5.bag"
__C.DATASET.USED_DATA_FOLDER                    = ['./../data/try_2', './../data/try_4']
__C.DATASET.MAX_SPEED                           = 2

# TRAIN options
__C.TRAIN                                       = edict()

__C.TRAIN.LEARNING_RATE                         = 0.002
__C.TRAIN.NBR_EPOCH                             = 50
__C.TRAIN.CHECKPOINT_SAVE_PATH                  = './../models/try_6/'
__C.TRAIN.VALIDATION_RATIO                      = 2
__C.TRAIN.GRADIANT_ACCUMULATION                 = 4
__C.TRAIN.IMAGE_SHAPE                           = (3,100,424)
__C.TRAIN.PRETRAINED_WEIGHTS_PATH               = "./best_model.pth"
__C.TRAIN.SEQUENCE_SIZE                         = 20

# EVALUATION options
__C.EVALUATION                                  = edict()

__C.EVALUATION.PRETRAINED_PATH                  = './../models/try_1/ckpt_19_metric_0.62096.ckpt'