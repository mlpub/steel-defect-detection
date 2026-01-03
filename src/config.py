import os

# data folder path relative to src folder
DATA_FOLDER = '../data'  
# path train images
IMAGE_FOLDER = os.path.join(DATA_FOLDER, 'train_images')
# path to train csv with labels
TRAIN_CSV = os.path.join(DATA_FOLDER, 'train_with_bboxes.csv')

SEED = 42
BATCH_SIZE = 20
IMAGE_SIZE = 256
NUM_WORKERS = 4
PREFETCH_FACTOR = 4
LEARNING_RATE = 1e-4
EPOCHS = 2


TRAIN_SMALL_SIZE = 3500
VAL_SMALL_SIZE = 1500

VAL_METRICS = ['val_loss', 'val_acc', 'val_precision', 'val_recall', 'val_f1', 'val_rocauc']
VAL_CF = ['val_tn', 'val_fp', 'val_fn', 'val_tp']
