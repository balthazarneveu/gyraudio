# Training modes (Train, Validation, Test)
TRAIN = "train"
VALID = "validation"
TEST = "test"

# Dataset properties (keys)
DATA_PATH = "path"
BATCH_SIZE = "batch_size"
SHUFFLE = "shuffle"
SNR_FILTER = "snr_filter"
AUGMENTATION = "augmentation"

# Training configuration properties (keys)

OPTIMIZER = "optimizer"
LEARNING_RATE = "lr"
EPOCHS = "epochs"
BATCH_SIZE = "batch_size"
MAX_STEPS_PER_EPOCH = "max_steps_per_epoch"


# Properties for the model
NAME = "name"
ANNOTATIONS = "annotations"
NB_PARAMS = "nb_params"
SHORT_NAME = "short_name"


# Loss
LOSS = "loss"
LOSS_L2 = "MSE"

# Checkpoint
MODEL = "model"
CURRENT_EPOCH = "current_epoch"
CONFIGURATION = "configuration"


# Signal names
CLEAN = "clean"
NOISY = "noise"
MIXED = "mixed"
