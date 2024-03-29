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
DATALOADER = "dataloader"


# Loss split
SIGNAL = "signal"
NOISE = "noise"
TOTAL = "total"
COEFFICIENT = "coefficient"


# Augmentation types
AUG_TRIM = "trim"  # trim batches to arbitrary length
AUG_AWGN = "awgn"  # add white gaussian noise
AUG_RESCALE = "rescale"  # rescale all signals arbitrarily

# Trim types
LENGTHS = "lengths"  # a list of min and max length
LENGTH_DIVIDER = "length_divider"  # an int that divides the length
TRIM_PROB = "trim_probability"  # a float in [0, 1] of trimming probability


# Training configuration properties (keys)

OPTIMIZER = "optimizer"
LEARNING_RATE = "lr"
WEIGHT_DECAY = "weight_decay"
BETAS = "betas"
EPOCHS = "epochs"
BATCH_SIZE = "batch_size"
MAX_STEPS_PER_EPOCH = "max_steps_per_epoch"


# Properties for the model
NAME = "name"
ANNOTATIONS = "annotations"
NB_PARAMS = "nb_params"
RECEPTIVE_FIELD = "receptive_field"
SHORT_NAME = "short_name"


# Scheduler
SCHEDULER = "scheduler"
SCHEDULER_CONFIGURATION = "scheduler_configuration"

# Loss
LOSS = "loss"
LOSS_L1 = "L1"
LOSS_L2 = "MSE"
LOSS_TYPE = "loss_type"
SNR = "snr"

# Checkpoint
MODEL = "model"
CURRENT_EPOCH = "current_epoch"
CONFIGURATION = "configuration"


# Signal names
CLEAN = "clean"
NOISY = "noise"
MIXED = "mixed"
PREDICTED = "predicted"


# MISC
PATHS = "paths"
BUFFERS = "buffers"
SAMPLING_RATE = "sampling_rate"
