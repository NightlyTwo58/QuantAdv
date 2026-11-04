"""Central configuration for active QuantAdv experiments.

Constants in this module define dataset paths, model names, perturbation
budgets, quantization defaults, attack settings, optional experiment switches,
and report output locations.  Runtime modules import these values directly, so
changes here affect both the main runner and standalone analysis scripts.
"""

import logging
import os
from pathlib import Path

import torch
import torchvision.datasets as tv_datasets
import torchvision.transforms as tv_transforms

logging.getLogger("torch.utils._pytree").setLevel(logging.ERROR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_TYPE = device.type
USE_AMP = torch.cuda.is_available()

CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIG_DIR.parent
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)
PYTORCHCV_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "pytorchcv")

DATASET_NAME = "CIFAR-100"
DATASET_CLASS = tv_datasets.CIFAR100
DATASET_ROOT = PROJECT_ROOT
DATASET_DIR = os.path.join(DATASET_ROOT, "cifar-100-python")
# Default: False.
DATASET_DOWNLOAD = False
# Default: {'train': True}.
DATASET_TRAIN_KWARGS = {"train": True}
# Default: {'train': False}.
DATASET_TEST_KWARGS = {"train": False}
# Default: 100.
DATASET_NUM_CLASSES = 100
# Default: 3.
DATASET_INPUT_CHANNELS = 3
# Default: 32.
DATASET_IMAGE_SIZE = 32
# Default: 4.
DATASET_RANDOM_CROP_PADDING = 4
# Default: (0.5071, 0.4867, 0.4408).
DATASET_MEAN_VALUES = (0.5071, 0.4867, 0.4408)
# Default: (0.2675, 0.2565, 0.2761).
DATASET_STD_VALUES = (0.2675, 0.2565, 0.2761)
# Default: DATASET_MEAN_VALUES reshaped to NCHW broadcast form.
DATASET_MEAN = torch.tensor(DATASET_MEAN_VALUES).view(1, DATASET_INPUT_CHANNELS, 1, 1)
# Default: DATASET_STD_VALUES reshaped to NCHW broadcast form.
DATASET_STD = torch.tensor(DATASET_STD_VALUES).view(1, DATASET_INPUT_CHANNELS, 1, 1)
# Default: random crop/flip, tensor conversion, CIFAR-100 normalization.
DATASET_TRAIN_TRANSFORM = tv_transforms.Compose(
    [
        tv_transforms.RandomCrop(
            DATASET_IMAGE_SIZE, padding=DATASET_RANDOM_CROP_PADDING
        ),
        tv_transforms.RandomHorizontalFlip(),
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean=DATASET_MEAN_VALUES, std=DATASET_STD_VALUES),
    ]
)
# Default: tensor conversion and CIFAR-100 normalization.
DATASET_TEST_TRANSFORM = tv_transforms.Compose(
    [
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean=DATASET_MEAN_VALUES, std=DATASET_STD_VALUES),
    ]
)
# Default: normalized lower pixel bound.
CLIP_MIN = (0.0 - DATASET_MEAN) / DATASET_STD
# Default: normalized upper pixel bound.
CLIP_MAX = (1.0 - DATASET_MEAN) / DATASET_STD
# Default: CLIP_MIN.
CLIP_MIN_DEV = CLIP_MIN.to(device)
# Default: CLIP_MAX.
CLIP_MAX_DEV = CLIP_MAX.to(device)

RESULTS_CSV = os.path.join(DATA_DIR, "accuracyresult.csv")
PER_EXAMPLE_DIR = os.path.join(DATA_DIR, "per_example")
PERFORMANCE_CSV = os.path.join(DATA_DIR, "performance_metrics.csv")
CHAOTIC_DITHER_SWEEP_CSV = os.path.join(DATA_DIR, "chaotic_dither_sweep.csv")
SWEEP_CSV = os.path.join(DATA_DIR, "sweepresult.csv")
PLOT_PNG = os.path.join(DATA_DIR, "accuracyplot.png")
SWEEP_PLOT_PNG = os.path.join(DATA_DIR, "sweepplot.png")
ABLATION_PLOT_PNG = os.path.join(DATA_DIR, "ablationplot.png")
TRAJECTORY_PLOT_PNG = os.path.join(DATA_DIR, "trajectoryplot.png")
LAYERWISE_PLOT_PNG = os.path.join(DATA_DIR, "layerwiseplot.png")
COMPONENT_ABLATION_PLOT_PNG = os.path.join(DATA_DIR, "componentablationplot.png")
CHUNK_QUANT_PLOT_PNG = os.path.join(DATA_DIR, "chunkquantplot.png")
MASKING_SUMMARY_PLOT_PNG = os.path.join(DATA_DIR, "maskingsummaryplot.png")
MARGIN_PLOT_PNG = os.path.join(DATA_DIR, "marginplot.png")
HEATMAP_PLOT_PNG = os.path.join(DATA_DIR, "heatmapplot.png")
PERFORMANCE_PLOT_PNG = os.path.join(DATA_DIR, "performance_metrics.png")
DITHER_PLOT_PNG = os.path.join(DATA_DIR, "chaotic_dither_sweep.png")
DEFENSE_PLOT_PNG = os.path.join(DATA_DIR, "defense_comparison.png")
ABLATION_COMBINED_CSV = os.path.join(DATA_DIR, "ablation_combined.csv")
LAYERWISE_COMBINED_CSV = os.path.join(DATA_DIR, "layerwise_combined.csv")
TRAJECTORY_COMBINED_CSV = os.path.join(DATA_DIR, "trajectory_combined.csv")
COMPONENT_ABLATION_COMBINED_CSV = os.path.join( DATA_DIR, "component_ablation_combined.csv")
MARGIN_COMBINED_CSV = os.path.join(DATA_DIR, "margin_combined.csv")
CHUNK_COMBINED_CSV = os.path.join(DATA_DIR, "chunk_quant_combined.csv")
DEFENSE_SUMMARY_CSV = os.path.join(DATA_DIR, "defense_summary.csv")

# Default: [0, 1, 2, 3, 4].
SEEDS = [0, 1, 2, 3, 4]

PRETRAINED_NAMES = {
    "ResNet56_C100": "resnet56_cifar100",
    "WRN28_10_C100": "wrn28_10_cifar100",
    "DenseNet100_C100": "densenet100_k12_bc_cifar100",
}
# Default: pretrained weights stored under models/pytorchcv.
PRETRAINED_MODEL_KWARGS = {
    "pretrained": True,
    "root": PYTORCHCV_MODEL_DIR,
}

# Default: all 10 variants supported by the model builder.
SUPPORTED_MODEL_VARIANTS = (
    "FP32",
    "int8_PTQ",
    "int4_PTQ",
    "int8_QAT",
    "int4_QAT",
    "FP32_Compressed",
    "chaotic_int8_PTQ",
    "chaotic_int4_PTQ",
    "chaotic_int8_PTQ_Compressed",
    "chaotic_int8_QAT",
)
# Default: FP32, int8 PTQ, and int8 QAT.
MODEL_VARIANTS = ("FP32", "int8_PTQ", "int8_QAT")

# Default: False.
QUANTIZATION_DEBUG_ONLY = False

# Default: 512.
DEFAULT_BATCH_SIZE = 512
# Default: 1000.
DEFAULT_EVAL_N = 1000
# Default: 10000.
DEFAULT_FINETUNE_N = 10000
# Default: 256.
DEFAULT_EVAL_BATCH_SIZE = 256
# Default: 8.
MAX_DATA_WORKERS = 8
# Default: True.
PIN_MEMORY = True
# Default: PIN_MEMORY and device.type == 'cuda'.
NON_BLOCKING_TRANSFER = PIN_MEMORY and device.type == "cuda"

# Default: True.
TRAIN_SHUFFLE = True
# Default: False.
EVAL_SHUFFLE = False

# Default: 1e-08.
QUANT_SCALE_MIN = 1e-8
# Default: False.
QUANT_DEFAULT_USE_STE = False
# Default: True.
QUANT_DEFAULT_WEIGHT = True
# Default: True.
QUANT_DEFAULT_ACT = True
# Default: 'logistic'.
CHAOTIC_QUANT_MAP = "logistic"
# Default: 0.731.
CHAOTIC_QUANT_SEED = 0.731
# Default: 3.99.
CHAOTIC_QUANT_R = 3.99
# Default: 1.999.
CHAOTIC_QUANT_MU = 1.999
# Default: 0.5.
CHAOTIC_QUANT_DITHER = 0.5
# Default: (0.0, 0.125, 0.25, 0.5, 0.75, 1.0).
CHAOTIC_DITHER_AMPLITUDES = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)
# Default: 17.
CHAOTIC_QUANT_WARMUP = 17
# Default: 24.
COMPRESS_IMAGE_SIZE = 24
# Default: 5.
COMPRESS_IMAGE_BITS = 5
# Default: 'bilinear'.
COMPRESS_IMAGE_MODE = "bilinear"
# Default: False.
COMPRESS_IMAGE_ALIGN_CORNERS = False

# Default: 32.
SUBSTITUTE_CONV1_CHANNELS = 32
# Default: 64.
SUBSTITUTE_CONV2_CHANNELS = 64
# Default: 3.
SUBSTITUTE_KERNEL_SIZE = 3
# Default: 1.
SUBSTITUTE_CONV_PADDING = 1
# Default: 2.
SUBSTITUTE_POOL_KERNEL = 2
# Default: 8.
SUBSTITUTE_LINEAR_FEATURE_MAP = 8
# Default: 256.
SUBSTITUTE_HIDDEN_DIM = 256
# Default: True.
SUBSTITUTE_RELU_INPLACE = True

# Default: 0.6.
PLOT_GRID_ALPHA = 0.6
# Default: 0.7.
SUMMARY_GRID_ALPHA = 0.7
# Default: 8.
PLOT_LEGEND_FONT_SIZE = 8
# Default: 90.
LAYERWISE_XTICK_ROTATION = 90
# Default: 6.
LAYERWISE_XTICK_FONT_SIZE = 6
# Default: 45.
SUMMARY_XTICK_ROTATION = 45
# Default: 0.8.
MASKING_BASELINE_LINEWIDTH = 0.8
# Default: 0.
HEATMAP_VMIN = 0
# Default: 1.
HEATMAP_VMAX = 1
# Default: 0.5.
HEATMAP_LINEWIDTHS = 0.5

# Default: 8 / 255.
DEFAULT_EPS = 8 / 255
# Default: 2 / 255.
PGD_ALPHA = 2 / 255
# Default: 20.
PGD_STEPS = 20
# Default: True.
PGD_RANDOM_START = True

# Default: 3.
QAT_EPOCHS_DEFAULT = 3
# Default: 10.
QAT_MAIN_EPOCHS = 10
# Default: 0.001.
QAT_LR = 1e-3
# Default: 0.9.
QAT_MOMENTUM = 0.9
# Default: 0.0005.
QAT_WEIGHT_DECAY = 5e-4
# Default: 5.
QAT_LOG_EVERY_BATCHES = 5

# Default: (8, 4).
PTQ_BITS = (8, 4)

# Default: 'Linf'.
AUTOATTACK_NORM = "Linf"
# Default: 'standard'.
AUTOATTACK_VERSION = "standard"
# Default: 0.
AUTOATTACK_SEED = 0
# Default: True.
AUTOATTACK_VERBOSE = True

# Default: 1.
CW_C = 1
# Default: 0.
CW_KAPPA = 0
# Default: 50.
CW_STEPS = 50
# Default: 0.01.
CW_LR = 0.01

# Default: 50.
DEEPFOOL_STEPS = 50
# Default: 0.02.
DEEPFOOL_OVERSHOOT = 0.02
# Default: 1.0.
JSMA_THETA = 1.0
# Default: 0.1.
JSMA_GAMMA = 0.1
# Default: 20.
JSMA_MAX_IMAGES = 20

# Default: 1.0.
MIFGSM_DECAY = 1.0

# Default: 0.2.
UAP_DELTA = 0.2
# Default: 10.
UAP_MAX_ITER = 10
# Default: 20.
UAP_DEEPFOOL_STEPS = 20
# Default: 0.02.
UAP_OVERSHOOT = 0.02
# Default: 1000.
UAP_MAX_IMAGES = 1000

# Default: 1.
BPDA_RESTARTS_DEFAULT = 1
# Default: 5.
BPDA_RESTARTS_SUITE = 5
# Default: 2.
BPDA_RESTARTS_SWEEP = 2

# Default: 4.
ADAPTIVE_EOT_SAMPLES = 4
# Default: 1.0.
ADAPTIVE_GUARDRAIL_LAMBDA = 1.0
# Default: 1.0.
ADAPTIVE_DETECTOR_LAMBDA = 1.0
# Default: 20.0.
ADAPTIVE_GUARDRAIL_SCALE = 20.0

# Default: 20.
NES_SAMPLES_DEFAULT = 20
# Default: 100.
NES_SAMPLES_SUITE = 100
# Default: 0.001.
NES_SIGMA = 1e-3
# Default: 10.
NES_STEPS = 10
# Default: 512.
NES_QUERY_CHUNK = 512

# Default: False.
RUN_EXTRA_WHITEBOX_ATTACKS = False
# Default: False.
RUN_UAP_ATTACKS = False
# Default: False.
RUN_SURROGATE_ATTACK = False
# Default: False.
RUN_REVERSE_TRANSFERS = False
# Default: False.
RUN_BOUNDARY_ATTACK = False
# Default: False.
RUN_NES_ATTACK = False
# Default: False.
RUN_DEFENSE_SUITE = False
# Default: False.
RUN_CHUNK_QUANTIZATION = False
# Default: False.
RUN_PGD_TRAJECTORY = False
# Default: False.
RUN_LAYERWISE_PROFILE = False
# Default: False.
RUN_CONFIDENCE_MARGIN = False
# Default: False.
RUN_COMPONENT_ABLATION = False
# Default: False.
RUN_EPSILON_SWEEP = False
# Default: False.
RUN_CHAOTIC_DITHER_SWEEP = False
# Default: True.
RECORD_RUN_METRICS = True

# Default: 0.95.
CI_CONFIDENCE = 0.95

# Default: DATASET_NUM_CLASSES.
SUBSTITUTE_NUM_CLASSES = DATASET_NUM_CLASSES
# Default: 6.
SUBSTITUTE_ROUNDS = 6
# Default: 10.
SUBSTITUTE_EPOCHS_PER_ROUND = 10
# Default: 0.001.
SUBSTITUTE_LR = 1e-3
# Default: 0.1.
SUBSTITUTE_LAMBDA = 0.1
# Default: 128.
SUBSTITUTE_BATCH_SIZE = 128
# Default: 500.
SURROGATE_SEED_N = 500

# Default: 50.
BOUNDARY_MAX_IMAGES_DEFAULT = 50
# Default: 30.
BOUNDARY_MAX_IMAGES_SUITE = 30
# Default: 200.
BOUNDARY_STEPS_DEFAULT = 200
# Default: 500.
BOUNDARY_STEPS_SUITE = 500
# Default: 0.
BOUNDARY_SEED = 0
# Default: 0.01.
BOUNDARY_SPHERICAL_STEP = 1e-2
# Default: 0.01.
BOUNDARY_SOURCE_STEP = 1e-2
# Default: 1.5.
BOUNDARY_STEP_ADAPT = 1.5
# Default: 200.
BOUNDARY_INIT_TRIES = 200
# Default: 25.
BOUNDARY_INIT_CHUNK = 25
# Default: 1e-12.
BOUNDARY_MIN_DIST = 1e-12
# Default: 10.
BOUNDARY_ADAPT_INTERVAL = 10
# Default: 0.5.
BOUNDARY_SPH_SUCCESS_HIGH = 0.5
# Default: 0.2.
BOUNDARY_SPH_SUCCESS_LOW = 0.2

# Default: 3.
GRAD_DIAG_MAX_BATCHES = 3
# Default: 1e-08.
GRAD_ZERO_THRESHOLD = 1e-8
# Default: (0, 1, 2, 3, 4, 5, 10).
PGD_ABLATION_STEPS = (0, 1, 2, 3, 4, 5, 10)
# Default: ().
PGD_ABLATION_VARIANTS = ()
# Default: 3.
TRAJECTORY_MAX_BATCHES = 3
# Default: 3.
LAYERWISE_MAX_BATCHES = 3
# Default: 1 / 255.
STAIRCASE_RADIUS = 1 / 255
# Default: 40.
STAIRCASE_N_POINTS = 40
# Default: 3.
MARGIN_MAX_BATCHES = 3
# Default: 20.
MARGIN_STEPS = 20

# Default: 9 budgets from 0 through 16/255.
SWEEP_EPSILONS = [
    0,
    1 / 255,
    2 / 255,
    3 / 255,
    4 / 255,
    6 / 255,
    8 / 255,
    12 / 255,
    16 / 255,
]

# Default: 300.
PLOT_DPI = 300
# Default: 'tight'.
PLOT_BBOX_INCHES = "tight"
# Default: 1.0.
PLOT_MAX_ACCURACY = 1.0
# Default: 3.
SWEEP_PLOT_COLS_MAX = 3
# Default: 6.
SWEEP_PLOT_WIDTH = 6
# Default: 4.5.
SWEEP_PLOT_HEIGHT = 4.5
# Default: (10, 6).
ABLATION_FIGSIZE = (10, 6)
# Default: (14, 5).
TRAJECTORY_FIGSIZE = (14, 5)
# Default: 2.
LAYERWISE_PLOT_COLS_MAX = 2
# Default: 9.
LAYERWISE_PLOT_WIDTH = 9
# Default: 4.5.
LAYERWISE_PLOT_HEIGHT = 4.5
# Default: 3.
COMPONENT_ABLATION_COL_WRAP = 3
# Default: 4.
COMPONENT_ABLATION_HEIGHT = 4
# Default: 4.
CHUNK_QUANT_NUM_CHUNKS = 4
# Default: 2.
CHUNK_QUANT_COL_WRAP = 2
# Default: 4.
CHUNK_QUANT_HEIGHT = 4
# Default: (14, 5).
MASKING_SUMMARY_FIGSIZE = (14, 5)
# Default: 3.
MARGIN_PLOT_COLS_MAX = 3
# Default: 6.
MARGIN_PLOT_WIDTH = 6
# Default: 4.
MARGIN_PLOT_HEIGHT = 4
# Default: 30.
MARGIN_HIST_BINS = 30
# Default: 0.6.
MARGIN_HIST_ALPHA = 0.6
# Default: 10.
HEATMAP_MIN_WIDTH = 10
# Default: 6.
HEATMAP_MIN_HEIGHT = 6
# Default: 0.5.
HEATMAP_ROW_HEIGHT = 0.5
# Default: (14, 6).
SUMMARY_PLOT_FIGSIZE = (14, 6)

# Default: fixed colors for attack and diagnostic metrics.
ATTACK_PALETTE = {
    "clean_acc": "#4D4D4D",
    "FGSM": "#1F77B4",
    "PGD": "#D62728",
    "CW": "#9467BD",
    "DeepFool": "#8C564B",
    "JSMA": "#E377C2",
    "AutoAttack": "#000000",
    "Transfer_from_FP32": "#17BECF",
    "MIM_Transfer": "#2CA02C",
    "UAP_Transfer": "#BCBD22",
    "Transfer_to_FP32": "#00A6D6",
    "MIM_Transfer_to_FP32": "#228B22",
    "UAP_Transfer_to_FP32": "#9ACD32",
    "Surrogate_Transfer": "#FF7F0E",
    "Random_Noise": "#7F7F7F",
    "BPDA_PGD": "#B22222",
    "BPDA_Adaptive": "#FF9896",
    "EOT_PGD": "#C5B0D5",
    "Adaptive_Guardrail": "#AEC7E8",
    "Adaptive_DetectGuard": "#FDBF6F",
    "NES": "#6A3D9A",
    "Boundary_acc": "#A65628",
    "PGD_acc": "#D62728",
    "Random_Noise_acc": "#7F7F7F",
    "BPDA_acc": "#B22222",
    "clean": "#4D4D4D",
    "PGD-adv": "#D62728",
    "hard-round": "#D62728",
    "STE": "#1F77B4",
}

# Default: 'RdYlGn'.
HEATMAP_CMAP = "RdYlGn"
# Default: 'black'.
BASELINE_COLOR = "black"
# Default: 12.
REPORT_METRIC_PAGE_SIZE = 12
# Default: 60.
REPORT_MAX_HEATMAP_ROWS = 60
# Default: 7.0.
REPORT_MIN_FIGURE_WIDTH = 7.0
# Default: 18.0.
REPORT_MAX_FIGURE_HEIGHT = 18.0
# Default: 0.35.
REPORT_ROW_HEIGHT = 0.35
