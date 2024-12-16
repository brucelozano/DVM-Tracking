# Path to the CSV file
FILE_PATH = "/Users/bruce/Documents/Research/Exported Sv CSV/B082D2_DVM_Focus.sv.csv"

# Visualization parameters
dmin = -74 # Display minimum value from Echoview
dmax = None # Display maximum where dmax = dmin + drange
drange = 15 # Display range from Echoview

# Preprocessing parameters
db_min = -75
db_max = -65

# Preprocessing parameters
TOTAL_DEPTH = 2604
GAUSSIAN_SIGMA = (4, 8)
DETAIL_WINDOW_SIZE = 11
BASE_THRESHOLD = 0.3
STRONG_SIGNAL_THRESHOLD = 0.6
MEDIUM_SIGNAL_THRESHOLD = 0.3
WEAK_SIGNAL_THRESHOLD = 0.15
THRESHOLD_MULTIPLIERS = {
    'strong': 0.9,
    'medium': 0.65,
    'weak': 0.5
}
THRESHOLD_SMOOTHING_SIGMA = 3

# Layer enhancement parameters
BASE_KERNEL_SIZE = 31
VERTICAL_KERNEL_SIZE = 5
HORIZONTAL_KERNEL_MULTIPLIER = 3

# Detection parameters
CHUNK_SIZE = 100000
MIGRATION_WINDOW_SIZE = 50
MIGRATION_WEIGHT_MULTIPLIER = 0.5

# HDBSCAN parameters
MIN_CLUSTER_SIZE_FACTOR = 0.02
MIN_SAMPLES = 4
CLUSTER_SELECTION_EPSILON = 0.025
CLUSTER_ALPHA = 0.8

# Layer merging parameters
TIME_THRESHOLD = 15
DEPTH_DIFF_THRESHOLD = 20
RATE_SIMILARITY_THRESHOLD = 0.1

# DVM Detection Parameters
DVM_VMIN = -75  # Adjusted to match db_min
DVM_VMAX = -65  # Adjusted to match db_max
DVM_THRESHOLD = 0.4  # Increased slightly for more precise detection
DVM_WINDOW_SIZE = 20  # Decreased for finer temporal resolution

# Migration Phase Parameters
DAWN_MIGRATION_HOURS = 2  # Hours to consider as dawn migration period
DUSK_MIGRATION_HOURS = 2  # Hours to consider as dusk migration period

# Visualization Parameters for DVM
DVM_PLOT_ALPHA = 0.7  # Alpha value for migration depth line
PHASE_PLOT_ALPHA = 0.2  # Alpha value for phase background
