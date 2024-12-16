# Diel Vertical Migration (DVM) Detection

## Overview
This Python-based program processes Echoview Sv CSV data to detect and analyze Deep Scattering Layers (DSL) and Diel Vertical Migration (DVM) patterns in marine environments. It uses various image processing and machine learning techniques to identify and track these biological phenomena.

## System Architecture
The system consists of four main Python files that work together:

### 1. echogram_processing.py (Main Processing Pipeline)
- **Primary Functions:**
  - `load_and_preprocess_echogram`: Loads CSV data from Echoview and handles initial preprocessing
  - `prepare_sample_data`: Converts data into format suitable for analysis
  - `process_echogram_with_dsl_detection`: Main pipeline coordinating DSL and DVM analysis
- **Role:** Acts as the main entry point and coordinates the overall analysis workflow

### 2. dsl_detection.py (Layer Detection)
- **Primary Functions:**
  - `preprocess_image`: Initial image processing and noise reduction
  - `detect_layers`: HDBSCAN-based clustering for layer detection
  - `merge_nearby_layers`: Intelligent layer merging based on migration patterns
  
- **Key Features:**
  - Uses HDBSCAN clustering for robust layer detection
  - Implements migration rate calculation with quality assessment
  - Handles temporal connectivity and layer merging

### 3. dvm_detection.py (Migration Analysis)
- **Primary Functions:**
  - `detect_dvm_pattern`: Identifies vertical migration patterns
  - `analyze_dvm`: Main DVM analysis pipeline
  - `get_sun_times`: Calculates sunrise/sunset for migration phase analysis
  
- **Key Features:**
  - Integrates with astronomical data for migration timing
  - Tracks vertical movement patterns
  - Classifies migration phases

### 4. params.py (Configuration)
- Contains all configurable parameters for the system
- Organized into logical groups:
  - Display parameters (dB ranges)
  - Processing parameters
  - Detection thresholds
  - Clustering parameters

## Workflow

1. **Data Input**
   ```python
   # Load and preprocess the echogram
   echogram, fixed_column_names, start_ping, end_ping = load_and_preprocess_echogram(FILE_PATH)
   ```

2. **Data Preparation**
   ```python
   # Prepare the sample data
   sample_data, image_data = prepare_sample_data(echogram)
   ```

3. **DSL Detection**
   ```python
   # Process the echogram with DSL detection
   dsl_results = analyze_dsl(image_data, ping_time_map, depth_start, depth_stop, total_depth)
   ```

4. **DVM Analysis**
   ```python
   # Analyze DVM patterns
   dvm_results = analyze_dvm(image_data, ping_time_map, depth_start, depth_stop, lat, lon)
   ```

5. **Combined Analysis**
   ```python
   # Combine DSL and DVM results
   combined_results = combine_dsl_dvm_analysis(dsl_results, dvm_results)
   ```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/brucelozano/dsl-dvm-detection.git
   cd dsl-dvm-detection
   ```

2. Install required packages:
   ```bash
   pip install numpy pandas opencv-python matplotlib hdbscan scipy astral
   ```

## Usage Guide

### 1. Data Preparation
- Export Sv data from Echoview as CSV
- Ensure the CSV contains required columns:
  - Ping_index, Distance_gps, Distance_vl
  - Ping_date, Ping_time, Ping_milliseconds
  - Latitude, Longitude
  - Depth_start, Depth_stop
  - Sample columns

### 2. Configuration
1. Update `params.py` with appropriate values:
   ```python
   # Set path to your CSV file
   FILE_PATH = "path/to/your/data.sv.csv"
   
   # Adjust display parameters
   db_min = -75  # Minimum dB threshold
   db_max = -65  # Maximum dB threshold
   ```

2. Key parameters to consider:
   - `MIGRATION_WINDOW_SIZE`: Controls temporal resolution
   - `MIN_CLUSTER_SIZE_FACTOR`: Affects layer detection sensitivity
   - `RATE_SIMILARITY_THRESHOLD`: Influences layer merging

### 3. Running the Analysis
```python
from echogram_processing import process_echogram_with_dsl_detection

# Run the complete analysis
results = process_echogram_with_dsl_detection()
```

## Output and Visualization

The system generates several output files in the specified directories:

### 1. Main Visualizations
- `/Figures/`:
  - `echogram.png`: Raw echogram visualization
  - `detected_layers.png`: DSL detection results
  - `dvm_analysis.png`: DVM pattern analysis
  - `combined_analysis.png`: Combined DSL/DVM visualization

### 2. Debug Output
- `/Debug Output/`:
  - `debug_1_cropped.png` to `debug_11_merged_layers.png`: Step-by-step processing visualization

## Parameter Tuning Guide

### DSL Detection
1. **Layer Sensitivity**
   - Adjust `MIN_CLUSTER_SIZE_FACTOR` (default: 0.02)
   - Lower values: More sensitive to smaller layers
   - Higher values: Focus on larger, more prominent layers

2. **Migration Detection**
   - Modify `MIGRATION_WEIGHT_MULTIPLIER` (default: 0.5)
   - Increase: Stronger emphasis on vertical movement
   - Decrease: More emphasis on spatial proximity

### DVM Analysis
1. **Migration Timing**
   - Adjust `DAWN_MIGRATION_HOURS` and `DUSK_MIGRATION_HOURS`
   - Default: 2 hours before/after sunrise/sunset

2. **Detection Sensitivity**
   - Modify `DVM_THRESHOLD` (default: 0.4)
   - Lower values: More sensitive to subtle movements
   - Higher values: Focus on stronger migration signals

## Troubleshooting

### Common Issues

1. **No Layers Detected**
   - Check `db_min` and `db_max` thresholds
   - Verify data quality in the CSV file
   - Adjust `MIN_CLUSTER_SIZE_FACTOR`

2. **Over-segmentation**
   - Increase `MIN_CLUSTER_SIZE_FACTOR`
   - Adjust `CLUSTER_SELECTION_EPSILON`
   - Check `RATE_SIMILARITY_THRESHOLD`

3. **Poor Migration Detection**
   - Verify latitude/longitude values
   - Adjust `DVM_THRESHOLD`
   - Check temporal resolution settings

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Contact

For questions or support, please contact me directly at brlozano@fiu.edu
