import pandas as pd
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import pyplot
import matplotlib.patches as mpatches
import matplotlib.patches as mpatches
from scipy import ndimage as ndi
pyplot.rcParams['figure.dpi'] = 800
pyplot.rcParams['savefig.dpi'] = 800

from dsl_detection import analyze_dsl, visualize_layers
from params import *
from dvm_detection import analyze_dvm, visualize_dvm, combine_dsl_dvm_analysis, visualize_combined_analysis

def load_and_preprocess_echogram(file_path):
    # Define the names of the fixed columns Echoview creates when exporting to CSV
    fixed_column_names = [
        'Ping_index', 'Distance_gps', 'Distance_vl', 'Ping_date',
        'Ping_time', 'Ping_milliseconds', 'Latitude', 'Longitude', 
        'Depth_start', 'Depth_stop', 'Range_start', 'Range_stop', 'Sample_count'
    ]

    # Read only the Sample_count column for analysis
    initial_data = pd.read_csv(file_path, usecols=fixed_column_names)
    max_samples = int(initial_data['Sample_count'].max())
    print(f"Maximum number of sample columns: {max_samples}\n")

    # Generate sample column names based on the determined sample count
    sample_column_names = ['Sample_' + str(i) for i in range(1, max_samples + 1)]

    # Combine the fixed and sample column names for full DataFrame reading
    column_names = fixed_column_names + sample_column_names

    # Read the full CSV with the correct number of columns
    echogram = pd.read_csv(file_path, names=column_names, header=0, low_memory=False)

    # Store original start and end ping indices
    original_start_ping = echogram['Ping_index'].iloc[0]
    original_end_ping = echogram['Ping_index'].iloc[-1]
    
    # Reset ping indices to start from 0 only if they don't already start from 0
    if original_start_ping != 0:
        echogram['Ping_index'] = echogram['Ping_index'] - original_start_ping
        print(f"Adjusted ping range: 0 to {original_end_ping - original_start_ping}")
    else:
        print("Ping indices already start from 0. No adjustment needed.")

    # Get placeholder value
    placeholder_val = -9.900000000000001e+37
    print("Placeholder value is: ", placeholder_val, end = "\n\n")

    # Replace placeholder_val of -9.900000000000001e+37 with 0.0 (NaN causes errors when importing back to Echoview)
    echogram.replace(placeholder_val, 0.0, inplace=True)
    
    print("Replaced all values containing -9.900000000000001e+37 to 0.0")
    print(f"Original ping range: {original_start_ping} to {original_end_ping}")

    return echogram, fixed_column_names, original_start_ping, original_end_ping

# Use this to load the data
print("Loading and preprocessing echogram...\n")
echogram, fixed_column_names, original_start_ping, original_end_ping = load_and_preprocess_echogram(FILE_PATH)

def prepare_sample_data(echogram):
    """
    Prepare sample data from the echogram DataFrame.

    Parameters:
    echogram (pd.DataFrame): The original echogram DataFrame.

    Returns:
    tuple: A tuple containing:
        - sample_data (pd.DataFrame): DataFrame with only sample columns.
        - image_data (np.ndarray): NumPy array of sample data, transposed and converted to float32.
    """
    # Get columns that start with 'Sample_' and exclude 'Sample_count'
    sample_columns = [col for col in echogram.columns if col.startswith('Sample_') and col != 'Sample_count']

    # Filter the DataFrame to only include these columns
    sample_data = echogram[sample_columns]
    
    # Convert the filtered DataFrame to a NumPy array to be used as an image matrix in OpenCV
    image_data = sample_data.to_numpy(dtype='float32')
    print("Before transposing:\n")
    print("Image dtype:", image_data.dtype)  # Check dtype
    print("Image shape:", image_data.shape, "\n")  # Should be (height, width) or (height, width, channels)

    # Transpose the image data
    image_data = image_data.T
    print("After transposing:\n")
    print("Image dtype:", image_data.dtype)  # Check dtype
    print("Image shape:", image_data.shape, "\n")  # Should be (height, width) or (height, width, channels)

    return sample_data, image_data

# Example usage:
sample_data, image_data = prepare_sample_data(echogram)

# Checking the data with our new filtered DataFrame
print(sample_data.head())
print("Sample data shape: ", sample_data.shape)
print("Image data shape: ", image_data.shape)

def calculate_depth_per_sample(depth_start, depth_stop, sample_count):
    """
    Calculate the depth per sample.

    Parameters:
    depth_start (float): The starting depth in meters
    depth_stop (float): The stopping depth in meters
    sample_count (int): The number of samples

    Returns:
    float: The depth per sample in meters, rounded to 3 decimal places
    """
    depth_per_sample = (depth_stop - depth_start) / sample_count
    return round(depth_per_sample, 3)

def get_depth_parameters(dataframe):
    """
    Extract depth_start, depth_stop, and sample_count from the first row of the dataframe.

    Parameters:
    dataframe (pd.DataFrame): The input DataFrame

    Returns:
    tuple: A tuple containing (depth_start, depth_stop, sample_count)
    """
    first_row = dataframe.iloc[0]
    depth_start = first_row['Depth_start']
    depth_stop = first_row['Depth_stop']
    sample_count = first_row['Sample_count']
    total_depth = depth_stop - depth_start
    return depth_start, depth_stop, sample_count, total_depth

# Example usage:
depth_start, depth_stop, sample_count, total_depth = get_depth_parameters(echogram)
print(f"Depth start: {depth_start}")
print(f"Depth stop: {depth_stop}")
print(f"Sample count: {sample_count}")
print(f"Total depth: {total_depth}")

# Now use these values in the calculate_depth_per_sample function
depth_per_sample = calculate_depth_per_sample(depth_start, depth_stop, sample_count)
print(f"{depth_per_sample} meters per sample")

def get_dmax(dmin, dmax, drange):
    if dmax is None:
        dmax = dmin + drange  # Display maximum intensity, derived from display minimum + display range
    return dmax

dmax = get_dmax(dmin, dmax, drange) # Get the display maximum

def plot_echogram(image_data, depth_start, depth_stop, dmin, dmax, title='Annotated Echogram'):
    """
    Plot the echogram with specified parameters.

    Parameters:
    image_data (np.array): The 2D array of image data
    Depth_start (float): The starting depth
    Depth_stop (float): The stopping depth
    dmin (float): The minimum value for color scaling (default -84)
    dmax (float): The maximum value for color scaling (default None, will be set to dmin + drange)
    title (str): The title for the plot (default 'Annotated Echogram')
    """
    print("Image dtype:", image_data.dtype)  # Check dtype
    print("Image shape:", image_data.shape)  # Should be (height, width) or (height, width, channels)

    print("Display minimum (dmin):", dmin)
    print("Display maximum (dmax):", dmax)
    print("Display range (drange):", drange) 


    plt.figure(figsize=(20, 6))
    plt.imshow(image_data, cmap='viridis', extent=[0, image_data.shape[1], depth_stop, depth_start], vmin=dmin, vmax=dmax)
    plt.colorbar(label='Sv Intensity (dB)')
    plt.xlabel('Ping Index')
    plt.ylabel('Depth (m)')
    plt.title(title)
    plt.savefig('/Users/bruce/Documents/Research/DSL Analysis/Figures/echogram.png')
    #plt.show()

# Plot the echogram
plot_echogram(image_data, depth_start, depth_stop, dmin, dmax)

# If you want to specify different dmin, dmax, or title:
# plot_echogram(image_data, depth_start, depth_stop, dmin=-90, dmax=-40, title='Custom Echogram Plot')

def create_ping_time_mapping(echogram):
    """
    Create an explicit mapping between ping indices and timestamps.
    
    Parameters:
    echogram (pd.DataFrame): DataFrame containing 'Ping_date', 'Ping_time', and 'Ping_index' columns
    
    Returns:
    dict: Mapping of ping indices to seconds since start
    """
    # Verify we have required columns
    required_columns = ['Ping_date', 'Ping_time']
    if not all(col in echogram.columns for col in required_columns):
        raise ValueError(f"Missing required columns: {required_columns}")
    
    # Create timestamps
    datetime_str = echogram['Ping_date'] + ' ' + echogram['Ping_time']
    timestamps = pd.to_datetime(datetime_str)
    seconds = (timestamps - timestamps.iloc[0]).dt.total_seconds()
    
    # Create explicit mapping
    ping_time_map = {}
    for idx in range(len(echogram)):
        ping_time_map[idx] = seconds.iloc[idx]
    
    # Add some validation
    print(f"Created time mapping for {len(ping_time_map)} pings")
    print(f"Time range: {ping_time_map[0]:.2f}s to {ping_time_map[len(echogram)-1]:.2f}s")
    
    return ping_time_map

def process_echogram_with_dsl_detection():
    """
    Process the echogram and detect DSL layers.
    """
    print("\nStarting DSL detection pipeline...")
    
    # Load and preprocess the echogram
    print("Loading and preprocessing echogram...\n")
    
    # Prepare sample data
    print("Preparing sample data...\n")
    sample_data, image_data = prepare_sample_data(echogram)
    
    # Get depth parameters
    print("Getting depth parameters...\n")
    depth_start, depth_stop, sample_count, total_depth = get_depth_parameters(echogram)
    
    # Override depth range for DSL detection
    depth_start = 125  # Set desired start depth
    depth_stop = 275   # Set desired stop depth
    print(f"Processing depth range: {depth_start:.1f}m to {depth_stop:.1f}m")
    
    # Create ping time mapping
    print("Creating ping time mapping...\n")
    ping_time_map = create_ping_time_mapping(echogram)
    
    # Get location data for DVM analysis
    lat = echogram['Latitude'].iloc[0]
    lon = echogram['Longitude'].iloc[0]
    
    # Analyze DSL
    print("Analyzing DSL patterns...\n")
    dsl_results = analyze_dsl(image_data, ping_time_map, depth_start, depth_stop, total_depth)
    
    # Analyze DVM
    print("Analyzing DVM patterns...\n")
    dvm_results = analyze_dvm(
        image_data, 
        ping_time_map, 
        depth_start, 
        depth_stop, 
        lat, 
        lon
    )
    
    # Combine DSL and DVM results
    print("Combining DSL and DVM analysis...\n")
    combined_results = combine_dsl_dvm_analysis(dsl_results, dvm_results)
    
    # Create visualizations
    print("Creating visualizations...\n")
    
    # DSL visualization (existing)
    visualize_layers(image_data, dsl_results['features'], dsl_results['labels'], 
                    depth_start, depth_stop, total_depth)
    
    # DVM visualization
    visualize_dvm(
        image_data, 
        dvm_results, 
        depth_start, 
        depth_stop,
        vmin=DVM_VMIN,
        vmax=DVM_VMAX,
        save_path='/Users/bruce/Documents/Research/DSL Analysis/Figures/dvm_analysis.png'
    )
    
    # Combined visualization
    visualize_combined_analysis(
        image_data,
        combined_results,
        depth_start,
        depth_stop,
        vmin=DVM_VMIN,
        vmax=DVM_VMAX,
        save_path='/Users/bruce/Documents/Research/DSL Analysis/Figures/combined_analysis.png'
    )
    
    print("Analysis complete!\n")
    return combined_results

if __name__ == "__main__":
    results = process_echogram_with_dsl_detection()
    
    # Print summary of results
    if 'layer_features' in results:
        print(f"\nDetected {len(results['layer_features'])} DSL layers")
        for i, layer in enumerate(results['layer_features'], 1):
            print(f"Layer {i}: Depth range {layer['start_depth']:.1f}m - {layer['end_depth']:.1f}m, "
                  f"Mean intensity: {layer['mean_intensity']:.1f} dB")