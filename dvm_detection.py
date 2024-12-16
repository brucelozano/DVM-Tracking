import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from astral import LocationInfo
from astral.sun import sun
from datetime import datetime, timezone
import cv2

from params import *

def detect_dvm_pattern(sv_data, depths, threshold=-70):
    """
    Detect the DVM pattern using image processing techniques.
    """
    # Create binary mask of strong signals using a more sensitive threshold
    mask = sv_data > (db_min + 2)  # Use slightly higher than db_min for better sensitivity
    
    # Apply Gaussian smoothing to reduce noise
    mask = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)
    mask = (mask > 0.5).astype(np.uint8)
    
    # Apply morphological operations to enhance the pattern
    kernel_vertical = np.ones((7, 1), np.uint8)
    kernel_horizontal = np.ones((1, 5), np.uint8)
    
    # First dilate to connect nearby signals
    mask = cv2.dilate(mask, kernel_vertical, iterations=1)
    
    # Then close to fill gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_horizontal)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_vertical)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    
    # Filter components by size and shape
    min_size = 200
    dvm_mask = np.zeros_like(mask)
    valid_components = []
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > min_size:
            component_mask = (labels == i)
            height = stats[i, cv2.CC_STAT_HEIGHT]
            width = stats[i, cv2.CC_STAT_WIDTH]
            if height > width * 0.3:  # Component should be reasonably tall
                dvm_mask[component_mask] = 1
                valid_components.append(i)
    
    # Extract pattern features with improved peak detection
    pattern_features = []
    window_size = 5  # Window for smoothing
    max_depth_change = 20  # Maximum allowed depth change in meters between pings
    
    # Initialize arrays for smoothing
    all_depths = []
    all_pings = []
    
    # First pass: collect all valid depths
    for ping_idx in range(sv_data.shape[1]):
        if np.any(dvm_mask[:, ping_idx]):
            depth_indices = np.where(dvm_mask[:, ping_idx])[0]
            sv_values = sv_data[depth_indices, ping_idx]
            
            # Find peaks in Sv values
            peaks = find_peaks(sv_values, distance=10)[0]
            
            if len(peaks) > 0:
                # Use the strongest peak
                peak_idx = peaks[np.argmax(sv_values[peaks])]
                peak_depth = depths[depth_indices[peak_idx]]
                
                all_depths.append(peak_depth)
                all_pings.append(ping_idx)
    
    # Convert to numpy arrays for processing
    if all_depths:
        all_depths = np.array(all_depths)
        all_pings = np.array(all_pings)
        
        # Apply median filter to smooth depths
        smoothed_depths = ndi.median_filter(all_depths, size=window_size)
        
        # Second pass: create features with validated depths
        prev_depth = None
        for i in range(len(all_pings)):
            current_depth = smoothed_depths[i]
            
            # Validate depth change
            if prev_depth is not None:
                depth_change = abs(current_depth - prev_depth)
                if depth_change > max_depth_change:
                    continue
            
            pattern_features.append({
                'ping': all_pings[i],
                'peak_depth': current_depth,
                'original_depth': all_depths[i]
            })
            prev_depth = current_depth
    
    return {
        'mask': dvm_mask,
        'features': pattern_features
    }

def get_sun_times(date, lat, lon):
    """
    Calculate sunrise and sunset times for a given location and date.
    """
    location = LocationInfo(latitude=lat, longitude=lon)
    sun_times = sun(location.observer, date)
    
    return {
        'sunrise': sun_times['sunrise'],
        'sunset': sun_times['sunset']
    }

def classify_migration_phase(timestamp, sun_times):
    """
    Classify the migration phase based on time of day.
    """
    if timestamp < sun_times['sunrise']:
        return 'pre_dawn'
    elif timestamp < sun_times['sunrise'] + pd.Timedelta(hours=2):
        return 'dawn_migration'
    elif timestamp > sun_times['sunset']:
        return 'post_dusk'
    elif timestamp > sun_times['sunset'] - pd.Timedelta(hours=2):
        return 'dusk_migration'
    else:
        return 'day'

def analyze_dvm(image_data, ping_time_map, depth_start, depth_stop, lat, lon):
    """
    Main DVM analysis function focusing on pattern detection.
    """
    # Ensure proper depth range
    pixels_per_meter = image_data.shape[0] / TOTAL_DEPTH
    depth_idx_start = int(depth_start * pixels_per_meter)
    depth_idx_end = int(depth_stop * pixels_per_meter)
    
    # Crop data to specified depth range
    cropped_data = image_data[depth_idx_start:depth_idx_end, :]
    depths = np.linspace(depth_start, depth_stop, cropped_data.shape[0])
    
    # Detect DVM pattern
    dvm_pattern = detect_dvm_pattern(cropped_data, depths, threshold=db_max)
    
    # Calculate migration rates from smoothed depths
    rates = []
    times = []
    
    if dvm_pattern['features']:
        features = dvm_pattern['features']
        for i in range(1, len(features)):
            prev = features[i-1]
            curr = features[i]
            
            time_diff = ping_time_map[curr['ping']] - ping_time_map[prev['ping']]
            if time_diff > 0:
                depth_diff = curr['peak_depth'] - prev['peak_depth']
                rate = depth_diff / time_diff
                # Only include reasonable rates
                if abs(rate) < 1.0:  # Max 1 meter per second
                    rates.append(rate)
                    times.append(ping_time_map[prev['ping']])
    
    # Get sun times and phases
    start_time = datetime.fromtimestamp(ping_time_map[0], timezone.utc)
    sun_times = get_sun_times(start_time.date(), lat, lon)
    timestamps = [datetime.fromtimestamp(t, timezone.utc) for t in ping_time_map.values()]
    phases = [classify_migration_phase(t, sun_times) for t in timestamps]
    
    return {
        'pattern_mask': dvm_pattern['mask'],
        'pattern_features': dvm_pattern['features'],
        'migration_rates': np.array(rates),
        'rate_times': np.array(times),
        'phases': phases,
        'sun_times': sun_times,
        'cropped_data': cropped_data
    }

def visualize_dvm(image_data, dvm_results, depth_start, depth_stop, 
                 vmin=-75, vmax=-65, save_path=None):
    """
    Visualize DVM analysis results focusing on the pattern.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[3, 1])
    
    # Plot echogram with correct dB range
    im = ax1.imshow(dvm_results['cropped_data'], aspect='auto', cmap='viridis',
                   extent=[0, image_data.shape[1], depth_stop, depth_start],
                   vmin=vmin, vmax=vmax)
    
    # Overlay DVM pattern with more visible coloring
    if 'pattern_mask' in dvm_results and np.any(dvm_results['pattern_mask']):
        pattern_overlay = np.ma.masked_where(dvm_results['pattern_mask'] == 0, 
                                           np.ones_like(dvm_results['pattern_mask']))
        ax1.imshow(pattern_overlay, aspect='auto', cmap='Reds', alpha=0.3,
                  extent=[0, image_data.shape[1], depth_stop, depth_start])
        
        # Plot peak migration depths if available
        if 'pattern_features' in dvm_results and dvm_results['pattern_features']:
            pings = [f['ping'] for f in dvm_results['pattern_features']]
            peak_depths = [f['peak_depth'] for f in dvm_results['pattern_features']]
            ax1.plot(pings, peak_depths, 'y-', linewidth=1, alpha=0.7, label='Peak Migration')
    
    # Add colorbar with correct dB labels
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Sv (dB)')
    
    # Plot migration rates if available
    if 'rate_times' in dvm_results and 'migration_rates' in dvm_results and len(dvm_results['rate_times']) > 0:
        ax2.plot(dvm_results['rate_times'], dvm_results['migration_rates'], 
                'b-', linewidth=1.5)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Migration Rate (m/s)')
        ax2.grid(True, alpha=0.3)
    
    # Add phase colors if available
    if 'phases' in dvm_results:
        unique_phases = np.unique(dvm_results['phases'])
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_phases)))
        phase_colors = dict(zip(unique_phases, colors))
        
        for i, phase in enumerate(dvm_results['phases']):
            ax1.axvspan(i, i+1, color=phase_colors[phase], alpha=0.1)
        
        # Add legend
        patches = [mpatches.Patch(color=color, label=phase, alpha=0.3) 
                  for phase, color in phase_colors.items()]
        ax1.legend(handles=patches, loc='upper right', fontsize=8)
    
    ax1.set_ylabel('Depth (m)')
    ax1.set_title('DVM Pattern Analysis')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def combine_dsl_dvm_analysis(dsl_results, dvm_results):
    """
    Combine DSL and DVM analysis results.
    """
    combined_results = {
        'dsl_layers': dsl_results['labels'],
        'dsl_features': dsl_results['features'],
        'pattern_mask': dvm_results['pattern_mask'],
        'pattern_features': dvm_results['pattern_features'],
        'migration_rates': dvm_results['migration_rates'],
        'rate_times': dvm_results['rate_times'],
        'phases': dvm_results['phases'],
        'sun_times': dvm_results['sun_times'],
        'cropped_data': dvm_results['cropped_data']
    }
    
    return combined_results

def visualize_combined_analysis(image_data, combined_results, depth_start, depth_stop,
                              vmin=-75, vmax=-65, save_path=None):
    """
    Create a combined visualization of DSL and DVM analysis.
    """
    plt.figure(figsize=(15, 8))
    
    # Plot echogram with correct dB range
    im = plt.imshow(combined_results['cropped_data'], aspect='auto', cmap='viridis',
                   extent=[0, image_data.shape[1], depth_stop, depth_start],
                   vmin=vmin, vmax=vmax)
    
    # Plot DSL layers if available
    if 'dsl_layers' in combined_results and 'dsl_features' in combined_results:
        unique_labels = np.unique(combined_results['dsl_layers'])
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                continue
            mask = combined_results['dsl_layers'] == label
            features = combined_results['dsl_features'][mask]
            
            # Filter features within depth range
            depth_mask = (features[:, 1] >= depth_start) & (features[:, 1] <= depth_stop)
            features = features[depth_mask]
            
            plt.scatter(features[:, 0], features[:, 1],
                       c=[color], label=f'Layer {label}',
                       s=1, alpha=0.5)
    
    # Overlay DVM pattern with more visible coloring
    if 'pattern_mask' in combined_results and np.any(combined_results['pattern_mask']):
        pattern_overlay = np.ma.masked_where(combined_results['pattern_mask'] == 0,
                                           np.ones_like(combined_results['pattern_mask']))
        plt.imshow(pattern_overlay, aspect='auto', cmap='Reds', alpha=0.3,
                  extent=[0, image_data.shape[1], depth_stop, depth_start],
                  label='DVM Pattern')
        
        # Plot peak migration depths if available
        if 'pattern_features' in combined_results and combined_results['pattern_features']:
            pings = [f['ping'] for f in combined_results['pattern_features']]
            peak_depths = [f['peak_depth'] for f in combined_results['pattern_features']]
            plt.plot(pings, peak_depths, 'y-', linewidth=1, alpha=0.7, label='Peak Migration')
    
    # Add day/night shading if available
    if 'phases' in combined_results:
        phase_colors = {
            'pre_dawn': 'navy',
            'dawn_migration': 'lightblue',
            'day': 'yellow',
            'dusk_migration': 'orange',
            'post_dusk': 'navy'
        }
        
        for i, phase in enumerate(combined_results['phases']):
            plt.axvspan(i, i+1, color=phase_colors[phase], alpha=0.1)
    
    # Add colorbar with correct dB labels
    cbar = plt.colorbar(im)
    cbar.set_label('Sv (dB)')
    
    plt.ylabel('Depth (m)')
    plt.xlabel('Ping Number')
    plt.title('Combined DSL and DVM Analysis')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close() 