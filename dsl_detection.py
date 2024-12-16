import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter
import hdbscan
from sklearn.preprocessing import StandardScaler
from scipy import ndimage as ndi

from params import *

def preprocess_image(image_data, depth_start, depth_end, db_min, db_max, total_depth):
    """
    Preprocess the echogram with adaptive thresholding based on signal strength.
    """
    # Convert depths to indices
    pixels_per_meter = image_data.shape[0] / total_depth
    depth_idx_start = int(depth_start * pixels_per_meter)
    depth_idx_end = int(depth_end * pixels_per_meter)
    
    # Crop to depth range
    cropped_data = image_data[depth_idx_start:depth_idx_end, :].copy()
    
    # Debug 1: Show cropped data
    print(f"db_min used for cropping: {db_min}, db_max used for cropping: {db_max}")
    plt.figure(figsize=(15, 8))
    plt.imshow(cropped_data, aspect='auto', cmap='viridis',
               extent=[0, cropped_data.shape[1], depth_end, depth_start])
    plt.colorbar(label='Sv (dB)')
    plt.title('Debug 1: Cropped Data')
    plt.savefig('/Users/bruce/Documents/Research/DSL Analysis/Debug Output/debug_1_cropped.png')
    plt.close()

    # Process density
    density = np.clip((cropped_data - db_min) / (db_max - db_min), 0, 1)
    
    # Debug 2: Show DB mask
    plt.figure(figsize=(15, 8))
    plt.imshow(density, aspect='auto', cmap='viridis',
               extent=[0, density.shape[1], depth_end, depth_start])
    plt.colorbar(label='Normalized Density')
    plt.title('Debug 2: DB Mask')
    plt.savefig('/Users/bruce/Documents/Research/DSL Analysis/Debug Output/debug_2_db_mask.png')
    plt.close()
    
    # Apply initial smoothing
    smoothed_density = gaussian_filter(density, sigma=GAUSSIAN_SIGMA)
    
    # Debug 3: Show smoothed density
    plt.figure(figsize=(15, 8))
    plt.imshow(smoothed_density, aspect='auto', cmap='viridis',
               extent=[0, smoothed_density.shape[1], depth_end, depth_start])
    plt.colorbar(label='Smoothed Density')
    plt.title('Debug 3: Smoothed Density')
    plt.savefig('/Users/bruce/Documents/Research/DSL Analysis/Debug Output/debug_3_smoothed.png')
    plt.close()

    # Calculate signal strength profile with more detail
    depth_profile = np.mean(smoothed_density, axis=1)
    
    # Use smaller window for finer detail detection
    window_size = DETAIL_WINDOW_SIZE  # Reduced from 15 for finer detail
    rolling_mean = np.convolve(depth_profile, np.ones(window_size)/window_size, mode='same')
    
    # Calculate local statistics for adaptive thresholding
    local_std = np.array([np.std(depth_profile[max(0, i-window_size//2):min(len(depth_profile), i+window_size//2)]) 
                         for i in range(len(depth_profile))])
    local_max = np.array([np.max(depth_profile[max(0, i-window_size//2):min(len(depth_profile), i+window_size//2)]) 
                         for i in range(len(depth_profile))])
    
    # More sophisticated signal detection
    consistency_mask = local_std < np.median(local_std) * 1.5  # More lenient consistency threshold
    signal_strength = rolling_mean / np.maximum(local_max, 1e-6)  # Local normalization
    
    # Multi-level thresholding
    base_threshold = BASE_THRESHOLD
    threshold_map = np.ones_like(depth_profile) * BASE_THRESHOLD
    
    # Define multiple signal strength levels with more sensitive thresholds
    strong_signal = signal_strength >= STRONG_SIGNAL_THRESHOLD  # Reduced from 0.7
    medium_signal = (signal_strength >= MEDIUM_SIGNAL_THRESHOLD) & (signal_strength < STRONG_SIGNAL_THRESHOLD)  # Adjusted ranges
    weak_signal = (signal_strength >= WEAK_SIGNAL_THRESHOLD) & (signal_strength < MEDIUM_SIGNAL_THRESHOLD)  # More sensitive to weak signals
    
    # Apply different thresholds based on signal strength and consistency
    threshold_map[strong_signal] = BASE_THRESHOLD * THRESHOLD_MULTIPLIERS['strong']  # Slightly lower base threshold
    threshold_map[medium_signal & consistency_mask] = BASE_THRESHOLD * THRESHOLD_MULTIPLIERS['medium']  # More sensitive
    threshold_map[weak_signal & consistency_mask] = BASE_THRESHOLD * THRESHOLD_MULTIPLIERS['weak']  # Much more sensitive
    
    # Smooth threshold transitions with larger sigma
    threshold_map = gaussian_filter(threshold_map, sigma=THRESHOLD_SMOOTHING_SIGMA)
    
    # Apply adaptive threshold with more sensitive local context
    feature_mask = np.zeros_like(smoothed_density, dtype=bool)
    for i in range(smoothed_density.shape[0]):
        local_density = smoothed_density[i, :]
        local_threshold = threshold_map[i]
        
        # More sensitive point inclusion
        absolute_strong = local_density > local_threshold
        relative_strong = local_density > (np.mean(local_density) + local_threshold * 0.4)  # Reduced from 0.5
        local_peaks = local_density > (np.median(local_density) + local_threshold * 0.3)  # Added local peaks detection
        feature_mask[i, :] = absolute_strong | relative_strong | local_peaks
    
    # Debug 4: Show enhanced mask
    plt.figure(figsize=(15, 8))
    plt.imshow(feature_mask, aspect='auto', cmap='gray',
               extent=[0, feature_mask.shape[1], depth_end, depth_start])
    plt.title('Debug 4: Enhanced Mask')
    plt.savefig('/Users/bruce/Documents/Research/DSL Analysis/Debug Output/debug_4_enhanced_mask.png')
    plt.close()

    # Apply additional processing to the feature mask if needed
    # For example, fill small gaps or apply additional smoothing
    final_mask = ndi.binary_fill_holes(feature_mask).astype(int)

    # Debug 5: Show final mask
    plt.figure(figsize=(15, 8))
    plt.imshow(final_mask, aspect='auto', cmap='gray',
               extent=[0, final_mask.shape[1], depth_end, depth_start])
    plt.title('Debug 5: Final Mask')
    plt.savefig('/Users/bruce/Documents/Research/DSL Analysis/Debug Output/debug_5_final_mask.png')
    plt.close()
    
    return final_mask, smoothed_density

def enhance_layers(mask, depth_start, depth_end):
    """
    Enhance layer features using morphological operations.
    """
    # Convert mask to uint8
    binary = mask.astype(np.uint8) * 255
    
    # Debug 6: Show initial binary image with contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 1)  # Draw contours in green
    
    plt.figure(figsize=(15, 8))
    plt.imshow(contour_image, aspect='auto', cmap='gray',
               extent=[0, binary.shape[1], depth_end, depth_start])
    plt.title('Debug 6: Initial Binary Image with Contours')
    plt.savefig('/Users/bruce/Documents/Research/DSL Analysis/Debug Output/debug_6_contours.png')
    plt.close()
    
    # Adaptive kernel sizes based on depth
    y_coords = np.arange(mask.shape[0])
    depth_values = depth_start + (y_coords / mask.shape[0]) * (depth_end - depth_start)
    
    # Create adaptive kernels
    base_kernel_size = BASE_KERNEL_SIZE
    kernel_sizes = np.ones_like(depth_values) * base_kernel_size
    
    # Apply morphological operations with adaptive kernels
    enhanced = np.copy(binary)
    for i in range(len(depth_values)):
        kernel_size = int(kernel_sizes[i])
        local_kernel = np.ones((HORIZONTAL_KERNEL_MULTIPLIER, kernel_size), np.uint8)
        row_data = enhanced[i:i+1, :]
        row_enhanced = cv2.morphologyEx(row_data, cv2.MORPH_CLOSE, local_kernel)
        enhanced[i:i+1, :] = row_enhanced
    
    # Debug 7: Show after horizontal closing
    plt.figure(figsize=(15, 8))
    plt.imshow(enhanced, aspect='auto', cmap='gray',
               extent=[0, enhanced.shape[1], depth_end, depth_start])
    plt.title('Debug 7: After Horizontal Closing')
    plt.savefig('/Users/bruce/Documents/Research/DSL Analysis/Debug Output/debug_7_horizontal_close.png')
    plt.close()
    
    # Final vertical smoothing
    kernel_v = np.ones((VERTICAL_KERNEL_SIZE, 1), np.uint8)
    enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel_v)
    
    # Debug 8: Show final enhanced image
    plt.figure(figsize=(15, 8))
    plt.imshow(enhanced, aspect='auto', cmap='gray',
               extent=[0, enhanced.shape[1], depth_end, depth_start])
    plt.title('Debug 8: Final Enhanced Image')
    plt.savefig('/Users/bruce/Documents/Research/DSL Analysis/Debug Output/debug_8_enhanced.png')
    plt.close()
    
    return enhanced

def detect_layers(enhanced_image, density_data, depth_start, depth_end):
    """
    Detect layers using density-weighted HDBSCAN clustering.
    """
    # Get coordinates where density is above threshold
    y_coords, x_coords = np.where(enhanced_image > 0)
    
    if len(y_coords) == 0:
        print("No layer pixels detected!")
        return np.array([]), np.array([])
    
    # Get density values
    densities = density_data[y_coords, x_coords]
    
    # Create features in chunks to save memory
    chunk_size = CHUNK_SIZE
    n_points = len(x_coords)
    features_list = []
    
    for i in range(0, n_points, chunk_size):
        end_idx = min(i + chunk_size, n_points)
        chunk_features = np.column_stack([
            x_coords[i:end_idx],
            (y_coords[i:end_idx] / enhanced_image.shape[0]) * (depth_end - depth_start) + depth_start,
            densities[i:end_idx] * 10
        ])
        features_list.append(chunk_features)
    
    features_scaled = np.vstack(features_list)
    del features_list  # Free memory
    
    # Debug 9: Show features before clustering
    plt.figure(figsize=(15, 8))
    plt.scatter(features_scaled[:, 0], features_scaled[:, 1], 
               c=features_scaled[:, 2], cmap='viridis', 
               s=1, alpha=0.5)
    plt.ylim(depth_end, depth_start)
    plt.title('Debug 9: Features Before Clustering')
    plt.colorbar(label='Density')
    plt.savefig('/Users/bruce/Documents/Research/DSL Analysis/Debug Output/debug_9_features.png')
    plt.close()
    
    # Calculate local migration rates in windows
    window_size = MIGRATION_WINDOW_SIZE  # Adjust as needed
    x_bins = np.arange(0, enhanced_image.shape[1], window_size)
    migration_weights = np.ones(len(features_scaled))
    
    for i in range(len(x_bins)-1):
        mask = (features_scaled[:, 0] >= x_bins[i]) & (features_scaled[:, 0] < x_bins[i+1])
        if np.sum(mask) > 10:  # Ensure enough points for calculation
            window_depths = features_scaled[mask, 1]
            window_times = features_scaled[mask, 0]
            try:
                # Calculate local migration rate with quality assessment
                slope, r2 = calculate_migration_rate(window_times, window_depths)
                # Only use rate if fit quality is good enough
                if r2 > 0.6:  # Adjust this threshold as needed
                    migration_weights[mask] = 1 + abs(slope) * MIGRATION_WEIGHT_MULTIPLIER
            except:
                continue
    
    # Normalize features with migration awareness
    features_normalized = np.column_stack([
        features_scaled[:, 0] / density_data.shape[1],  # Time normalization
        (features_scaled[:, 1] - depth_start) / (depth_end - depth_start) * migration_weights,  # Depth with migration weight
        features_scaled[:, 2] / np.max(features_scaled[:, 2])  # Density normalization
    ])
    
    # Calculate minimum cluster size
    total_points = len(features_normalized)
    min_cluster_size = max(100, int(total_points * MIN_CLUSTER_SIZE_FACTOR))
    
    # HDBSCAN clustering with adjusted parameters
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=MIN_SAMPLES,
        cluster_selection_epsilon=CLUSTER_SELECTION_EPSILON,
        metric='euclidean',
        cluster_selection_method='leaf',
        alpha=CLUSTER_ALPHA
    )
    
    print(f"Clustering {total_points} points...")
    cluster_labels = clusterer.fit_predict(features_normalized)
    
    # Debug 10: Show clustering results
    plt.figure(figsize=(15, 8))
    unique_labels = np.unique(cluster_labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    for label, color in zip(unique_labels, colors):
        mask = cluster_labels == label
        plt.scatter(features_scaled[mask, 0], features_scaled[mask, 1],
                   c=[color], label=f'Layer {label}' if label != -1 else 'Noise',
                   s=1, alpha=0.5)
    plt.ylim(depth_end, depth_start)
    plt.title('Debug 10: Clustering Results')
    plt.legend()
    plt.savefig('/Users/bruce/Documents/Research/DSL Analysis/Debug Output/debug_10_clusters.png')
    plt.close()
    
    return features_scaled, cluster_labels

def visualize_layers(image_data, features, labels, depth_start, depth_stop, total_depth):
    """
    Visualize detected DSL layers with density information.
    """
    plt.figure(figsize=(15, 8))
    
    # Calculate indices for cropping the visualization
    pixels_per_meter = image_data.shape[0] / total_depth
    depth_idx_start = int(depth_start * pixels_per_meter)
    depth_idx_end = int(depth_stop * pixels_per_meter)
    
    # Crop image data to the specified depth range
    cropped_image = image_data[depth_idx_start:depth_idx_end, :]
    
    # Plot cropped echogram
    plt.imshow(cropped_image, 
              aspect='auto',
              extent=[0, image_data.shape[1], depth_stop, depth_start],
              cmap='viridis',
              vmin=-100,
              vmax=-40)
    
    if len(features) > 0:
        unique_labels = np.unique(labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                continue
                
            mask = labels == label
            # Use density values to vary point opacity
            densities = features[mask, 2] / 10  # Normalize back from scaling
            color_with_alpha = np.array(color)
            
            plt.scatter(features[mask, 0],
                       features[mask, 1],
                       c=[color],
                       alpha=0.3,
                       s=2 + densities * 3,  # Vary point size with density
                       label=f'Layer {label}',
                       edgecolors='none')
    
    plt.colorbar(label='Sv (dB)')
    plt.ylabel('Depth (m)')
    plt.xlabel('Ping Number')
    plt.title('Detected DSL Layers')
    plt.grid(True, alpha=0.3)
    plt.ylim(depth_stop, depth_start)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('/Users/bruce/Documents/Research/DSL Analysis/Figures/detected_layers.png')
    plt.close()

def split_or_merge_layer(labels, current_label, features, max_allowed_gap=10, depth_threshold=15):
    """
    Split or merge layers based on temporal gaps and vertical continuity.
    """
    mask = labels == current_label
    ping_indices = features[mask, 0]
    depths = features[mask, 1]
    
    # Sort by ping indices and get corresponding depths
    sort_idx = np.argsort(ping_indices)
    sorted_indices = ping_indices[sort_idx]
    sorted_depths = depths[sort_idx]
    
    # Find temporal gaps
    gaps = np.diff(sorted_indices)
    split_points = np.where(gaps > max_allowed_gap)[0]
    
    if len(split_points) > 0:
        # Create a mask for split points to keep
        keep_splits = np.ones(len(split_points), dtype=bool)
        
        # Check each split point
        for i in range(len(split_points)):
            # Check depth difference across gap
            depth_before = sorted_depths[split_points[i]]
            depth_after = sorted_depths[split_points[i] + 1]
            depth_diff = abs(depth_after - depth_before)
            
            # If depth difference is small enough, mark split point for removal
            if depth_diff <= depth_threshold:
                keep_splits[i] = False
        
        # Filter split points using the mask
        split_points = split_points[keep_splits]
        
        # Only split if there are remaining split points
        if len(split_points) > 0:
            new_label = np.max(labels) + 1
            for split_point in split_points:
                split_index = sorted_indices[split_point + 1]
                labels[mask & (features[:, 0] >= split_index)] = new_label
                new_label += 1
    
    # Debug: Output the number of splits
    print(f"Label {current_label}: {len(split_points)} splits applied.")
    
    return labels 

def merge_nearby_layers(labels, features):
    """
    Merge layers that are temporally and spatially adjacent, accounting for DVM direction.
    """
    unique_labels = np.unique(labels[labels != -1])
    
    for label1 in unique_labels:
        mask1 = labels == label1
        if not np.any(mask1):
            continue
            
        for label2 in unique_labels:
            if label1 >= label2:
                continue
                
            mask2 = labels == label2
            if not np.any(mask2):
                continue
            
            # Get temporal boundaries and all points for each layer
            layer1_times = features[mask1, 0]
            layer1_depths = features[mask1, 1]
            layer2_times = features[mask2, 0]
            layer2_depths = features[mask2, 1]
            
            time1_end = np.max(layer1_times)
            time2_start = np.min(layer2_times)
            
            # Calculate migration rate for both layers with quality assessment
            rate1, r2_1 = calculate_migration_rate(layer1_times, layer1_depths)
            rate2, r2_2 = calculate_migration_rate(layer2_times, layer2_depths)
            
            # Only proceed if both fits are good enough
            if r2_1 > 0.6 and r2_2 > 0.6:  # Adjust threshold as needed
                # Check if rates are in the same direction
                same_direction = (rate1 * rate2) > 0
                rate_similarity = abs(rate1 - rate2) < RATE_SIMILARITY_THRESHOLD
                
                if same_direction and rate_similarity:
                    # Get depths at boundaries
                    depth1_end = layer1_depths[layer1_times == time1_end].mean()
                    depth2_start = layer2_depths[layer2_times == time2_start].mean()
                    
                    # Calculate expected depth difference based on migration rates
                    time_gap = time2_start - time1_end
                    expected_depth_change = (rate1 + rate2) / 2 * time_gap
                    actual_depth_change = depth2_start - depth1_end
                    
                    if (time_gap <= TIME_THRESHOLD and 
                        abs(actual_depth_change - expected_depth_change) <= DEPTH_DIFF_THRESHOLD):
                        # Merge label2 into label1
                        labels[labels == label2] = label1
                        print(f"Merging label {label2} into {label1}")
    
    return labels 

def check_temporal_connectivity(features, labels):
    """
    Check and maintain temporal connectivity of layers.
    """
    # First pass: split layers with large temporal gaps
    unique_labels = np.unique(labels[labels != -1])
    for label in unique_labels:
        mask = labels == label
        temporal_gaps = find_temporal_gaps(features[mask, 0])
        if temporal_gaps > 0:
            labels = split_or_merge_layer(labels, label, features)
    
    # Second pass: merge nearby layers
    labels = merge_nearby_layers(labels, features)
    
    # Debug: Visualize merged layers
    plt.figure(figsize=(15, 8))
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    for label, color in zip(unique_labels, colors):
        if label == -1:
            continue
        mask = labels == label
        plt.scatter(features[mask, 0], features[mask, 1],
                    c=[color], label=f'Layer {label}',
                    s=1, alpha=0.5)
    plt.ylim(np.max(features[:, 1]), np.min(features[:, 1]))
    plt.title('Debug: Merged Layers')
    plt.legend()
    plt.savefig('/Users/bruce/Documents/Research/DSL Analysis/Debug Output/debug_11_merged_layers.png')
    plt.close()
    
    return labels

def find_temporal_gaps(ping_indices, max_allowed_gap=5):
    """
    Identify temporal gaps in ping indices.
    
    Parameters:
    ping_indices (np.ndarray): Array of ping indices for a cluster.
    max_allowed_gap (int): Maximum allowed gap between consecutive pings.
    
    Returns:
    int: Number of gaps exceeding the allowed threshold.
    """
    sorted_indices = np.sort(ping_indices)
    gaps = np.diff(sorted_indices)
    large_gaps = gaps > max_allowed_gap
    return np.sum(large_gaps)

def calculate_migration_rate(times, depths):
    """
    Calculate migration rate with quality assessment.
    
    Parameters:
    times (np.ndarray): Array of time points
    depths (np.ndarray): Array of depth measurements
    
    Returns:
    tuple: (slope, r2) where slope is the migration rate and r2 is the fit quality
    """
    if len(times) < 2:
        return 0, 0
        
    # Fit linear regression
    slope, intercept = np.polyfit(times, depths, 1)
    # Calculate R² to assess fit quality
    y_pred = slope * times + intercept
    r2 = 1 - (np.sum((depths - y_pred) ** 2) / np.sum((depths - np.mean(depths)) ** 2))
    return slope, r2

def analyze_dsl(image_data, ping_time_map, depth_start, depth_stop, total_depth):
    """
    Main DSL analysis pipeline.
    """
    # Preprocess the image
    preprocessed, filtered_data = preprocess_image(image_data, depth_start, depth_stop, db_min, db_max, total_depth)
    
    # Enhance layer features
    enhanced = enhance_layers(preprocessed, depth_start, depth_stop)
    
    # Detect layers using HDBSCAN with depth range
    features, labels = detect_layers(enhanced, filtered_data, depth_start, depth_stop)
    
    # Check temporal connectivity with improved functions
    labels = check_temporal_connectivity(features, labels)
    
    # Visualize results
    visualize_layers(image_data, features, labels, depth_start, depth_stop, total_depth)
    
    # Count number of layers (excluding noise)
    n_layers = len(np.unique(labels)) - 1  # Subtract 1 to exclude noise label (-1)
    print(f"Detected {n_layers} distinct layers")
    
    results = {
        'n_layers': n_layers,
        'labels': labels,
        'features': features
    }
    
    return results