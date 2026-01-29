import numpy as np
from scipy.ndimage import label, generate_binary_structure


def center_crop(img, crop_size):

    num_dims = len(img.shape)
    center = [i // 2 for i in img.shape[-3:]]

    start = [center[i] - crop_size[i] // 2 for i in range(3)]
    end = [start[i] + crop_size[i] for i in range(3)]

    if num_dims == 3:
        return img[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    elif num_dims == 4:
        return img[:, start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    else:
        raise ValueError("Input must be a 3D or 4D array")


def pad_to_original_size(pred, original_shape):

    padding = []
    for pred_dim, orig_dim in zip(pred.shape, original_shape):
        total_pad = orig_dim - pred_dim
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        padding.append((pad_before, pad_after))

    padded_pred = np.pad(pred, padding, mode='constant', constant_values=0)

    return padded_pred


def min_max_normalization(data):
    min_val = data.min()
    max_val = data.max()
    if max_val > min_val:
        return (data - min_val) / (max_val - min_val)
    return data


def remove_small_components_overall(segmentation, threshold, print_volumes=False):
    """
    Remove small connected components from the entire segmentation.
    """
    binary_mask = segmentation != 0
    labeled, num_features = label(binary_mask)
    output = np.copy(segmentation)
    for component_label in range(1, num_features + 1):
        component = (labeled == component_label)
        volume = np.sum(component)
        if print_volumes:
            print(f"Component {component_label}: Volume = {volume} voxels")
        if volume < threshold:
            output[component] = 0
    return output


def remove_small_components_per_class(pred_label, pred_prob, threshold_replace=4, threshold_remove_CL=10, threshold_remove_MD=100, threshold_remove_other=10, num_classes=14):
    """
    Remove or replace small connected components for each class separately.

    """
    cleaned = np.copy(pred_label)
    structure = generate_binary_structure(3, 1)  # 6-connectivity
    
    for cls in range(1, num_classes):
        mask = (pred_label == cls)
        labeled, num_features = label(mask, structure=structure)
        
        for component_id in range(1, num_features + 1):
            component = (labeled == component_id)
            volume = component.sum()
            coords = np.argwhere(component)
            
            # Case 1: Very small components - replace with neighbor's most probable class (excluding current class)
            if volume < threshold_replace:
                for x, y, z in coords:
                    neighbors = []
                    for dx, dy, dz in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
                        nx, ny, nz = x + dx, y + dy, z + dz
                        if 0 <= nx < pred_label.shape[0] and \
                           0 <= ny < pred_label.shape[1] and \
                           0 <= nz < pred_label.shape[2]:
                            neighbors.append(pred_prob[:, nx, ny, nz])
                    if neighbors:
                        avg_probs = np.stack(neighbors).mean(axis=0)
                        # Exclude current class from consideration
                        avg_probs[cls] = -1
                        new_cls = np.argmax(avg_probs)
                    else:
                        new_cls = 0
                    cleaned[x, y, z] = new_cls
            
            # Case 2: CL (class 13) - remove if smaller than threshold_remove_CL
            elif cls == 13 and volume < threshold_remove_CL:
                for x, y, z in coords:
                    cleaned[x, y, z] = 0
            
            # Case 3: MD (class 5) - remove if smaller than threshold_remove_MD
            elif cls == 5 and volume < threshold_remove_MD:
                for x, y, z in coords:
                    cleaned[x, y, z] = 0
            
            # Case 4: Other classes - remove if between threshold_replace and threshold_remove_other voxels
            elif cls not in [5, 13] and threshold_replace <= volume < threshold_remove_other:
                for x, y, z in coords:
                    cleaned[x, y, z] = 0
                    
    return cleaned
