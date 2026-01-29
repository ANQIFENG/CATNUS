#!/usr/bin/env python

import os
import subprocess
import logging


def apply_inverse_transform(
    segmentation_mni,      # Segmentation result in MNI space
    original_image,        # Reference image in original space
    transform_matrix,      # Forward transformation matrix (.mat)
    output_path           # Output path
):
    """
    Apply inverse transform to bring segmentation from MNI space 
    back to original space using ANTs antsApplyTransforms
    
    Parameters:
    -----------
    segmentation_mni : str
        Path to segmentation result in MNI space
    original_image : str
        Path to reference image in original space (defines output space)
    transform_matrix : str
        Path to forward transformation matrix (.mat file)
    output_path : str
        Path to output segmentation in original space
    
    Notes:
    ------
    - Uses NearestNeighbor interpolation (suitable for labels)
    - Uses inverse transform (add 1 after transform matrix)
    """
    
    logging.info(f"Applying inverse transform to segmentation...")
    logging.info(f"  Input (MNI space): {segmentation_mni}")
    logging.info(f"  Reference (Original space): {original_image}")
    logging.info(f"  Transform matrix: {transform_matrix}")
    logging.info(f"  Output: {output_path}")
    
    # Build ANTs command
    cmd = [
        "antsApplyTransforms",
        "--dimensionality", "3",
        "--input", segmentation_mni,
        "--reference-image", original_image,
        "--output", output_path,
        "--interpolation", "NearestNeighbor",  # Use nearest neighbor for labels
        "--transform", f"[{transform_matrix},1]",  # 1 indicates inverse transform
        "--verbose", "1"
    ]
    
    # Execute command
    try:
        subprocess.run(cmd, check=True)
        logging.info(f"Successfully applied inverse transform. Output saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error applying inverse transform: {e}")
        raise
    
    return output_path

