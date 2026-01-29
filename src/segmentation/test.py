import os
import torch
import argparse
import numpy as np
import nibabel as nib

import sys
sys.path.append('/opt/run')
from unet3d_coord import UnetL4
from dataloader import thalamus_dataloader
from utils import pad_to_original_size, remove_small_components_overall, remove_small_components_per_class


def load_model(model_path, in_dim, out_dim, num_filters, output_activation, device):

    model = UnetL4(in_dim=in_dim, out_dim=out_dim, num_filters=num_filters, output_activation=output_activation, with_r=False).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    return model


def test(args):

    # Prepare model
    model = load_model(args.model_weight, in_dim=1, out_dim=14, num_filters=4, output_activation=torch.nn.Softmax(dim=1), device=args.device)

    # Prepare dataloader
    data_loaders = thalamus_dataloader(data_path=args.data_path)

    for batch_idx, data in enumerate(data_loaders):

        # Model prediction
        data = data.type(torch.float32).to(args.device)
        
        with torch.no_grad():
            pred = model(data).to(args.device)

        # Convert probability to index
        pred_labels = torch.argmax(pred, dim=1).type(torch.int32)

        # Convert torch tensor to array
        pred_arr = pred_labels.squeeze().detach().cpu().numpy()
        prob_arr = pred.squeeze().detach().cpu().numpy()

        # Post-processing: Two-step approach
        # Step 1: Remove small connected components overall
        pred_arr = remove_small_components_overall(pred_arr, threshold=300)
        
        # Step 2: Per-class post-processing
        pred_arr = remove_small_components_per_class(pred_arr, prob_arr, threshold_replace=4, threshold_remove_CL=10, threshold_remove_MD=100, threshold_remove_other=10)

        # Pad to original size
        shape = nib.load(args.data_path).shape
        pred_arr_padded = pad_to_original_size(pred_arr, shape)

        # Get affine matrix
        affine = nib.load(args.data_path).affine

        # Save results to output dir with modality-specific suffix
        base_name = os.path.basename(args.data_path)
        # Handle both .nii and .nii.gz extensions
        if base_name.endswith('.nii.gz'):
            out_fn = base_name.replace(".nii.gz", f"_catnus-{args.modality}.nii.gz")
        else:
            out_fn = base_name.replace(".nii", f"_catnus-{args.modality}.nii")
        
        out_fp = os.path.join(args.out_dir, out_fn)
        nib.save(nib.Nifti1Image(pred_arr_padded.astype(np.int32), affine=affine), out_fp)
        print(f"Segmentation Done using {args.modality} model. Written to {out_fp}")


def main(args=None):
    parser = argparse.ArgumentParser(description="Thalamic Nuclei Segmentation Inference")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input data.")
    parser.add_argument('--model_weight', type=str, required=True, help="Path to the model weight")
    parser.add_argument('--out_dir', type=str, required=True, help="Path to the output directory")
    parser.add_argument('--modality', type=str, required=True, choices=['fgatir', 'mprage', 't1map'], 
                        help="Input modality: fgatir, mprage, or t1map.")
    parser.add_argument('--device', '-d', type=str, default='gpu', choices=['gpu', 'cpu'], help="Specify device to use.")
    args = parser.parse_args(args if args is not None else sys.argv[1:])

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Error: The specified data path '{args.data_path}' does not exist.")

    if not os.path.exists(args.model_weight):
        raise FileNotFoundError(f"Error: The specified model weight file '{args.model_weight}' does not exist.")

    if not os.path.exists(args.out_dir):
        print(f"Output directory '{args.out_dir}' does not exist. Creating it now...")
        os.makedirs(args.out_dir)

    if args.device == 'gpu':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print("Warning: No available GPU detected, automatically switching to CPU.")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    args.device = device

    test(args)
