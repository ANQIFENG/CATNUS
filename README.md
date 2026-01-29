# CATNUS: Coordinate-Aware Thalamic Nuclei Segmentation Using T1-Weighted MRI [[Paper]](https://arxiv.org/abs/2512.05329)


CATNUS is a deep learning-based method for fast and accurate thalamic nuclei segmentation from T1-weighted MRI. It can distinguish **13 thalamic nuclei classes** to enable detailed structural and neuroanatomical analysis. CATNUS supports three types of T1-weighted MRI as inputs:
- **MPRAGE** (Magnetization Prepared Rapid Gradient Echo)
- **FGATIR** (Fast Gray Matter Acquisition T1 Inversion Recovery, also known as white-matter-nulled MPRAGE)
- **T1 map** (Quantitative T1 Map)

## Our Pipeline 🧩

CATNUS performs the following processing steps:

### Preprocessing
1. **Brain Extraction**: HD-BET for skull stripping
2. **MNI Registration**: Register image to MNI space
3. **Bias Field Correction**: N4 bias field correction
4. **Brain Masking**: Generate brain and background masks
5. **Harmonization**: HACA3 harmonization (MPRAGE only)

### Segmentation
6. **Thalamic Nuclei Segmentation**: Coordinate-aware 3D U-Net inference in MNI space

### Post-processing
7. **Inverse Transform**: Apply inverse transformation to bring segmentation back to original space

### Important Notes

⭐**Segmentation Quality**: Although we provide segmentation results in both MNI and original space, we **highly recommend using the MNI space segmentation**. The inverse transformation to original space involves interpolation, which can significantly degrade segmentation quality due to the small size of thalamic nuclei. We also provide the registered MNI space image for your reference.

⭐**Harmonization Status**: Currently, HACA3 harmonization is only available for MPRAGE. Harmonization methods for FGATIR and T1 map are under development and will be extended in future releases.

⭐**T1 Map Calculation**: If you have paired MPRAGE and FGATIR images and want to calculate T1 maps, you can use our [T1map calculation pipeline](https://github.com/ANQIFENG/multi-TI-image-calc-pipeline)!

## How to Run 🏃

### Prerequisites
- **Hardware**: GPU is recommended; CPU is also supported
- **Singularity**: >= 3.5

### Installation

Download the Singularity container from GitHub Releases:

```bash
wget https://github.com/[your-org]/CATNUS/releases/download/v1.0.0/catnus_v1.0.0.sif
```

### Usage

Replace placeholder paths with actual input files and output directory. Remove `--nv` if using a CPU. Input files must be in NIfTI format (`.nii` or `.nii.gz`).

```bash
singularity run -e --nv catnus_v1.0.0.sif \
  --data_path /path/to/input.nii.gz \
  --out_dir /path/to/output \
  --modality [fgatir|mprage|t1map] \      # Choose one modality
  --device [gpu|cpu] \                    # Default: gpu
  --save_intermediate                     # Default: False
```

## Details 🧠

### Inputs

| Arg | Description | Required |
|-----|-------------|----------|
| `--data_path` | Path to the input T1-weighted MRI image (.nii or .nii.gz) | ✅ |
| `--out_dir` | Path to the output directory | ✅ |
| `--modality` | MRI sequence type: either `mprage`, `fgatir`, or `t1map` | ✅ |
| `--device` | Computation device: either `gpu` or `cpu` (default: `gpu`) | ⭕ |
| `--save_intermediate` | Save intermediate processing results (default: `False`) | ⭕ |
✅ Required; ⭕ Optional.

### Outputs

#### Output Structure

The output directory (`/path/to/output`) is organized into subdirectories:

```
/path/to/output
    └── proc
        └── [output NIfTI files]
    └── qa
        └── [QA images]
    └── tmp
        └── [intermediate processing files]
```

- **proc**: Stores the final output NIfTI files.
- **qa**: Stores QA images for quick result review.
- **tmp**: Stores intermediate processing files (only if `--save_intermediate` is set).

#### Output Files

##### Final Outputs in `proc/`

- `*_reg.nii.gz`: Image registered to MNI space
- `*_reg.mat`: Transformation matrix for registration
- `*_catnus-{modality}-mni.nii.gz`: Thalamic nuclei segmentation in MNI space
- `*_catnus-{modality}-org.nii.gz`: Thalamic nuclei segmentation in original space

##### Intermediate Files in `tmp/` (if `--save_intermediate` is set)

- `*_reg.nii.gz`: Image registered to MNI space
- `*_reg.mat`: Transformation matrix for registration
- `*_n4.nii.gz`: Image after N4 bias field correction
- `*_brain.nii.gz`: Brain-extracted image
- `*_brain_mask.nii.gz`: Brain mask in MNI space
- `*_bgmask.nii.gz`: Background mask in MNI space
- `*_wm_mask.nii.gz`: White matter mask in MNI space 
- `*_haca3.nii.gz`: Image after HACA3 harmonization 


#### Label and Color Table

The output segmentation labels 13 distinct thalamic nuclei, with `0` representing the background and `1-13` corresponding to specific nuclei labels as follows:

- `1`: Anterior Nucleus (AN)
- `2`: Central Lateral (CL)
- `3`: Center Median (CM)
- `4`: Lateral Dorsal (LD)
- `5`: Lateral Posterior (LP)
- `6`: Mediodorsal (MD)
- `7`: Anterior Pulvinar (PuA)
- `8`: Inferior Pulvinar (PuI)
- `9`: Ventral Anterior (VA)
- `10`: Ventral Lateral Anterior (VLA)
- `11`: Ventral Lateral Posterior (VLP)
- `12`: Ventral Posterior Lateral (VPL)
- `13`: Ventral Posterior Medial (VPM)

Each nucleus is uniquely identified by a color code to facilitate visual analysis of the segmentation results. The color table can be viewed and downloaded from: [CATNUS Color Table](catnus_color_table.txt).

## Sample Data 💾

We provide **sample data** from one subject for validation, approved by IRB. Please download and test!

**Original Space Images** (`data/original/`):
- **MPRAGE**: `mtbi_data_org_mprage.nii.gz`
- **FGATIR**: `mtbi_data_org_fgatir.nii.gz`

**MNI Space Images** (`data/mni/`):
- **T1 map**: `mtbi_data_mni_t1map.nii.gz`
- **MPRAGE**: `mtbi_data_mni_mprage.nii.gz`
- **FGATIR**: `mtbi_data_mni_fgatir.nii.gz`

**Note**: MPRAGE and FGATIR in original space are directly acquired images. MNI space data are registered to MNI space. The T1 map is subsequently calculated from the registered MPRAGE–FGATIR pairs rather than directly acquired. In practice, we **highly recommend** using MNI space data as inputs for much better performance. For detailed acquisition parameters, see [`data/DATA_INFO.md`](data/DATA_INFO.md).

## Citation 📄

If you use CATNUS in your research, please cite our paper:

```bibtex
@article{feng2025catnus,
  title={CATNUS: Coordinate-Aware Thalamic Nuclei Segmentation Using T1-Weighted MRI},
  author={Feng, Anqi and Bian, Zhangxing and Remedios, Samuel W and Hays, Savannah P and Dewey, Blake E and Colinco, Alexa and Zhuo, Jiachen and Benjamini, Dan and Prince, Jerry L},
  journal={arXiv preprint arXiv:2512.05329},
  year={2025}
}
```

## Contact 📧

For questions or support, please contact [afeng11@jhu.edu](mailto:afeng11@jhu.edu) or post through [GitHub Issues](https://github.com/[your-org]/CATNUS/issues).
