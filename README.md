# GEPAR3D: Geometry Prior-Assisted Learning for 3D Tooth Segmentation
---
This is the official code for "GEPAR3D: Geometry Prior-Assisted Learning for 3D Tooth Segmentation" accepted for the 28th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI) 2025.

- configs/data_split.json - exact patient IDS used for evaluation
- configs/experiment_config.yaml - configuration for experiments: table 1 and table 2
- configs/requirements.yaml - conda environment configuration
- scripts/ablation_study/train.py - script to train all experiments of ablation study, including proposed solution
- scripts/ablation_study/inference.py - script to run inference of models within ablation study, including proposed solution
- scripts/general_segmentation_methods/train.py - script to train all general segmentation methods, table 1
