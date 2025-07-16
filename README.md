# GEPAR3D: Geometry Prior-Assisted Learning for 3D Tooth Segmentation
---
This is the official code for "GEPAR3D: Geometry Prior-Assisted Learning for 3D Tooth Segmentation" accepted for the 28th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI) 2025.

Complete code comming soon.

### Data
Datasets related to this paper:
- https://zenodo.org/records/15739014 - GEPAR3D T.Szczepański et al.
- https://www.nature.com/articles/s41467-022-29637-2  - Z.Cui et al.
- https://ditto.ing.unimore.it/toothfairy2/ - Bolelli et al.


### Details

- configs/data_split.json - exact patient IDS used for evaluation
- configs/experiment_config.yaml - configuration for experiments: table 1 and table 2
- configs/requirements.yaml - conda environment configuration
- scripts/ablation_study/train.py - script to train all experiments of ablation study, including proposed solution
- scripts/ablation_study/inference.py - script to run inference of models within ablation study, including proposed solution
- scripts/general_segmentation_methods/train.py - script to train all general segmentation methods, table 1
