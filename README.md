# Hackathon Workshop on Generative Modeling 8-12 July 2024

This repository contains our work for **Challenge 4: Non-Convex Optimization** from the Hackathon Workshop on Generative Modeling.

The event was part of the Isaac Newton Institute (INI) satellite programme on *Diffusions in machine learning: Foundations, generative models and non-linear filtering*.

**Organisers:** Stefano Bruno, Dong-Young Lim, Sotirios Sabanis, Sara Wade and Ying Zhang.

## Challenge 4: Non-Convex Optimization

The official challenge description and starting material are available in [`Challenge4.html`](Challenge4.html).

In this challenge, we explore optimisation methods for training generative/diffusion models in a non-convex setting. This repository contains baseline training code, GSAM-based experiments, testing scripts, SLURM job files, logs and checkpoints.

## Repository structure

```text
.
├── Challenge4.html          # Official challenge overview and starting point
├── environment_dmog.yml     # Conda environment file
├── data.py                  # Data loading and preprocessing utilities
├── train.py                 # Main/baseline training script
├── train_gsam.py            # Training script using GSAM-style optimisation
├── train_gsam_cross.py      # Additional GSAM/cross-experiment training script
├── test.py                  # Evaluation/testing script
├── adamw.py                 # AdamW optimiser implementation/configuration
├── label.py                 # Labelling or plotting utility script
├── theopoula.py             # Project-specific experiment/model code
├── gpu_train.slurm          # SLURM script for training on GPU
├── gpu_adamw.slurm          # SLURM script for AdamW experiments
├── gpu_test.slurm           # SLURM script for testing on GPU
├── gpu_theo.slurm           # Additional SLURM experiment script
├── ckpts/                   # Saved model checkpoints
├── logs/                    # Training logs
├── GSAM/                    # GSAM-related code
├── Hackathon/               # Hackathon starter/supporting material
└── THEO_POULA/              # Additional team/project files
```

## Running the code

### Baseline training

```bash
python train.py
```

### GSAM training

```bash
python train_gsam.py
```

or:

```bash
python train_gsam_cross.py
```

### Testing/evaluation

```bash
python test.py
```

## Running on a GPU cluster with SLURM

Example training job:

```bash
sbatch gpu_train.slurm
```

AdamW experiment:

```bash
sbatch gpu_adamw.slurm
```

Testing job:

```bash
sbatch gpu_test.slurm
```

## Outputs

Training outputs are saved in:

- `ckpts/` — model checkpoints
- `logs/` — training logs


## Contributors

Andrea Meda, Yiming Xi, Nikolaos Makras, Huizi Zhang.

## Acknowledgements

This work was completed as part of the Hackathon Workshop on Generative Modeling associated with the Isaac Newton Institute satellite programme.

