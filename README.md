# NHODE: Neural Hamiltonian Ordinary Differential Equations for Partially Observed Dynamical Systems

This repository contains the research code accompanying the paper:

> **[Paper title placeholder]**
> [Author list placeholder]
> *[Journal placeholder]*, [Year placeholder].
> DOI: [paper DOI]

---

## Overview

In many physical systems, some variables are are hidden or unobserved, yet these variables have an influence on the observed dynamics. This repository provides the implementation of neural Hamiltonian ordinary differential equations (NHODEs), a framework combining Hamiltonian neural networks and neural ODEs. The goal is to investigate how physicsal prioirs improves the learning of partially observed dynamical systems.

The code is structured as a collection of self-contained experiment scripts rather than a general-purpose library. It is intended to support reproducibility of the results reported in the paper.

---

## Repository structure

```
nhode/
├── experiments/
│   ├── two_mass_two_spring_1d/          # Two-mass two-spring system in 1D
│   ├── three_mass_three_nonlinear_spring_2d/  # Three-mass three-nonlinear-spring in 2D
│   ├── three_body_problem_3d/           # Three-body gravitational problem in 3D
│   └── two_mass_two_spring_learning_ic/ # Two-mass two-spring with learned initial conditions
│
│   Each experiment directory contains:
│       ├── run_training.py              # Training entry point
│       ├── train.py                     # Core training loop
│       ├── models.py                    # Model definitions
│       ├── vector_fields.py             # ODE right-hand sides
│       ├── utilities.py                 # Data generation and helpers
│       ├── visualize_results.ipynb      # Result visualisation notebook
│       └── configs/
│           ├── hnn_pot/                 # NHODE potential-energy model configs
│           ├── hnn_tot/                 # NHODE total-energy model configs
│           ├── node_phys/               # NODE physics-informed configs
│           └── node_vanilla/            # NODE vanilla configs
│
├── docs/
│   ├── paper_result_mapping.md          # Maps paper figures/tables to code
│   └── slurm_example.slurm              # Example Slurm script for cluster jobs
│
├── notebooks/
│   ├── simple_example.ipynb             # Quickstart notebook with simple example
│   └── advanced_example.ipynb           # Quickstart notebook with advanced example
│
├── requirements.txt                     # Minimal dependency list
└── README.md
```

---

## Installation

The code requires Python ≥ 3.10 and is based on [JAX](https://github.com/google/jax), [Equinox](https://github.com/patrick-kidger/equinox), and [Diffrax](https://github.com/patrick-kidger/diffrax).

### Using conda (recommended)

```bash
conda create -n nhode python=3.10
conda activate nhode
pip install -r requirements.txt
```


> **Note on JAX and GPU support.** The default `requirements.txt` installs CPU JAX. For GPU acceleration, install the appropriate JAX version for your CUDA version by following the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) before installing the remaining dependencies.

---

## Quick start

Each experiment is run from within its own directory. The general pattern is:

```bash
cd experiments/<experiment_name>
python run_training.py --config configs/<model_variant>/<config_file>.yaml
```

For example, to train the NHODE potential-energy model on the three-body problem (seed 0, relative coordinates):

```bash
cd experiments/three_body_problem_3d
python run_training.py --config configs/hnn_pot_rel/hnn_pot_0.yaml
```

Configuration files specify hyperparameters, architecture, data generation settings, and the output checkpoint directory. See any `.yaml` file under `configs/` for an example.

### Running on a cluster (Slurm)

An example Slurm submission script is provided in [docs/slurm_example.slurm](docs/slurm_example.slurm). Edit the `PROJECT_ROOT` and `CONDA_ENV` variables before submitting:

```bash
sbatch docs/slurm_example.slurm
```

---

## Reproducing paper results

The file [docs/paper_result_mapping.md](docs/paper_result_mapping.md) maps each figure and table in the paper to the script or notebook that produced it, including the relevant configuration files and any post-processing steps.

To reproduce all results from scratch, run the training jobs for all seeds and model variants for a given experiment, then open the corresponding `visualize_results.ipynb` notebook. The ten seeds per model variant correspond to the numbered configuration files (e.g., `hnn_pot_0.yaml` through `hnn_pot_9.yaml`).

---

## Model variants

| Short name | Class | Description |
|---|---|---|
| `hnn_pot` | NHODE, potential-energy | Assumes separable Hamiltonian, $\mathcal{H}(\mathbf{q}, \mathbf{p}) = T(\mathbf{p}) + V_\theta(\mathbf{q})$ and known form of the kinetic energy $T(\mathbf{p}) = \mathbf{p}^2/2m$, learns only the potential energy $V_\theta(\mathbf{q})$. |
| `hnn_tot` | NHODE, total-energy | Learns the full Hamiltonian $\mathcal{H}(\mathbf{q}, \mathbf{p})$. |
| `node_phys` | NODE, physics-informed | Baseline neural ODE model, assumes $\dot{\mathbf{q}} = \mathbf{p}/m$ and only learns $\dot{\mathbf{p}}$. |
| `node_vanilla` | NODE, vanilla | Standard neural ODE baseline, learns $(\dot{\mathbf{q}}, \dot{\mathbf{p}})$. |

For the systems with translational and rotational symmetry, variants are additionally distinguished by the coordinate representation used:

- `*_abs` — absolute Cartesian coordinates.
- `*_rel` — relative (pairwise) coordinates.

---

## Data, checkpoints, and saved outputs

Training data is generated at runtime; no external datasets are required. The trained model object, losses and other metadata for each run are written to files specified in each configuration file.

A complete archive of trained model checkpoints used in the paper is provided at:

> Zenodo: [Zenodo DOI]

---

## Hardware 

- All training jobs were run on a single GPU. The full set of experiments requires significant compute and is intended to be distributed across cluster nodes using the provided Slurm script as a template.
- CPU-only training is supported but will be substantially slower for the larger experiments.

---

## Citation

If you use this code in your work, please cite the paper and, if applicable, the code archive:

```bibtex
@article{[citekey],
  title   = {[Paper title placeholder]},
  author  = {[Author list placeholder]},
  journal = {[Journal placeholder]},
  year    = {[Year placeholder]},
  doi     = {[paper DOI]}
}
```

```bibtex
@software{[citekey]_code,
  title   = {{NHODE}: Neural Hamiltonian Ordinary Differential Equations for Partially Observed Dynamical Systems --- Code},
  author  = {[Author list placeholder]},
  year    = {[Year placeholder]},
  doi     = {[Zenodo DOI]},
  url     = {[repository URL]}
}
```

---

## License

This repository is released under the [LICENSE](LICENSE) license. See the `LICENSE` file for details.

---

## Contact and issues

For questions about the code or to report a bug, please open an issue on GitHub at [repository URL] or contact [contact email].

