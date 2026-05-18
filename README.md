# NHODE: Neural Hamiltonian Ordinary Differential Equations

This repository contains the code for the paper "Learning partially observed systems with neural Hamiltonian ordinary differential equations". The code is structured as a collection of self-contained experiment scripts rather than a general-purpose library. It is intended to support reproducibility of the results reported in the paper.

---


## Repository structure

```
nhode/
├── experiments/    # One subdirectory per experiment; see experiments/README.md
├── docs/           # Example slurm script
├── notebooks/      # Quickstart notebooks
├── requirements.txt
└── README.md
```

See [experiments/README.md](experiments/README.md) for the detailed layout of each experiment directory.

---

## Installation

The code requires Python ≥ 3.9.6 and is based on [JAX](https://github.com/google/jax), [Equinox](https://github.com/patrick-kidger/equinox), and [Diffrax](https://github.com/patrick-kidger/diffrax).

### Using conda

```bash
conda create -n nhode python=3.10
conda activate nhode
pip install -r requirements.txt
```


<!-- > **Note on JAX and GPU support.** The default `requirements.txt` installs CPU JAX. For GPU acceleration, install the appropriate JAX version for your CUDA version by following the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) before installing the remaining dependencies. -->

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

The configuration files specify training hyperparameters, model architecture, data generation settings, and the output checkpoint directory. See any `.yaml` file under `configs/` for an example.

### Running on a cluster (Slurm)

An example Slurm submission script is provided in [docs/slurm_example.slurm](docs/slurm_example.slurm). Change the `PROJECT_ROOT` and `CONDA_ENV` variables before submitting:

```bash
sbatch docs/slurm_example.slurm
```

---

## Reproducing paper results

See [experiments/README.md](experiments/README.md) for detailed instructions on running individual experiments and reproducing all paper figures and tables.

---

## Model variants

| Short name | Class | Description |
|---|---|---|
| `hnn_pot` | NHODE, potential-energy | Assumes separable Hamiltonian, $\mathcal{H}(\mathbf{q}, \mathbf{p}) = T(\mathbf{p}) + V_\theta(\mathbf{q})$ and known form of the kinetic energy $T(\mathbf{p}) = \mathbf{p}^2/2m$, learns only the potential energy $V_\theta(\mathbf{q})$. |
| `hnn_tot` | NHODE, total-energy | Learns the full Hamiltonian $\mathcal{H}(\mathbf{q}, \mathbf{p})$. |
| `node_phys` | NODE, physics-informed | Physics-informed neural ODE baseline, assumes $\dot{\mathbf{q}} = \mathbf{p}/m$ and only learns $\dot{\mathbf{p}}$. |
| `node_vanilla` | NODE, vanilla | Standard neural ODE baseline, learns $(\dot{\mathbf{q}}, \dot{\mathbf{p}})$. |

For the systems with translational and rotational symmetry, variants are additionally distinguished by the coordinate representation used:

- `*_abs` — absolute Cartesian coordinates.
- `*_rel` — relative (pairwise) coordinates.

---

## Data, checkpoints, and saved outputs

Training data is generated at runtime; no external datasets are required. The trained model object, losses and other metadata for each run are written to files specified in each configuration file.

<!-- A complete archive of trained model checkpoints used in the paper is provided at:

> Zenodo: [Zenodo DOI] -->

---

## Hardware 

- All training jobs were run on a single GPU. The full set of experiments requires significant compute and is intended to be distributed across cluster nodes using the provided Slurm script as a template.
- CPU-only training is supported but will be substantially slower for the larger experiments.

---

<!-- ## Citation

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

---
``` -->

<!-- 
## Contact and issues

For questions about the code or to report a bug, please contact sunniva.meltzer@sintef.no.
 -->
