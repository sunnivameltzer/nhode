# Experiments

This directory contains one subdirectory per experiment reported in the paper. Each experiment is self-contained: data generation, training, and result visualisation are all handled from within the experiment directory.

---

## Directory structure

```
experiments/
├── two_mass_two_spring_1d/               # Two-mass two-spring system in 1D
├── three_mass_three_nonlinear_spring_2d/ # Three-mass three-nonlinear-spring system in 2D
├── three_body_problem_3d/                # Three-body gravitational problem in 3D
└── two_mass_two_spring_learning_ic/      # Two-mass two-spring with learned initial conditions
```

Each experiment directory follows the same layout:

```
<experiment>/
├── run_training.py          # Training entry point
├── train.py                 # Core training loop
├── models.py                # Model definitions
├── vector_fields.py         # ODE right-hand sides
├── utilities.py             # Data generation and shared helpers
├── visualize_results.ipynb  # Notebook for loading results and producing figures
└── configs/
    ├── hnn_pot/             # NHODE potential-energy model
    ├── hnn_tot/             # NHODE total-energy model
    ├── node_phys/           # NODE physics-informed baseline
    └── node_vanilla/        # NODE vanilla baseline
```

> For the three-body problem, each model variant is split into two coordinate representations:
> `hnn_pot_abs` / `hnn_pot_rel`, `hnn_tot_abs` / `hnn_tot_rel`, etc.

---

## Training

Each experiment is trained using `run_training.py` inside the corresponding experiment directory. Configuration files are stored under `configs/<model_variant>/` and specify the training seed, model hyperparameters, data-generation settings, optimisation settings, and checkpoint output directory.

| Model variant | Config directory |
|---|---|
| NODE, vanilla | `configs/node_vanilla/` |
| NODE, physics-informed | `configs/node_phys/` |
| NHODE, total-energy model | `configs/hnn_tot/` |
| NHODE, potential-energy model | `configs/hnn_pot/` |

The general command pattern is:

```bash
cd experiments/<experiment_name>
python run_training.py --config configs/<model_variant>/<config_file>.yaml
```

**Example** — NHODE potential-energy model, three-body problem, seed 0, relative coordinates:

```bash
cd experiments/three_body_problem_3d
python run_training.py --config configs/hnn_pot_rel/hnn_pot_0.yaml
```

The numbered configuration files correspond to independent random seeds (`hnn_pot_0.yaml` through `hnn_pot_9.yaml`).

### Running on a cluster (Slurm)

A template Slurm submission script is provided at [`docs/slurm_example.slurm`](../docs/slurm_example.slurm). Before using it, set `PROJECT_ROOT` and `CONDA_ENV` to match your local setup, then submit from the repository root:

```bash
sbatch docs/slurm_example.slurm
```

---

## Visualising results

Once training is complete, open `visualize_results.ipynb` in the relevant experiment directory:

```bash
jupyter notebook experiments/<experiment_name>/visualize_results.ipynb
```

The notebook loads saved checkpoints and metadata and produces the figures reported in the paper. Ensure that the checkpoint paths in the notebook match the `savedir` values set in the configuration files you used.

---

<!-- ## Checkpoints

The trained model checkpoints used to generate the paper results are archived on Zenodo: **[Zenodo DOI]**. To retrain from scratch, run all configuration files for the desired experiment as described above.

--- -->
## Checkpoints

The trained model checkpoints used to generate the paper results are saved under ```<experiment_name>/checkpoints/```. To retrain from scratch, run all configuration files for the desired experiment as described above.

---

## Paper result mapping

All figures and tables with the paper results were generated from trained model checkpoints using the `visualize_results.ipynb` notebook in the corresponding experiment directory.

### Experiment 1 — Two-mass two-spring system (1D)

**Directory:** `experiments/two_mass_two_spring_1d/`

| Figure / Table | Script / Notebook | Notes |
|---|---|---|
| Figure 3 | `visualize_results.ipynb` | Generates the prediction rollout and comparison plots for the two-mass two-spring system. |
| Table 4 | `visualize_results.ipynb` | Generates the two-mass two-spring entries in the summary error table. |

### Experiment 2 — Three-mass three-nonlinear-spring system (2D)

**Directory:** `experiments/three_mass_three_nonlinear_spring_2d/`

| Figure / Table | Script / Notebook | Notes |
|---|---|---|
| Figure 4 | `visualize_results.ipynb` | Generates the prediction rollout and comparison plots for the nonlinear three-mass spring system. |
| Table 2 | `visualize_results.ipynb` | Generates the entries in the table comparing short- and long-horizon errors for the different methods. |
| Table 4 | `visualize_results.ipynb` | Generates the nonlinear three-mass spring entries in the summary error table. |

### Experiment 3 — Three-body problem (3D)

**Directory:** `experiments/three_body_problem_3d/`

| Figure / Table | Script / Notebook | Notes |
|---|---|---|
| Figure 5 | `visualize_results.ipynb` | Generates the prediction rollout and comparison plots for the three-body problem. |
| Table 3 | `visualize_results.ipynb` | Generates the entries in the table comparing short- and long-horizon errors for the different methods. |
| Table 4 | `visualize_results.ipynb` | Generates the three-body problem entries in the summary error table. |
| Supplementary Figure 7 | `epsilon_range_test.ipynb` | Generates the supplementary result used to determine the value of the softening parameter $\varepsilon$. |

### Experiment 4 — Two-mass two-spring system with learned initial conditions

**Directory:** `experiments/two_mass_two_spring_learning_ic/`

| Figure / Table | Script / Notebook | Notes |
|---|---|---|
| Figure 6 | `visualize_results.ipynb` | Generates the prediction rollout and comparison plots for the two-mass two-spring system with learned initial conditions. |
