# Hyperelastic Image Registration via Physics-Informed Neural Networks

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4.13+-red.svg)](https://github.com/google/jax)
[![Equinox](https://img.shields.io/badge/Equinox-0.10.6+-green.svg)](https://github.com/patrick-kidger/equinox)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a **single‑file implementation** of a complete pipeline for hyperelastic image registration and material parameter identification using physics‑informed neural networks (PINNs). It reproduces and extends the work of Gao & Desai (2010) – *“Estimating zero‑strain states of very soft tissue under gravity loading using digital image correlation”* – by replacing the two‑stage DIC+curve‑fitting approach with an end‑to‑end PINN that directly learns the displacement field and the Ogden material parameters from image data.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Outputs](#outputs)
- [Code Structure](#code-structure)
- [Methodology](#methodology)
- [Results](#results)
- [License](#license)
- [Citation](#citation)

## Overview

The pipeline consists of several interconnected components, all contained in a single Python script:

1. **Forward FEM simulation** of a soft tissue strip (porcine liver mimic) under gravity and uniaxial tension using a plane‑stress Ogden hyperelastic model.
2. **Computation of reaction forces** and incremental strain fields.
3. **Synthetic image generation** with realistic tissue texture (Perlin noise + network pattern) warped by the FEM displacement fields.
4. **Two PINN‑based inverse approaches**:
   - **v9**: material identification from displacement observations (as in previous work).
   - **Registration**: full image‑based identification – the novel contribution that replaces DIC.
5. **Extensive visualisation** including loss curves, parameter trajectories, image registration results, incremental strain plots, and deformed meshes.

The code is written in **JAX/Equinox** for automatic differentiation and JIT compilation, making it both efficient and differentiable end‑to‑end.

## Features

- **Single‑file convenience** – easy to run, modify, and share.
- **JIT‑compiled stress routines** – fast computation of PK1 stress and tangent stiffness.
- **Newton‑Raphson solver** with backtracking line search for the forward FEM.
- **Realistic synthetic image generation** – mimics liver surface with lobular patterns.
- **Multi‑phase PINN training** – anchors first with image data, then adds physics and force matching.
- **Comparison mode** – optionally trains a displacement‑based PINN (v9) for benchmarking.
- **Comprehensive diagnostics** – over 10 different plots to monitor convergence, accuracy, and physical consistency.

## Requirements

- Python 3.8 or later
- [JAX](https://github.com/google/jax) (with `jaxlib`)
- [Equinox](https://github.com/patrick-kidger/equinox)
- [Optax](https://github.com/deepmind/optax)
- NumPy, SciPy, Matplotlib

All dependencies can be installed via pip (see below).

## Installation

1. **Clone the repository** (or simply download the single script):
   ```bash
   git clone https://github.com/gk408829/Hyperelastic-Image-Registration.git
   cd hyperelastic-registration-pinn
   ```
2. **Create a virtual environment (recommended):**
```python
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
```
3. **Install dependencies**
```bash
pip install jax jaxlib equinox optax numpy scipy matplotlib
```
Note: On some systems you may need to install a CPU/GPU‑specific version of JAX – see JAX installation instructions.

## Usage

The script is self‑contained and can be run directly from the command line:
```bash
python hyperelastic_registration_pinn.py
```
This executes the full pipeline (forward simulation, image generation, registration PINN, and optionally v9 comparison). On a typical desktop, the full run takes about 10–15 minutes.

For a quick test (reduced mesh, fewer training iterations), use:
```bash
python hyperelastic_registration_pinn.py --quick
```
To skip the v9 comparison (registration only):
```bash
python hyperelastic_registration_pinn.py --skip-v9
```
All output figures are saved in the outputs/ directory, which is created automatically.

## Outputs

`01_synthetic_images.png`	Grid of synthetic tissue images at selected load steps.

`02_reference_mesh.png`	Reference texture overlaid with the FEM mesh.

`03_incremental_strain.png`	Incremental strain fields (ΔE_yy) – cf. Gao & Desai Fig. 5.

`04_registration_diagnostics.png`	Training diagnostics: loss curves, μ and α trajectories, parameter space.

`05_registration_result.png`	Reference, deformed, PINN‑warped, and error images.

`06_comparison_v9_vs_reg.png`	Comparison of μ and α convergence between displacement‑based (v9) and image‑based registration.

`07_force_displacement.png`	Reaction force vs. top displacement curve.

`08_deformed_mesh.png`	Deformed meshes at selected steps (gray = reference).

`09_pinn_displacement_error.png`	Heatmap of PINN displacement error relative to FEM ground truth.

## Code Structure

The entire pipeline is contained in a single Python file (hyperelastic_registration_pinn.py), organised into logical sections with clear comments and docstrings. The main sections are:

    Constitutive model – Ogden hyperelasticity in plane stress.

    FEM forward simulation – mesh generation, assembly, Newton solver.

    Reaction force computation – integration of tractions on the top boundary.

    Synthetic image generation – texture synthesis and warping.

    Observation model – displacement sampling for the v9 approach.

    PINN definition (SoftTissuePINN) – MLP with learnable material parameters.

    Loss functions – data loss, physics loss, boundary loss, multi‑step force loss.

    Training loops – for both v9 (displacement‑based) and registration (image‑based).

    Visualisation – all plotting functions.

    Main driver – command‑line argument parsing and pipeline orchestration.

This monolithic structure makes the code easy to distribute and run without worrying about multiple files, while the internal comments and type hints keep it maintainable.

## Methodology

### Forward Problem

A rectangular strip (15 mm × 25 mm) is meshed with triangular elements. The material follows an incompressible Ogden model:

```bash
W = (2μ/α²)(λ₁ᵃ + λ₂ᵃ + λ₃ᵃ − 3)
```

Gravity is applied incrementally, followed by tensile displacement of the top edge. The equilibrium equations are solved with a Newton–Raphson scheme.

### Inverse Problem (Registration PINN)

The PINN maps reference coordinates X → displacement u(X). The total loss is:

```bash
L = w_img·L_image + w_phys·L_physics + w_bc·L_BC + w_force·L_force
```

- `L_image`: intensity difference between the warped reference image and the observed deformed image (replaces DIC).

- `L_physics`: residual of the hyperelastic equilibrium equation div P + ρg = 0.

- `L_BC`: penalty on boundary conditions (fixed bottom, prescribed top displacement).

- `L_force`: multi‑step reaction force matching, which isolates material behaviour.

Training proceeds in three phases:

1. Anchor – only image and BC losses → learn a plausible displacement field.

2. Physics – add physics loss and force loss → begin material identification.

3. Identify – increase force weight to refine μ and α.

The material parameters are stored in log space to ensure positivity and are updated with a higher learning rate than the MLP weights.

## Results

## License

This project is licensed under the `MIT License` – see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bash
@misc{hyperelastic_registration_pinn2025,
  author = {Gaurav Khanal},
  title = {Hyperelastic Image Registration via Physics-Informed Neural Networks},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/Hyperelastic-Image-Registration-
  PINN}}
}
```

Additionally, consider citing the original paper:

```bash
@article{gao2010estimating,
  title={Estimating zero-strain states of very soft tissue under gravity loading using 
  digital image correlation},
  author={Gao, Z. and Desai, J.P.},
  journal={Medical Image Analysis},
  volume={14},
  number={2},
  pages={126--137},
  year={2010},
  publisher={Elsevier}
}
```

Happy coding! If you encounter any issues or have questions, please open an issue on GitHub.
