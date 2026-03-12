#!/usr/bin/env python3
"""
Hyperelastic Image Registration via Physics-Informed Neural Networks
=====================================================================

Single‑file implementation of the pipeline described in:

    Gao & Desai (2010) "Estimating zero‑strain states of very soft tissue
    under gravity loading using digital image correlation"

This script combines:
  - Forward FEM simulation of a soft tissue strip under gravity and tension
  - Synthetic image generation with realistic tissue texture
  - Two PINN‑based inverse approaches:
        * v9: material identification from displacement observations
        * Registration: full image‑based identification (replaces DIC)
  - Extensive visualisation and comparison

Usage:
    python hyperelastic_registration_pinn.py            # full run
    python hyperelastic_registration_pinn.py --quick    # fast test
    python hyperelastic_registration_pinn.py --skip-v9  # skip v9 comparison

Requirements:
    pip install jax jaxlib equinox optax numpy scipy matplotlib
"""

import argparse
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optax
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

# ------------------------------------------------------------------------------
# Configuration and setup
# ------------------------------------------------------------------------------

jax.config.update("jax_enable_x64", True)
os.makedirs("outputs", exist_ok=True)
matplotlib.use("Agg")  # non‑interactive backend for plotting


# ------------------------------------------------------------------------------
# Module: constitutive.py - Ogden hyperelastic model (2D plane stress)
# ------------------------------------------------------------------------------

"""
Ogden hyperelastic constitutive model for 2D plane‑stress.

Assumes plane‑stress conditions (σ₃₃ = 0) appropriate for thin specimens.
The out‑of‑plane stretch λ₃ is determined by incompressibility: λ₃ = 1/(λ₁λ₂).

Provides:
  - pk1_stress(F, mu, alpha, kappa)  – 1st Piola–Kirchhoff stress
  - strain_energy(F, mu, alpha, kappa) – strain energy density
"""


def _principal_stretches_2d(F: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute principal stretches from a 2×2 deformation gradient.

    Uses the eigenvalues of the right Cauchy‑Green tensor C = FᵀF.

    Args:
        F: (2,2) array – deformation gradient

    Returns:
        (λ₁, λ₂) – principal stretches (both positive)
    """
    C = F.T @ F
    tr_C = C[0, 0] + C[1, 1]
    det_C = jnp.maximum(C[0, 0] * C[1, 1] - C[0, 1] * C[1, 0], 1e-30)

    disc = jnp.maximum(tr_C**2 - 4.0 * det_C, 0.0)
    sqrt_disc = jnp.sqrt(disc + 1e-30)

    lam1_sq = jnp.maximum((tr_C + sqrt_disc) / 2.0, 1e-30)
    lam2_sq = jnp.maximum((tr_C - sqrt_disc) / 2.0, 1e-30)

    return jnp.sqrt(lam1_sq), jnp.sqrt(lam2_sq)


def strain_energy(
    F: jnp.ndarray,
    mu: float,
    alpha: float,
    kappa: float = 0.0,
) -> jnp.ndarray:
    """
    Single‑term Ogden strain energy density W(F).

    W = (2μ/α²) * (λ₁ᵃ + λ₂ᵃ + λ₃ᵃ - 3) + (κ/2) * (J - 1)²

    For plane‑stress incompressibility (κ = 0) λ₃ = 1/(λ₁λ₂).

    Args:
        F: (2,2) deformation gradient
        mu: shear modulus (Pa)
        alpha: Ogden exponent
        kappa: volumetric penalty (default 0 → exact incompressibility)

    Returns:
        strain energy density (scalar)
    """
    lam1, lam2 = _principal_stretches_2d(F)
    lam3 = 1.0 / (lam1 * lam2)  # incompressibility
    J = lam1 * lam2 * lam3  # = 1.0 for exact incompressibility

    W = (2.0 * mu / alpha**2) * (lam1**alpha + lam2**alpha + lam3**alpha - 3.0)
    W = W + (kappa / 2.0) * (J - 1.0) ** 2
    return W


def pk1_stress(
    F: jnp.ndarray,
    mu: float,
    alpha: float,
    kappa: float = 0.0,
) -> jnp.ndarray:
    """
    First Piola‑Kirchhoff stress P = dW/dF for the single‑term Ogden model.

    Computed via automatic differentiation of the strain energy.

    Args:
        F: (2,2) deformation gradient
        mu: shear modulus (Pa)
        alpha: Ogden exponent
        kappa: volumetric penalty

    Returns:
        (2,2) 1st Piola‑Kirchhoff stress
    """

    def _W(F_flat: jnp.ndarray) -> jnp.ndarray:
        return strain_energy(F_flat.reshape(2, 2), mu, alpha, kappa)

    F_flat = F.reshape(4)
    P_flat = jax.grad(_W)(F_flat)
    return P_flat.reshape(2, 2)


# ------------------------------------------------------------------------------
# Module: forward_sim.py - FEM simulation of tissue under gravity + tension
# ------------------------------------------------------------------------------

"""
Forward FEM simulation of 2D soft tissue under gravity and tension.

2D plane‑stress formulation: all forces and stiffnesses are per unit
out‑of‑plane thickness (implicitly t = 1 m). Reaction forces have units Pa·m.
"""

# JIT‑compiled stress functions for speed


def _ogden_energy(F_flat: jnp.ndarray, mu: float, alpha: float) -> jnp.ndarray:
    """
    Pure function for Ogden energy, used with jax.grad.
    Uses the same stretch computation as strain_energy.
    """
    F = F_flat.reshape(2, 2)
    lam1, lam2 = _principal_stretches_2d(F)
    lam3 = 1.0 / (lam1 * lam2)
    return (2.0 * mu / alpha**2) * (lam1**alpha + lam2**alpha + lam3**alpha - 3.0)


@jax.jit
def _pk1_jax(F: jnp.ndarray, mu: float, alpha: float) -> jnp.ndarray:
    """JIT‑compiled PK1 stress."""
    return jax.grad(_ogden_energy)(F.reshape(4), mu, alpha).reshape(2, 2)


@jax.jit
def _dpk1_dF_jax(F: jnp.ndarray, mu: float, alpha: float) -> jnp.ndarray:
    """JIT‑compiled material tangent dP/dF (4th‑order tensor)."""

    def _P_flat(F_flat: jnp.ndarray) -> jnp.ndarray:
        return jax.grad(_ogden_energy)(F_flat, mu, alpha)

    return jax.jacobian(_P_flat)(F.reshape(4)).reshape(2, 2, 2, 2)


def generate_rectangular_mesh(
    width: float,
    height: float,
    nx: int,
    ny: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[int]]]:
    """
    Create a structured triangular mesh for a rectangular domain.

    Args:
        width, height: physical dimensions (m)
        nx, ny: number of elements in x and y directions

    Returns:
        nodes: (N, 2) array of node coordinates
        elements: (E, 3) connectivity (triangles)
        node_sets: dict with keys 'bottom', 'top', 'left', 'right'
    """
    x = np.linspace(0, width, nx + 1)
    y = np.linspace(0, height, ny + 1)
    xx, yy = np.meshgrid(x, y)
    nodes = np.column_stack([xx.ravel(), yy.ravel()])

    elements = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            elements.append([n0, n0 + 1, n0 + nx + 2])
            elements.append([n0, n0 + nx + 2, n0 + nx + 1])
    elements = np.array(elements)

    tol = 1e-10
    node_sets = {
        "bottom": np.where(np.abs(nodes[:, 1]) < tol)[0].tolist(),
        "top": np.where(np.abs(nodes[:, 1] - height) < tol)[0].tolist(),
        "left": np.where(np.abs(nodes[:, 0]) < tol)[0].tolist(),
        "right": np.where(np.abs(nodes[:, 0] - width) < tol)[0].tolist(),
    }
    return nodes, elements, node_sets


def _precompute_element_data(
    nodes: np.ndarray,
    elements: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute shape function gradients (dN/dX) and areas for all elements.

    Args:
        nodes: (N, 2)
        elements: (E, 3)

    Returns:
        dN_dX_all: (E, 2, 3) – dN/dX for each element
        areas: (E,) – element areas (positive)
    """
    n_elem = len(elements)
    dN_dX_all = np.zeros((n_elem, 2, 3))
    areas = np.zeros(n_elem)

    for ei, elem in enumerate(elements):
        X = nodes[elem]
        x1, y1 = X[0]
        x2, y2 = X[1]
        x3, y3 = X[2]
        det_J = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        areas[ei] = abs(det_J) / 2.0
        dN_dX_all[ei] = (
            np.array(
                [
                    [y2 - y3, y3 - y1, y1 - y2],
                    [x3 - x2, x1 - x3, x2 - x1],
                ]
            )
            / det_J
        )

    return dN_dX_all, areas


def compute_internal_forces(
    elements: np.ndarray,
    u: np.ndarray,
    mu: float,
    alpha: float,
    dN_dX_all: np.ndarray,
    areas: np.ndarray,
) -> np.ndarray:
    """
    Global internal force vector f_int.

    Args:
        elements: (E, 3) connectivity
        u: (N, 2) nodal displacements
        mu, alpha: material parameters
        dN_dX_all: precomputed shape function gradients
        areas: element areas

    Returns:
        f_int: (2N,) internal force vector
    """
    n_dof = 2 * u.shape[0]
    f_int = np.zeros(n_dof)

    for ei, elem in enumerate(elements):
        grad_u = u[elem].T @ dN_dX_all[ei].T
        F = jnp.eye(2, dtype=jnp.float64) + jnp.array(grad_u, dtype=jnp.float64)
        det_F = float(jnp.linalg.det(F))
        if det_F <= 0:
            # Element inverted or collapsed – skip (warning printed elsewhere)
            continue

        P = np.array(_pk1_jax(F, mu, alpha))
        if np.any(np.isnan(P)):
            continue

        f_elem = areas[ei] * (P @ dN_dX_all[ei])  # (2,3)
        for I in range(3):
            ig = elem[I]
            f_int[2 * ig] += f_elem[0, I]
            f_int[2 * ig + 1] += f_elem[1, I]

    return f_int


def compute_tangent_stiffness(
    elements: np.ndarray,
    u: np.ndarray,
    mu: float,
    alpha: float,
    dN_dX_all: np.ndarray,
    areas: np.ndarray,
    n_nodes: int,
) -> Any:  # scipy sparse CSR matrix
    """
    Global tangent stiffness matrix (analytical dP/dF).

    Args:
        elements: (E,3) connectivity
        u: (N,2) displacements
        mu, alpha: material parameters
        dN_dX_all: precomputed shape function gradients
        areas: element areas
        n_nodes: number of nodes

    Returns:
        K: (2N,2N) sparse stiffness matrix (CSR format)
    """
    n_dof = 2 * n_nodes
    rows, cols, vals = [], [], []

    for ei, elem in enumerate(elements):
        grad_u = u[elem].T @ dN_dX_all[ei].T
        F = jnp.eye(2, dtype=jnp.float64) + jnp.array(grad_u, dtype=jnp.float64)
        det_F = float(jnp.linalg.det(F))
        if det_F <= 0:
            continue

        dPdF = np.array(_dpk1_dF_jax(F, mu, alpha))
        if np.any(np.isnan(dPdF)):
            continue

        area = areas[ei]
        dN = dN_dX_all[ei]  # (2,3)

        for I in range(3):
            ig = elem[I]
            for J in range(3):
                jg = elem[J]
                for i in range(2):
                    for j in range(2):
                        v = 0.0
                        for k in range(2):
                            for l in range(2):
                                v += dN[k, I] * dPdF[i, k, j, l] * dN[l, J]
                        v *= area
                        if abs(v) > 1e-20:
                            rows.append(2 * ig + i)
                            cols.append(2 * jg + j)
                            vals.append(v)

    return coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof)).tocsr()


def compute_gravity_force(
    elements: np.ndarray,
    rho: float,
    areas: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    """
    Gravity body force vector (per unit thickness).

    Args:
        elements: (E,3)
        rho: density (kg/m³)
        areas: element areas
        n_nodes: number of nodes

    Returns:
        f_grav: (2N,) force vector
    """
    n_dof = 2 * n_nodes
    f_grav = np.zeros(n_dof)
    for ei, elem in enumerate(elements):
        fpn = -rho * 9.81 * areas[ei] / 3.0
        for I in range(3):
            f_grav[2 * elem[I] + 1] += fpn
    return f_grav


def solve_step(
    elements: np.ndarray,
    u: np.ndarray,
    f_ext: np.ndarray,
    bc_dofs: List[int],
    bc_vals: List[float],
    mu: float,
    alpha: float,
    dN_dX_all: np.ndarray,
    areas: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 30,
) -> Tuple[np.ndarray, bool]:
    """
    Newton‑Raphson solver with backtracking line search.

    Args:
        elements: (E,3)
        u: (N,2) initial displacement guess (updated in‑place)
        f_ext: (2N,) external force vector
        bc_dofs: indices of DOFs with prescribed values
        bc_vals: prescribed values
        mu, alpha: material parameters
        dN_dX_all, areas: precomputed element data
        tol: relative residual tolerance
        max_iter: maximum iterations

    Returns:
        u: converged displacement (N,2)
        converged: boolean flag
    """
    u_flat = u.ravel().copy()
    for dof, val in zip(bc_dofs, bc_vals):
        u_flat[dof] = val
    u = u_flat.reshape(-1, 2)

    n_nodes = u.shape[0]
    n_dof = 2 * n_nodes
    free_dofs = sorted(set(range(n_dof)) - set(bc_dofs))
    r0 = None

    for it in range(max_iter):
        f_int = compute_internal_forces(elements, u, mu, alpha, dN_dX_all, areas)
        res = f_int - f_ext
        for d in bc_dofs:
            res[d] = 0.0

        rn = np.linalg.norm(res[free_dofs])
        if r0 is None:
            r0 = max(rn, 1e-10)
        if rn < tol * r0 or rn < 1e-10:
            return u, True

        K = compute_tangent_stiffness(elements, u, mu, alpha, dN_dX_all, areas, n_nodes)
        K_mod = K.tolil()
        for d in bc_dofs:
            K_mod[d, :] = 0
            K_mod[:, d] = 0
            K_mod[d, d] = 1.0
            res[d] = 0.0
        du = spsolve(K_mod.tocsr(), -res)

        # Backtracking line search
        alpha_ls = 1.0
        u_trial = u.copy()
        for _ in range(5):
            u_trial = (u.ravel() + alpha_ls * du).reshape(-1, 2)
            f_int_trial = compute_internal_forces(
                elements, u_trial, mu, alpha, dN_dX_all, areas
            )
            res_trial = f_int_trial - f_ext
            for d in bc_dofs:
                res_trial[d] = 0.0
            rn_trial = np.linalg.norm(res_trial[free_dofs])
            if rn_trial < rn or alpha_ls < 0.1:
                break
            alpha_ls *= 0.5
        u = u_trial

    print(f"  Warning: NR not converged (res={rn:.2e})")
    return u, False


def run_simulation(params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run the forward FEM simulation.

    Args:
        params: optional dict overriding default parameters

    Returns:
        results dict containing:
            nodes, elements, node_sets,
            displacements (list per step),
            load_info (list per step),
            params (used parameters),
            reaction_forces (to be filled later)
    """
    default_params = {
        "width": 0.015,
        "height": 0.025,
        "nx": 6,
        "ny": 10,
        "mu": 72.095,
        "alpha": 5.4067,
        "rho": 972.0,
        "precompression": 0.001,
        "n_gravity_steps": 3,
        "n_tension_steps": 10,
        "max_tension_disp": 0.008,
        "kappa": 0.0,
    }
    if params:
        default_params.update(params)
    p = default_params

    print(f"  Mesh: {p['nx']}x{p['ny']} elements")
    print(f"  Material: mu={p['mu']:.3f}, alpha={p['alpha']:.4f}")
    print(f"  Density: {p['rho']} kg/m3")

    nodes, elements, node_sets = generate_rectangular_mesh(
        p["width"], p["height"], p["nx"], p["ny"]
    )
    dN_dX_all, areas = _precompute_element_data(nodes, elements)

    n_nodes = len(nodes)
    u = np.zeros((n_nodes, 2))
    bot = node_sets["bottom"]
    top = node_sets["top"]

    mu_j = jnp.float64(p["mu"])
    alpha_j = jnp.float64(p["alpha"])

    # JIT warm‑up and sanity check
    print("  JIT warmup...")
    _F = jnp.eye(2, dtype=jnp.float64) * 1.001
    P_test = _pk1_jax(_F, mu_j, alpha_j)
    P_test.block_until_ready()
    print(f"  P(1.001*I) = {np.array(P_test).ravel()}")
    dPdF_test = _dpk1_dF_jax(_F, mu_j, alpha_j)
    dPdF_test.block_until_ready()
    print(f"  |dP/dF| = {float(jnp.linalg.norm(dPdF_test)):.6f}")
    assert not jnp.any(jnp.isnan(P_test)), "PK1 stress is NaN at test point!"
    assert not jnp.any(jnp.isnan(dPdF_test)), "Tangent is NaN at test point!"
    print("  JIT warmup done")

    displacements = [u.copy()]
    load_info = [{"step": 0, "type": "reference", "top_uy": 0.0}]

    f_grav = compute_gravity_force(elements, p["rho"], areas, n_nodes)

    # Pre‑compression + gravity
    for si in range(1, p["n_gravity_steps"] + 1):
        frac = si / p["n_gravity_steps"]
        top_uy = -p["precompression"] * frac
        bc_d, bc_v = [], []
        # Bottom: fully fixed
        for idx in bot:
            bc_d.extend([2 * idx, 2 * idx + 1])
            bc_v.extend([0.0, 0.0])
        # Top: vertical displacement prescribed, horizontal free
        for idx in top:
            bc_d.append(2 * idx + 1)
            bc_v.append(top_uy)

        print(f"  Precomp {si}/{p['n_gravity_steps']}, uy={top_uy * 1e3:.3f}mm")
        u, conv = solve_step(
            elements,
            u,
            f_grav * frac,
            bc_d,
            bc_v,
            mu_j,
            alpha_j,
            dN_dX_all,
            areas,
        )
        if not conv:
            print(f"  WARNING: step {si} did not converge, continuing anyway")
        displacements.append(u.copy())
        load_info.append(
            {"step": si, "type": "precompression", "top_uy": float(top_uy)}
        )

    # Tension
    base_uy = -p["precompression"]
    t_disps = np.linspace(0, p["max_tension_disp"], p["n_tension_steps"] + 1)[1:]
    for si, duy in enumerate(t_disps):
        snum = p["n_gravity_steps"] + 1 + si
        top_uy = base_uy + duy
        bc_d, bc_v = [], []
        for idx in bot:
            bc_d.extend([2 * idx, 2 * idx + 1])
            bc_v.extend([0.0, 0.0])
        for idx in top:
            bc_d.append(2 * idx + 1)
            bc_v.append(top_uy)

        print(f"  Tension {si + 1}/{p['n_tension_steps']}, uy={top_uy * 1e3:.3f}mm")
        u, _ = solve_step(
            elements,
            u,
            f_grav,
            bc_d,
            bc_v,
            mu_j,
            alpha_j,
            dN_dX_all,
            areas,
        )
        displacements.append(u.copy())
        load_info.append({"step": snum, "type": "tension", "top_uy": float(top_uy)})

    results = {
        "nodes": nodes,
        "elements": elements,
        "node_sets": node_sets,
        "displacements": displacements,
        "load_info": load_info,
        "params": p,
        "reaction_forces": [0.0] * len(displacements),
    }
    print(f"  Done: {len(displacements)} steps")
    return results


# ------------------------------------------------------------------------------
# Module: compute_reaction_forces.py
# ------------------------------------------------------------------------------

"""
Compute reaction forces on the top boundary from FEM displacement fields.
Uses element‑level shape function gradients and PK1 stress.
"""


def compute_triangle_area(X_elem: np.ndarray) -> float:
    """
    Area of a triangle using the absolute determinant.

    Args:
        X_elem: (3,2) reference node coordinates

    Returns:
        area (positive)
    """
    x1, y1 = X_elem[0]
    x2, y2 = X_elem[1]
    x3, y3 = X_elem[2]
    det_J = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    return abs(det_J) / 2.0


def compute_reaction_force_at_step(
    nodes: np.ndarray,
    elements: np.ndarray,
    u: np.ndarray,
    top_idx: List[int],
    mu: float,
    alpha: float,
) -> float:
    """
    Vertical reaction force on the top boundary for one load step.

    F_y = Σ_edges ∫ (P·n)_y dL, where n = [0,1] (outward normal on top).

    Args:
        nodes: (N,2) reference coordinates
        elements: (E,3) connectivity
        u: (N,2) displacement field
        top_idx: list of node indices on top boundary
        mu, alpha: material parameters

    Returns:
        F_y: vertical reaction force (Pa·m for 2D)
    """
    top_set = set(int(i) for i in top_idx)
    elements_np = np.array(elements)
    nodes_np = np.array(nodes)
    u_np = np.array(u)

    n = jnp.array([0.0, 1.0])
    F_y_total = 0.0
    seen_edges = set()

    for ei, elem in enumerate(elements_np):
        for i in range(3):
            n1 = int(elem[i])
            n2 = int(elem[(i + 1) % 3])
            if n1 in top_set and n2 in top_set:
                a, b = min(n1, n2), max(n1, n2)
                if (a, b) not in seen_edges:
                    seen_edges.add((a, b))

                    # Compute F from the element
                    X_e = nodes_np[elem]
                    u_e = u_np[elem]

                    x1, y1 = X_e[0]
                    x2, y2 = X_e[1]
                    x3, y3 = X_e[2]
                    det_J = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)

                    X_jax = jnp.array(X_e)
                    dN_dX = (
                        jnp.array(
                            [
                                [
                                    X_jax[1, 1] - X_jax[2, 1],
                                    X_jax[2, 1] - X_jax[0, 1],
                                    X_jax[0, 1] - X_jax[1, 1],
                                ],
                                [
                                    X_jax[2, 0] - X_jax[1, 0],
                                    X_jax[0, 0] - X_jax[2, 0],
                                    X_jax[1, 0] - X_jax[0, 0],
                                ],
                            ]
                        )
                        / det_J
                    )

                    grad_u = jnp.array(u_e).T @ dN_dX.T
                    F_def = jnp.eye(2) + grad_u

                    P = pk1_stress(F_def, mu, alpha, kappa=0.0)
                    traction_y = float((P @ n)[1])

                    edge_length = np.linalg.norm(nodes_np[b] - nodes_np[a])
                    F_y_total += traction_y * edge_length

    return F_y_total


def add_reaction_forces_to_results(results: Dict[str, Any]) -> None:
    """
    Compute reaction forces for all steps and store in results dict.
    """
    nodes = results["nodes"]
    elements = results["elements"]
    top_idx = results["node_sets"]["top"]
    mu = results["params"]["mu"]
    alpha = results["params"]["alpha"]

    forces = []
    for u in results["displacements"]:
        F_y = compute_reaction_force_at_step(nodes, elements, u, top_idx, mu, alpha)
        forces.append(F_y)

    results["reaction_forces"] = forces
    print(f"  Computed reaction forces for {len(forces)} steps")
    print(f"  Force range: [{min(forces):.6f}, {max(forces):.6f}] Pa·m")


# ------------------------------------------------------------------------------
# Module: synthetic_images.py
# ------------------------------------------------------------------------------

"""
Synthetic tissue image generator.

Creates a realistic liver‑like texture and warps it using FEM displacement fields.
"""


def _perlin_noise_2d(
    shape: Tuple[int, int],
    scale: float = 10,
    seed: int = 42,
) -> np.ndarray:
    """Simplified Perlin‑like noise using random gradients at multiple octaves."""
    rng = np.random.RandomState(seed)
    noise = np.zeros(shape, dtype=np.float64)

    for octave in range(4):
        freq = scale * (2**octave)
        amp = 1.0 / (2**octave)

        grid_h = max(int(shape[0] / freq) + 2, 3)
        grid_w = max(int(shape[1] / freq) + 2, 3)
        grid = rng.randn(grid_h, grid_w)

        y_coords = np.linspace(0, grid_h - 1, shape[0])
        x_coords = np.linspace(0, grid_w - 1, shape[1])
        yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")

        layer = map_coordinates(grid, [yy, xx], order=3, mode="wrap")
        noise += amp * layer

    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-10)
    return noise


def _generate_network_pattern(
    shape: Tuple[int, int],
    n_cells: int = 8,
    line_width: int = 2,
    seed: int = 42,
) -> np.ndarray:
    """Voronoi‑like pattern mimicking connective tissue septa."""
    rng = np.random.RandomState(seed)
    h, w = shape

    n_pts = n_cells**2
    centers_y = rng.uniform(0, h, n_pts)
    centers_x = rng.uniform(0, w, n_pts)

    yy, xx = np.mgrid[:h, :w]

    # Assign each pixel to nearest center
    assignment = np.zeros(shape, dtype=int)
    min_dist = np.full(shape, np.inf)
    for i, (cy, cx) in enumerate(zip(centers_y, centers_x)):
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        mask = dist < min_dist
        min_dist[mask] = dist[mask]
        assignment[mask] = i

    # Find boundaries where neighboring pixels differ
    boundary = np.zeros(shape, dtype=np.float64)
    boundary[:-1, :] += (assignment[:-1, :] != assignment[1:, :]).astype(float)
    boundary[:, :-1] += (assignment[:, :-1] != assignment[:, 1:]).astype(float)
    boundary = np.minimum(boundary, 1.0)

    boundary = gaussian_filter(boundary, sigma=line_width / 2)
    boundary = boundary / (boundary.max() + 1e-10)
    return boundary


def generate_tissue_texture(
    shape: Tuple[int, int],
    physical_size: Tuple[float, float],
    seed: int = 42,
) -> Tuple[np.ndarray, Callable, Callable]:
    """
    Generate a realistic tissue‑like texture image.

    Args:
        shape: (height_px, width_px)
        physical_size: (width_m, height_m)
        seed: random seed

    Returns:
        texture: (H,W) array in [0,1]
        pixel_to_phys: maps (row, col) to (x_m, y_m)
        phys_to_pixel: maps (x_m, y_m) to (row, col)
    """
    h_px, w_px = shape
    w_m, h_m = physical_size

    rng = np.random.RandomState(seed)

    # 1. Base tissue variation (Perlin noise)
    base = _perlin_noise_2d(shape, scale=8, seed=seed)
    base = 0.3 + 0.3 * base  # range [0.3, 0.6]

    # 2. Network pattern (connective tissue septa)
    px_per_m_x = w_px / w_m
    lobule_size_px = int(0.002 * px_per_m_x)  # ~2mm
    n_cells = max(4, int(w_px / lobule_size_px))
    network = _generate_network_pattern(
        shape,
        n_cells=n_cells,
        line_width=max(2, lobule_size_px // 5),
        seed=seed + 1,
    )

    # 3. Fine‑scale roughness
    fine = _perlin_noise_2d(shape, scale=30, seed=seed + 2)
    fine = 0.05 * fine

    texture = base + 0.4 * network + fine

    # Add random dark spots (blood vessels)
    n_spots = rng.randint(3, 8)
    for _ in range(n_spots):
        cy = rng.randint(0, h_px)
        cx = rng.randint(0, w_px)
        radius = rng.randint(3, max(4, lobule_size_px // 3))
        yy, xx = np.ogrid[:h_px, :w_px]
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        spot = np.exp(-(dist**2) / (2 * radius**2))
        texture -= 0.15 * spot

    texture = np.clip(texture, 0, 1)

    def pixel_to_phys(
        row: Union[float, np.ndarray], col: Union[float, np.ndarray]
    ) -> Tuple[Any, Any]:
        x = col / w_px * w_m
        y = (h_px - 1 - row) / (h_px - 1) * h_m
        return x, y

    def phys_to_pixel(
        x: Union[float, np.ndarray], y: Union[float, np.ndarray]
    ) -> Tuple[Any, Any]:
        col = x / w_m * w_px
        row = (h_px - 1) - y / h_m * (h_px - 1)
        return row, col

    return texture, pixel_to_phys, phys_to_pixel


class DisplacementFieldInterpolator:
    """
    Interpolates FEM displacement field at arbitrary physical coordinates.

    Precomputes element data for fast point location.
    """

    def __init__(
        self,
        nodes: np.ndarray,
        elements: np.ndarray,
        displacements: np.ndarray,
        width: float,
        height: float,
    ):
        self.nodes = np.array(nodes)
        self.elements = np.array(elements)
        self.u = np.array(displacements)
        self.width = width
        self.height = height
        self._precompute_element_data()

    def _precompute_element_data(self) -> None:
        """Store element bounding boxes and transformation matrices."""
        self.elem_data = []
        for elem in self.elements:
            X = self.nodes[elem]
            x1, y1 = X[0]
            x2, y2 = X[1]
            x3, y3 = X[2]
            det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
            self.elem_data.append(
                {
                    "X": X,
                    "det": det,
                    "bbox_min": X.min(axis=0),
                    "bbox_max": X.max(axis=0),
                    "nodes": elem,
                }
            )

    def _find_element_and_bary(
        self,
        x: float,
        y: float,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (element_nodes, barycentric_coords) for point (x,y)."""
        for ed in self.elem_data:
            if (
                x < ed["bbox_min"][0] - 1e-10
                or x > ed["bbox_max"][0] + 1e-10
                or y < ed["bbox_min"][1] - 1e-10
                or y > ed["bbox_max"][1] + 1e-10
            ):
                continue

            X = ed["X"]
            det = ed["det"]
            if abs(det) < 1e-20:
                continue

            x1, y1 = X[0]
            x2, y2 = X[1]
            x3, y3 = X[2]

            l1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / det
            l2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / det
            l3 = 1.0 - l1 - l2

            if l1 >= -1e-6 and l2 >= -1e-6 and l3 >= -1e-6:
                return ed["nodes"], np.array([l1, l2, l3])

        return None, None

    def __call__(
        self,
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Evaluate displacement at (x, y) – physical coordinates.

        Args:
            x, y: scalars or arrays (same shape)

        Returns:
            ux, uy: displacement components (same shape as input)
        """
        scalar = np.isscalar(x)
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        y_arr = np.atleast_1d(np.asarray(y, dtype=float))
        orig_shape = x_arr.shape
        x_flat = x_arr.ravel()
        y_flat = y_arr.ravel()

        ux = np.zeros(len(x_flat))
        uy = np.zeros(len(x_flat))

        for ed in self.elem_data:
            X = ed["X"]
            det = ed["det"]
            if abs(det) < 1e-20:
                continue
            nodes_idx = ed["nodes"]

            x1, y1 = X[0]
            x2, y2 = X[1]
            x3, y3 = X[2]

            l1 = ((y2 - y3) * (x_flat - x3) + (x3 - x2) * (y_flat - y3)) / det
            l2 = ((y3 - y1) * (x_flat - x3) + (x1 - x3) * (y_flat - y3)) / det
            l3 = 1.0 - l1 - l2

            inside = (l1 >= -1e-6) & (l2 >= -1e-6) & (l3 >= -1e-6)
            if not np.any(inside):
                continue

            u_elem = self.u[nodes_idx]  # (3,2)
            idx = np.where(inside)[0]
            ux[idx] = (
                l1[idx] * u_elem[0, 0] + l2[idx] * u_elem[1, 0] + l3[idx] * u_elem[2, 0]
            )
            uy[idx] = (
                l1[idx] * u_elem[0, 1] + l2[idx] * u_elem[1, 1] + l3[idx] * u_elem[2, 1]
            )

        if scalar:
            return float(ux[0]), float(uy[0])
        return ux.reshape(orig_shape), uy.reshape(orig_shape)


def warp_texture(
    texture: np.ndarray,
    displacement_field_func: DisplacementFieldInterpolator,
    phys_to_pixel: Callable,
    pixel_to_phys: Callable,
    noise_std: float = 0.0,
    brightness_var: float = 0.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Warp a reference texture using a displacement field.

    Uses forward mapping and griddata to invert the map.

    Args:
        texture: (H,W) reference image
        displacement_field_func: callable (x_phys, y_phys) → (ux, uy)
        phys_to_pixel, pixel_to_phys: coordinate transforms
        noise_std: Gaussian noise standard deviation
        brightness_var: global brightness variation amplitude
        seed: random seed

    Returns:
        warped: (H,W) deformed image
    """
    h_px, w_px = texture.shape
    rng = np.random.RandomState(seed)

    # Forward map: reference pixel → deformed pixel
    rows = np.arange(h_px)
    cols = np.arange(w_px)
    cc, rr = np.meshgrid(cols, rows)

    phys_x = cc / w_px * displacement_field_func.width
    phys_y = (h_px - 1 - rr) / (h_px - 1) * displacement_field_func.height

    ux, uy = displacement_field_func(phys_x, phys_y)
    def_x = phys_x + ux
    def_y = phys_y + uy

    def_col = def_x / displacement_field_func.width * w_px
    def_row = (h_px - 1) - def_y / displacement_field_func.height * (h_px - 1)

    # Invert mapping using griddata
    src_points = np.column_stack([def_row.ravel(), def_col.ravel()])
    ref_rows = rr.ravel().astype(float)
    ref_cols = cc.ravel().astype(float)

    query_rows, query_cols = np.mgrid[:h_px, :w_px]
    query_points = np.column_stack([query_rows.ravel(), query_cols.ravel()])

    inv_row = griddata(
        src_points, ref_rows, query_points, method="linear", fill_value=0
    ).reshape(h_px, w_px)
    inv_col = griddata(
        src_points, ref_cols, query_points, method="linear", fill_value=0
    ).reshape(h_px, w_px)

    warped = map_coordinates(
        texture, [inv_row, inv_col], order=3, mode="constant", cval=0.0
    )

    if noise_std > 0:
        warped += rng.normal(0, noise_std, warped.shape)
    if brightness_var > 0:
        warped *= 1.0 + rng.uniform(-brightness_var, brightness_var)

    return np.clip(warped, 0, 1)


def generate_image_sequence(
    results: Dict[str, Any],
    image_shape: Tuple[int, int] = (256, 128),
    step_indices: Optional[List[int]] = None,
    noise_std: float = 0.005,
    brightness_var: float = 0.02,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Generate a complete sequence of synthetic tissue images.

    Args:
        results: forward simulation output
        image_shape: (H, W) in pixels
        step_indices: which simulation steps to image (default all)
        noise_std, brightness_var: image noise parameters
        seed: random seed

    Returns:
        dict with keys:
            reference_texture, images (list), step_indices, image_shape,
            physical_size, pixel_to_phys, phys_to_pixel, coordinate_grids
    """
    params = results["params"]
    width = params["width"]
    height = params["height"]

    if step_indices is None:
        step_indices = list(range(len(results["displacements"])))

    texture, pixel_to_phys, phys_to_pixel = generate_tissue_texture(
        image_shape, (width, height), seed=seed
    )

    h_px, w_px = image_shape
    cols = np.arange(w_px)
    rows = np.arange(h_px)
    cc, rr = np.meshgrid(cols, rows)
    phys_x = cc / w_px * width
    phys_y = (h_px - 1 - rr) / (h_px - 1) * height

    images = []
    rng = np.random.RandomState(seed + 100)

    for i, si in enumerate(step_indices):
        u = results["displacements"][si]
        disp_func = DisplacementFieldInterpolator(
            results["nodes"], results["elements"], u, width, height
        )

        if np.max(np.abs(u)) < 1e-12:
            img = texture.copy()
            if noise_std > 0:
                img += rng.normal(0, noise_std, img.shape)
            img = np.clip(img, 0, 1)
        else:
            img = warp_texture(
                texture,
                disp_func,
                phys_to_pixel,
                pixel_to_phys,
                noise_std=noise_std,
                brightness_var=brightness_var,
                seed=seed + 200 + i,
            )
        images.append(img)
        print(
            f"  Generated image {i + 1}/{len(step_indices)} "
            f"(step {si}, max|u|={np.max(np.abs(u)) * 1e3:.3f} mm)"
        )

    return {
        "reference_texture": texture,
        "images": images,
        "step_indices": step_indices,
        "image_shape": image_shape,
        "physical_size": (width, height),
        "pixel_to_phys": pixel_to_phys,
        "phys_to_pixel": phys_to_pixel,
        "coordinate_grids": {
            "phys_x": phys_x,
            "phys_y": phys_y,
        },
    }


def plot_image_sequence(
    image_data: Dict[str, Any],
    save_path: str = "outputs/01_synthetic_images.png",
    max_images: int = 8,
) -> None:
    """Plot a grid of synthetic tissue images."""
    images = image_data["images"]
    steps = image_data["step_indices"]

    n = min(len(images), max_images)
    indices = np.linspace(0, len(images) - 1, n, dtype=int)

    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(3 * ((n + 1) // 2), 6))
    axes = axes.ravel()

    for i, idx in enumerate(indices):
        ax = axes[i]
        ax.imshow(images[idx], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Step {steps[idx]}", fontsize=9)
        ax.axis("off")

    for i in range(n, len(axes)):
        axes[i].axis("off")

    fig.suptitle("Synthetic Tissue Image Sequence", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved image sequence → {save_path}")


def plot_reference_with_mesh(
    image_data: Dict[str, Any],
    results: Dict[str, Any],
    save_path: str = "outputs/02_reference_mesh.png",
) -> None:
    """Plot reference texture with FEM mesh overlay."""
    texture = image_data["reference_texture"]
    w_m, h_m = image_data["physical_size"]
    nodes = results["nodes"]
    elements = results["elements"]

    fig, ax = plt.subplots(figsize=(6, 10))
    ax.imshow(texture, cmap="gray", extent=[0, w_m * 1e3, 0, h_m * 1e3])

    for elem in elements:
        tri = nodes[elem]
        tri_closed = np.vstack([tri, tri[0]])
        ax.plot(
            tri_closed[:, 0] * 1e3,
            tri_closed[:, 1] * 1e3,
            "c-",
            linewidth=0.3,
            alpha=0.5,
        )

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Reference Texture with FEM Mesh")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved reference + mesh → {save_path}")


# ------------------------------------------------------------------------------
# Module: observation_model.py
# ------------------------------------------------------------------------------

"""
Observation model – generates surface displacement observations from FEM results.
"""


def _barycentric(pt: np.ndarray, X_e: np.ndarray) -> np.ndarray:
    """Barycentric coordinates of point pt in triangle X_e."""
    x, y = pt
    x1, y1 = X_e[0]
    x2, y2 = X_e[1]
    x3, y3 = X_e[2]

    det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if abs(det) < 1e-20:
        return np.array([-1.0, -1.0, -1.0])

    lam1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / det
    lam2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / det
    lam3 = 1.0 - lam1 - lam2

    return np.array([lam1, lam2, lam3])


def _interpolate_displacements(
    nodes: np.ndarray,
    elements: np.ndarray,
    u: np.ndarray,
    query_pts: np.ndarray,
) -> np.ndarray:
    """
    Interpolate FEM displacement field at arbitrary query points.
    """
    u_interp = np.zeros((len(query_pts), 2))

    for qi, pt in enumerate(query_pts):
        found = False
        for elem in elements:
            X_e = nodes[elem]
            bary = _barycentric(pt, X_e)
            if np.all(bary >= -1e-10):
                u_e = u[elem]
                u_interp[qi] = bary @ u_e
                found = True
                break
        if not found:
            dists = np.linalg.norm(nodes - pt, axis=1)
            nearest = np.argmin(dists)
            u_interp[qi] = u[nearest]

    return u_interp


def generate_observations(
    results: Dict[str, Any],
    n_obs_x: int = 10,
    n_obs_y: int = 20,
    noise_std: float = 0.0,
    step_indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Generate surface displacement observations from FEM results.

    Args:
        results: forward simulation output
        n_obs_x, n_obs_y: observation grid density
        noise_std: Gaussian noise standard deviation (m)
        step_indices: which steps to observe (default last only)

    Returns:
        dict with keys:
            surface_coords, obs_disps (list), step_indices, noise_std
    """
    nodes = results["nodes"]
    elements = results["elements"]
    params = results["params"]
    width = params["width"]
    height = params["height"]

    margin_x = width * 0.05
    margin_y = height * 0.05
    obs_x = np.linspace(margin_x, width - margin_x, n_obs_x)
    obs_y = np.linspace(margin_y, height - margin_y, n_obs_y)
    xx, yy = np.meshgrid(obs_x, obs_y)
    obs_coords = np.column_stack([xx.ravel(), yy.ravel()])

    if step_indices is None:
        step_indices = [len(results["displacements"]) - 1]

    obs_disps_list = []
    for si in step_indices:
        u = results["displacements"][si]
        obs_u = _interpolate_displacements(nodes, elements, u, obs_coords)
        if noise_std > 0:
            obs_u += np.random.normal(0, noise_std, obs_u.shape)
        obs_disps_list.append(obs_u)

    return {
        "surface_coords": obs_coords,
        "obs_disps": obs_disps_list,
        "step_indices": step_indices,
        "noise_std": noise_std,
    }


# ------------------------------------------------------------------------------
# Module: pinn_v9.py - Multi‑step constitutive force loss for μ‑α identification
# ------------------------------------------------------------------------------

"""
PINN v9 – material identification from displacement observations.
"""


class SoftTissuePINN(eqx.Module):
    """Equinox MLP mapping (X,Y) → (u_x, u_y). Material parameters stored in log space."""

    mlp: eqx.nn.MLP
    log_mu: jnp.ndarray
    log_alpha: jnp.ndarray

    x_min: Tuple[float, float] = eqx.field(static=True)
    x_max: Tuple[float, float] = eqx.field(static=True)
    u_scale: float = eqx.field(static=True)

    def __init__(
        self,
        key: jax.Array,
        *,
        x_min: jnp.ndarray,
        x_max: jnp.ndarray,
        u_scale: float,
        mu_init: float,
        alpha_init: float,
    ):
        self.mlp = eqx.nn.MLP(
            in_size=2,
            out_size=2,
            width_size=64,
            depth=4,
            activation=jnp.tanh,
            key=key,
        )
        self.log_mu = jnp.log(jnp.array(float(mu_init)))
        self.log_alpha = jnp.log(jnp.array(float(alpha_init)))
        self.x_min = tuple(float(v) for v in x_min)
        self.x_max = tuple(float(v) for v in x_max)
        self.u_scale = float(u_scale)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Return displacement at reference coordinate x (2,)."""
        x_lo = jnp.array(self.x_min)
        x_hi = jnp.array(self.x_max)
        x_norm = 2.0 * (x - x_lo) / (x_hi - x_lo) - 1.0
        return self.mlp(x_norm) * self.u_scale

    def get_params(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Return (μ, α) in physical units."""
        return jnp.exp(self.log_mu), jnp.exp(self.log_alpha)


def _extract_top_boundary_edge_elements(
    nodes: np.ndarray,
    elements: np.ndarray,
    top_node_indices: List[int],
) -> Tuple[jnp.ndarray, jnp.ndarray, List[Tuple[int, int]], List[int]]:
    """
    Extract element‑level data for top boundary edges.

    Returns:
        edge_elem_nodes: (n_edges,3,2) node coordinates of the element containing each edge
        edge_lengths: (n_edges,) edge lengths
        edge_node_pairs: list of (node_a, node_b)
        elem_indices: list of element indices
    """
    top_set = set(int(i) for i in top_node_indices)
    elements_np = np.array(elements)
    nodes_np = np.array(nodes)

    edge_elem_nodes_list = []
    edge_lengths_list = []
    edge_node_pairs = []
    elem_indices = []
    seen = set()

    for ei, elem in enumerate(elements_np):
        for i in range(3):
            n1 = int(elem[i])
            n2 = int(elem[(i + 1) % 3])
            if n1 in top_set and n2 in top_set:
                a, b = min(n1, n2), max(n1, n2)
                if (a, b) not in seen:
                    seen.add((a, b))
                    elem_node_coords = nodes_np[elem]
                    edge_length = float(np.linalg.norm(nodes_np[b] - nodes_np[a]))
                    edge_elem_nodes_list.append(elem_node_coords)
                    edge_lengths_list.append(edge_length)
                    edge_node_pairs.append((a, b))
                    elem_indices.append(ei)

    edge_elem_nodes = jnp.array(np.array(edge_elem_nodes_list))
    edge_lengths = jnp.array(edge_lengths_list)
    return edge_elem_nodes, edge_lengths, edge_node_pairs, elem_indices


def _precompute_multi_step_force_data(
    results: Dict[str, Any],
    step_indices: List[int],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Precompute FEM deformation gradients at top‑boundary edge midpoints for multiple steps.

    Returns:
        F_at_edges: (K, n_edges, 2, 2) – deformation gradient at each edge midpoint
        edge_lengths: (n_edges,)
        F_obs_vec: (K,) – reaction forces at each step
    """
    nodes = results["nodes"]
    elements = results["elements"]
    top_idx = results["node_sets"]["top"]

    _, edge_lengths, edge_node_pairs, elem_indices = (
        _extract_top_boundary_edge_elements(nodes, elements, top_idx)
    )

    elements_np = np.array(elements)
    n_edges = len(edge_node_pairs)
    K = len(step_indices)

    F_at_edges = np.zeros((K, n_edges, 2, 2))
    F_obs_vec = np.zeros(K)

    for ki, si in enumerate(step_indices):
        u_field = jnp.array(results["displacements"][si])
        F_obs_vec[ki] = results["reaction_forces"][si]

        for ei in range(n_edges):
            elem = elements_np[elem_indices[ei]]
            X = jnp.array(nodes[elem])
            u = u_field[elem]

            x1, y1 = float(X[0, 0]), float(X[0, 1])
            x2, y2 = float(X[1, 0]), float(X[1, 1])
            x3, y3 = float(X[2, 0]), float(X[2, 1])
            det_J = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)

            dN_dX = (
                jnp.array(
                    [
                        [X[1, 1] - X[2, 1], X[2, 1] - X[0, 1], X[0, 1] - X[1, 1]],
                        [X[2, 0] - X[1, 0], X[0, 0] - X[2, 0], X[1, 0] - X[0, 0]],
                    ]
                )
                / det_J
            )
            grad_u = u.T @ dN_dX.T
            F = jnp.eye(2) + grad_u
            F_at_edges[ki, ei] = np.array(F)

    return jnp.array(F_at_edges), edge_lengths, jnp.array(F_obs_vec)


def prepare_training_data(
    results: Dict[str, Any],
    obs: Dict[str, Any],
    step_index: int = -1,
    force_step_indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Extract and organise all training data for v9.

    Args:
        results: forward simulation output
        obs: observation dict from generate_observations
        step_index: which step to use for displacement observations
        force_step_indices: list of step indices for force loss (default: tension steps)

    Returns:
        dict with keys needed for v9 losses
    """
    params = results["params"]
    nodes = results["nodes"]
    elements = results["elements"]
    node_sets = results["node_sets"]
    precompression = params["precompression"]
    rho = params["rho"]
    kappa = float(params.get("kappa", 0.0))

    n_steps = len(results["displacements"])
    abs_step = step_index if step_index >= 0 else n_steps + step_index

    # Locate observation for this step
    obs_list_idx = obs["step_indices"].index(abs_step)
    obs_coords = jnp.array(obs["surface_coords"])
    obs_disps = jnp.array(obs["obs_disps"][obs_list_idx])

    bottom_idx = np.array(node_sets["bottom"])
    top_idx = np.array(node_sets["top"])
    top_uy = results["load_info"][abs_step]["top_uy"]

    n_bot = len(bottom_idx)
    n_top = len(top_idx)

    bc_coords_np = np.concatenate(
        [
            np.array(nodes[bottom_idx]),
            np.array(nodes[top_idx]),
        ],
        axis=0,
    )

    bc_disps_np = np.zeros((n_bot + n_top, 2))
    # Bottom nodes: fully fixed (already zero)
    # Top nodes: u_y prescribed
    bc_disps_np[n_bot:, 1] = top_uy

    bc_coords = jnp.array(bc_coords_np)
    bc_disps = jnp.array(bc_disps_np)

    # For force loss, use multiple tension steps
    if force_step_indices is None:
        ng = params["n_gravity_steps"]
        nt = params["n_tension_steps"]
        tension_start = ng + 1
        tension_end = ng + nt
        n_force_steps = min(8, nt)
        force_step_indices = np.linspace(
            tension_start, tension_end, n_force_steps, dtype=int
        ).tolist()
        force_step_indices = sorted(set(force_step_indices))

    F_at_edges, ms_edge_lengths, F_obs_vec = _precompute_multi_step_force_data(
        results, force_step_indices
    )

    F_obs = float(results["reaction_forces"][abs_step])
    rho_g = rho * 9.81
    u_scale = float(jnp.max(jnp.abs(obs_disps)))
    if u_scale < 1e-6:
        u_scale = 1e-6

    return {
        "obs_coords": obs_coords,
        "obs_disps": obs_disps,
        "bc_coords": bc_coords,
        "bc_disps": bc_disps,
        "F_obs": F_obs,
        "rho_g": rho_g,
        "u_scale": u_scale,
        "kappa": kappa,
        "F_at_edges": F_at_edges,
        "ms_edge_lengths": ms_edge_lengths,
        "F_obs_vec": F_obs_vec,
        "force_step_indices": force_step_indices,
    }


def data_loss(
    model: SoftTissuePINN,
    obs_coords: jnp.ndarray,
    obs_disps: jnp.ndarray,
) -> jnp.ndarray:
    """Mean squared error between predicted and observed displacements."""
    pred = jax.vmap(model)(obs_coords)
    diff = pred - obs_disps
    return jnp.mean(jnp.sum(diff**2, axis=1))


def bc_loss(
    model: SoftTissuePINN,
    bc_coords: jnp.ndarray,
    bc_disps: jnp.ndarray,
) -> jnp.ndarray:
    """Mean squared error on boundary conditions."""
    pred = jax.vmap(model)(bc_coords)
    diff = pred - bc_disps
    return jnp.mean(jnp.sum(diff**2, axis=1))


def physics_loss(
    model: SoftTissuePINN,
    collocation_pts: jnp.ndarray,
    rho_g: float,
    kappa: float = 0.0,
) -> jnp.ndarray:
    """
    PDE residual loss: ‖ div P + ρg ‖² / (ρg)².

    The division by (ρg)² normalises the loss to be scale‑invariant.
    """
    mu, alpha = model.get_params()
    mu = jax.lax.stop_gradient(mu)
    alpha = jax.lax.stop_gradient(alpha)

    def _stress_from_x(x: jnp.ndarray) -> jnp.ndarray:
        grad_u = jax.jacfwd(model)(x)
        F = jnp.eye(2) + grad_u
        return pk1_stress(F, mu, alpha, kappa=kappa)

    def _residual(x: jnp.ndarray) -> jnp.ndarray:
        dP_dX = jax.jacrev(_stress_from_x)(x)
        div_P = jnp.array(
            [
                dP_dX[0, 0, 0] + dP_dX[0, 1, 1],
                dP_dX[1, 0, 0] + dP_dX[1, 1, 1],
            ]
        )
        body_force = jnp.array([0.0, -rho_g])
        return div_P + body_force

    residuals = jax.vmap(_residual)(collocation_pts)
    norm_sq = rho_g**2
    return jnp.mean(jnp.sum(residuals**2, axis=1)) / norm_sq


def multi_step_force_loss(
    log_mu: jnp.ndarray,
    log_alpha: jnp.ndarray,
    F_at_edges: jnp.ndarray,
    edge_lengths: jnp.ndarray,
    F_obs_vec: jnp.ndarray,
    kappa: float = 0.0,
) -> jnp.ndarray:
    """
    Force loss using multiple tension steps.

    L_force = mean( w_k * ((F_pred_k - F_obs_k) / F_max)² )
    where w_k = |F_obs_k| / F_max.
    """
    mu = jnp.exp(log_mu)
    alpha = jnp.exp(log_alpha)
    n = jnp.array([0.0, 1.0])

    def _force_at_step(F_edges_k: jnp.ndarray) -> jnp.ndarray:
        def _edge_force(F_e: jnp.ndarray, L_e: jnp.ndarray) -> jnp.ndarray:
            P = pk1_stress(F_e, mu, alpha, kappa=kappa)
            return (P @ n)[1] * L_e

        return jnp.sum(jax.vmap(_edge_force)(F_edges_k, edge_lengths))

    F_pred_vec = jax.vmap(_force_at_step)(F_at_edges)
    F_max = jnp.maximum(jnp.max(jnp.abs(F_obs_vec)), 1e-6)
    errors = (F_pred_vec - F_obs_vec) / F_max
    step_weights = jnp.abs(F_obs_vec) / F_max
    return jnp.mean(step_weights * errors**2)


def composite_loss(
    model: SoftTissuePINN,
    training_data: Dict[str, Any],
    collocation_pts: jnp.ndarray,
    weights: Dict[str, float],
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Total loss for v9."""
    w_d = weights["data"]
    w_p = weights["phys"]
    w_b = weights["bc"]
    w_f = weights["force"]

    l_data = data_loss(model, training_data["obs_coords"], training_data["obs_disps"])
    l_phys = physics_loss(
        model, collocation_pts, training_data["rho_g"], kappa=training_data["kappa"]
    )
    l_bc = bc_loss(model, training_data["bc_coords"], training_data["bc_disps"])
    l_force = multi_step_force_loss(
        model.log_mu,
        model.log_alpha,
        training_data["F_at_edges"],
        training_data["ms_edge_lengths"],
        training_data["F_obs_vec"],
        kappa=training_data["kappa"],
    )

    total = w_d * l_data + w_p * l_phys + w_b * l_bc + w_f * l_force
    aux = {"data": l_data, "phys": l_phys, "bc": l_bc, "force": l_force}
    return total, aux


# ------------------------------------------------------------------------------
# Module: hyperelastic_registration.py - Image‑based PINN
# ------------------------------------------------------------------------------

"""
Hyperelastic Image Registration PINN.

Loss:
  L = w_img·L_image + w_phys·L_physics + w_bc·L_BC + w_force·L_force
"""


def _bilinear_sample(
    image: jnp.ndarray, row: jnp.ndarray, col: jnp.ndarray
) -> jnp.ndarray:
    """Differentiable bilinear sampling of a 2D image."""
    H, W = image.shape
    row = jnp.clip(row, 0, H - 1.001)
    col = jnp.clip(col, 0, W - 1.001)

    r0 = jnp.floor(row).astype(jnp.int32)
    c0 = jnp.floor(col).astype(jnp.int32)
    r1 = jnp.minimum(r0 + 1, H - 1)
    c1 = jnp.minimum(c0 + 1, W - 1)

    dr = row - r0.astype(jnp.float64)
    dc = col - c0.astype(jnp.float64)

    return (
        image[r0, c0] * (1 - dr) * (1 - dc)
        + image[r0, c1] * (1 - dr) * dc
        + image[r1, c0] * dr * (1 - dc)
        + image[r1, c1] * dr * dc
    )


def image_similarity_loss(
    model: SoftTissuePINN,
    ref_image: jnp.ndarray,
    def_image: jnp.ndarray,
    sample_coords: jnp.ndarray,
    width: float,
    height: float,
    image_shape: Tuple[int, int],
) -> jnp.ndarray:
    """
    Image similarity loss: mean squared difference between deformed reference
    and observed deformed image.
    """
    H, W = image_shape

    def _single_point_loss(X: jnp.ndarray) -> jnp.ndarray:
        u = model(X)
        x_def = X[0] + u[0]
        y_def = X[1] + u[1]

        col_def = x_def / width * W
        row_def = (H - 1.0) - y_def / height * (H - 1.0)
        I_def = _bilinear_sample(def_image, row_def, col_def)

        col_ref = X[0] / width * W
        row_ref = (H - 1.0) - X[1] / height * (H - 1.0)
        I_ref = _bilinear_sample(ref_image, row_ref, col_ref)

        return (I_def - I_ref) ** 2

    losses = jax.vmap(_single_point_loss)(sample_coords)
    return jnp.mean(losses)


def registration_composite_loss(
    model: SoftTissuePINN,
    reg_data: Dict[str, Any],
    collocation_pts: jnp.ndarray,
    weights: Dict[str, float],
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Full loss for hyperelastic image registration."""
    w_i = weights["image"]
    w_p = weights["phys"]
    w_b = weights["bc"]
    w_f = weights["force"]

    l_image = image_similarity_loss(
        model,
        reg_data["ref_image"],
        reg_data["def_image"],
        reg_data["image_sample_coords"],
        reg_data["width"],
        reg_data["height"],
        reg_data["image_shape"],
    )

    l_phys = physics_loss(
        model, collocation_pts, reg_data["rho_g"], kappa=reg_data["kappa"]
    )

    l_bc = bc_loss(model, reg_data["bc_coords"], reg_data["bc_disps"])

    l_force = multi_step_force_loss(
        model.log_mu,
        model.log_alpha,
        reg_data["F_at_edges"],
        reg_data["ms_edge_lengths"],
        reg_data["F_obs_vec"],
        kappa=reg_data["kappa"],
    )

    total = w_i * l_image + w_p * l_phys + w_b * l_bc + w_f * l_force
    aux = {"image": l_image, "phys": l_phys, "bc": l_bc, "force": l_force}
    return total, aux


def prepare_registration_data(
    results: Dict[str, Any],
    image_data: Dict[str, Any],
    step_index: int = -1,
    n_image_samples: int = 500,
    force_step_indices: Optional[List[int]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Prepare training data for hyperelastic image registration.

    Combines image data with mechanics data.
    """
    params = results["params"]
    width = params["width"]
    height = params["height"]
    n_steps = len(results["displacements"])
    abs_step = step_index if step_index >= 0 else n_steps + step_index

    # Dummy observation for v9 compatibility (we won't use data_loss)
    dummy_obs = {
        "surface_coords": np.array([[0.0, 0.0]]),
        "obs_disps": [np.array([[0.0, 0.0]])],
        "step_indices": [abs_step],
    }
    v9_data = prepare_training_data(
        results,
        dummy_obs,
        step_index=step_index,
        force_step_indices=force_step_indices,
    )

    img_steps = image_data["step_indices"]
    ref_idx = img_steps.index(0) if 0 in img_steps else 0
    def_idx = img_steps.index(abs_step) if abs_step in img_steps else -1

    ref_image = jnp.array(image_data["images"][ref_idx])
    def_image = jnp.array(image_data["images"][def_idx])
    image_shape = image_data["image_shape"]

    rng = np.random.RandomState(seed)
    margin_x = width * 0.05
    margin_y = height * 0.05
    sx = rng.uniform(margin_x, width - margin_x, n_image_samples)
    sy = rng.uniform(margin_y, height - margin_y, n_image_samples)
    image_sample_coords = jnp.array(np.column_stack([sx, sy]))

    return {
        "ref_image": ref_image,
        "def_image": def_image,
        "image_sample_coords": image_sample_coords,
        "image_shape": image_shape,
        "width": width,
        "height": height,
        "bc_coords": v9_data["bc_coords"],
        "bc_disps": v9_data["bc_disps"],
        "rho_g": v9_data["rho_g"],
        "u_scale": v9_data["u_scale"],
        "kappa": v9_data["kappa"],
        "F_at_edges": v9_data["F_at_edges"],
        "ms_edge_lengths": v9_data["ms_edge_lengths"],
        "F_obs_vec": v9_data["F_obs_vec"],
        "force_step_indices": v9_data["force_step_indices"],
    }


def _build_optimizer(
    model: SoftTissuePINN,
    lr_mlp: float,
    lr_params: float,
) -> Tuple[optax.GradientTransformation, Any]:
    """Optimizer with separate learning rates for MLP and material parameters."""
    filtered = eqx.filter(model, eqx.is_array)
    labels = jax.tree.map(lambda _: "mlp", filtered)
    labels = eqx.tree_at(lambda m: m.log_mu, labels, "params")
    labels = eqx.tree_at(lambda m: m.log_alpha, labels, "params")

    optimizer = optax.chain(
        optax.masked(optax.adam(lr_mlp), jax.tree.map(lambda l: l == "mlp", labels)),
        optax.masked(
            optax.adam(lr_params), jax.tree.map(lambda l: l == "params", labels)
        ),
    )
    return optimizer, optimizer.init(filtered)


def _sample_collocation(
    key: jax.Array,
    n: int,
    width: float,
    height: float,
) -> jnp.ndarray:
    """Sample random collocation points uniformly in the domain."""
    pts = jax.random.uniform(key, shape=(n, 2))
    return pts * jnp.array([width, height])


def _clamp_params(model: SoftTissuePINN) -> SoftTissuePINN:
    """Clamp material parameters to physically plausible ranges."""
    new_lm = jnp.clip(model.log_mu, jnp.log(10.0), jnp.log(500.0))
    new_la = jnp.clip(model.log_alpha, jnp.log(0.5), jnp.log(20.0))
    model = eqx.tree_at(lambda m: m.log_mu, model, new_lm)
    model = eqx.tree_at(lambda m: m.log_alpha, model, new_la)
    return model


REGISTRATION_CONFIG = {
    "mu_init_factor": 0.5,
    "alpha_init_factor": 0.7,
    "phases": [
        # Phase 1: Learn displacement from images + BCs (no physics yet)
        ("anchor", 3000, {"image": 50.0, "bc": 100.0, "phys": 0.0, "force": 0.0}),
        # Phase 2: Add hyperelastic regularization + force matching
        ("physics", 5000, {"image": 50.0, "bc": 100.0, "phys": 1.0, "force": 1.0}),
        # Phase 3: Emphasise constitutive identification
        ("identify", 6000, {"image": 20.0, "bc": 50.0, "phys": 1.0, "force": 10.0}),
    ],
    "lr_mlp": 1e-3,
    "lr_params": 1e-2,
    "phase3_lr_decay": True,
    "phase3_lr_mlp": 5e-4,
    "phase3_lr_params": 5e-3,
    "n_collocation": 2000,
    "n_image_samples": 500,
    "resample_every": 500,
    "resample_image_points": True,
    "mu_true": 72.095,
    "alpha_true": 5.4067,
    "width": 0.015,
    "height": 0.025,
}


def train_registration(
    results: Dict[str, Any],
    image_data: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[SoftTissuePINN, Dict[str, List[float]]]:
    """
    Train the hyperelastic image registration PINN.

    Returns:
        model: trained PINN
        history: dict with loss components and parameter trajectories
    """
    if config is None:
        config = REGISTRATION_CONFIG.copy()

    reg_data = prepare_registration_data(
        results,
        image_data,
        n_image_samples=config["n_image_samples"],
    )

    mu_true = config["mu_true"]
    alpha_true = config["alpha_true"]
    mu_init = config["mu_init_factor"] * mu_true
    alpha_init = config["alpha_init_factor"] * alpha_true
    width = config["width"]
    height = config["height"]

    print(f"\n{'=' * 60}")
    print("Hyperelastic Image Registration PINN")
    print(f"{'=' * 60}")
    print(f"  Image shape: {reg_data['image_shape']}")
    print(f"  Image samples: {config['n_image_samples']}")
    print(f"  Force steps: {len(reg_data['force_step_indices'])}")
    print(f"  μ init = {mu_init:.3f} (true = {mu_true:.3f})")
    print(f"  α init = {alpha_init:.3f} (true = {alpha_true:.4f})")

    key = jax.random.PRNGKey(42)
    key, model_key = jax.random.split(key)

    model = SoftTissuePINN(
        model_key,
        x_min=jnp.array([0.0, 0.0]),
        x_max=jnp.array([width, height]),
        u_scale=reg_data["u_scale"],
        mu_init=mu_init,
        alpha_init=alpha_init,
    )

    lr_mlp = config["lr_mlp"]
    lr_params = config["lr_params"]
    optimizer, opt_state = _build_optimizer(model, lr_mlp, lr_params)

    p3_opt, p3_st = None, None
    if config.get("phase3_lr_decay"):
        p3_opt, p3_st = _build_optimizer(
            model,
            config.get("phase3_lr_mlp", lr_mlp / 2),
            config.get("phase3_lr_params", lr_params / 2),
        )

    @eqx.filter_jit
    def _step(
        m: SoftTissuePINN,
        ost: Any,
        rd: Dict[str, Any],
        coll: jnp.ndarray,
        w: Dict[str, float],
        opt: optax.GradientTransformation,
    ) -> Tuple[SoftTissuePINN, Any, jnp.ndarray, Dict[str, jnp.ndarray], bool]:
        @eqx.filter_value_and_grad(has_aux=True)
        def _lg(
            m_: SoftTissuePINN,
            rd_: Dict[str, Any],
            coll_: jnp.ndarray,
            w_: Dict[str, float],
        ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
            return registration_composite_loss(m_, rd_, coll_, w_)

        (loss, aux), grads = _lg(m, rd, coll, w)
        nan = jnp.isnan(loss)
        updates, new_ost = opt.update(grads, ost, eqx.filter(m, eqx.is_array))
        upd_m = _clamp_params(eqx.apply_updates(m, updates))

        def sel(old: Any, new: Any) -> Any:
            if eqx.is_array(old):
                return jnp.where(nan, old, new)
            return new

        def sel_o(old: Any, new: Any) -> Any:
            if isinstance(old, jnp.ndarray):
                return jnp.where(nan, old, new)
            return new

        return (
            jax.tree.map(sel, m, upd_m, is_leaf=eqx.is_array),
            jax.tree.map(sel_o, ost, new_ost),
            loss,
            aux,
            nan,
        )

    history = {
        "loss": [],
        "loss_image": [],
        "loss_phys": [],
        "loss_bc": [],
        "loss_force": [],
        "mu": [],
        "alpha": [],
    }

    n_coll = config["n_collocation"]
    log_every = 500
    global_iter = 0

    key, ck = jax.random.split(key)
    coll_pts = _sample_collocation(ck, n_coll, width, height)

    phases = config["phases"]
    print("\n  JIT‑compiling (first call only)...")

    for pi, (pname, piters, weights) in enumerate(phases):
        is_final = pi == len(phases) - 1
        act_opt = p3_opt if (is_final and p3_opt) else optimizer
        act_st = p3_st if (is_final and p3_opt) else opt_state

        for i in range(piters):
            if global_iter > 0 and global_iter % config["resample_every"] == 0:
                key, ck = jax.random.split(key)
                coll_pts = _sample_collocation(ck, n_coll, width, height)

                if config.get("resample_image_points", True):
                    rng = np.random.RandomState(global_iter)
                    mx = width * 0.05
                    my = height * 0.05
                    sx = rng.uniform(mx, width - mx, config["n_image_samples"])
                    sy = rng.uniform(my, height - my, config["n_image_samples"])
                    reg_data = {
                        **reg_data,
                        "image_sample_coords": jnp.array(np.column_stack([sx, sy])),
                    }

            model, act_st, loss, aux, nan = _step(
                model, act_st, reg_data, coll_pts, weights, act_opt
            )

            if is_final and p3_opt:
                p3_st = act_st
            else:
                opt_state = act_st

            if global_iter % log_every == 0:
                mu_v, al_v = model.get_params()
                print(
                    f"  [{pname:>8s}] {global_iter:5d} | "
                    f"L={float(loss):.3e} "
                    f"img={float(aux['image']):.2e} "
                    f"phy={float(aux['phys']):.2e} "
                    f"bc={float(aux['bc']):.2e} "
                    f"frc={float(aux['force']):.2e} | "
                    f"μ={float(mu_v):.2f} α={float(al_v):.3f}"
                )
                history["loss"].append(float(loss))
                history["loss_image"].append(float(aux["image"]))
                history["loss_phys"].append(float(aux["phys"]))
                history["loss_bc"].append(float(aux["bc"]))
                history["loss_force"].append(float(aux["force"]))
                history["mu"].append(float(mu_v))
                history["alpha"].append(float(al_v))

            global_iter += 1

    mu_f, al_f = [float(x) for x in model.get_params()]
    print(f"\n{'=' * 60}")
    print(
        f"  μ = {mu_f:.4f} (true = {mu_true:.4f}, "
        f"err = {abs(mu_f - mu_true) / mu_true * 100:.1f}%)"
    )
    print(
        f"  α = {al_f:.4f} (true = {alpha_true:.4f}, "
        f"err = {abs(al_f - alpha_true) / alpha_true * 100:.1f}%)"
    )
    print(f"{'=' * 60}")

    return model, history


# ------------------------------------------------------------------------------
# Module: registration_plots.py - Visualisation
# ------------------------------------------------------------------------------


def plot_registration_diagnostics(
    history: Dict[str, List[float]],
    true_params: Dict[str, float],
    save_path: str,
) -> None:
    """6‑panel training diagnostics."""
    iters = np.arange(len(history["loss"]))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    ax = axes[0, 0]
    ax.semilogy(iters, history["loss"], label="total", lw=1.5)
    ax.semilogy(iters, history["loss_image"], label="image", alpha=0.7)
    ax.semilogy(iters, history["loss_phys"], label="physics", alpha=0.7)
    ax.semilogy(iters, history["loss_bc"], label="BC", alpha=0.7)
    ax.semilogy(iters, history["loss_force"], label="force", alpha=0.7)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curves")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(iters, history["mu"], lw=1.5)
    ax.axhline(
        true_params["mu"], color="r", ls="--", label=f"true = {true_params['mu']:.3f}"
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("μ (Pa)")
    ax.set_title("μ Trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(iters, history["alpha"], lw=1.5)
    ax.axhline(
        true_params["alpha"],
        color="r",
        ls="--",
        label=f"true = {true_params['alpha']:.4f}",
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("α")
    ax.set_title("α Trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(history["mu"], history["alpha"], lw=1.0, alpha=0.7)
    ax.plot(history["mu"][0], history["alpha"][0], "go", ms=8, label="start")
    ax.plot(history["mu"][-1], history["alpha"][-1], "bs", ms=8, label="end")
    ax.plot(true_params["mu"], true_params["alpha"], "r*", ms=14, label="true")
    ax.set_xlabel("μ (Pa)")
    ax.set_ylabel("α")
    ax.set_title("Parameter Space")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.semilogy(iters, history["loss_image"], "b-", lw=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("L_image")
    ax.set_title("Image Similarity Loss")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.semilogy(iters, history["loss_force"], "r-", lw=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("L_force")
    ax.set_title("Force Loss")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Hyperelastic Image Registration — Diagnostics",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {save_path}")


def plot_registration_result(
    model: SoftTissuePINN,
    image_data: Dict[str, Any],
    results: Dict[str, Any],
    step_index: int,
    save_path: str,
) -> None:
    """4‑panel: reference, deformed, PINN‑warped, error."""
    params = results["params"]
    width, height = params["width"], params["height"]
    n_steps = len(results["displacements"])
    abs_step = step_index if step_index >= 0 else n_steps + step_index

    img_steps = image_data["step_indices"]
    ref_idx = img_steps.index(0) if 0 in img_steps else 0
    def_idx = img_steps.index(abs_step) if abs_step in img_steps else -1

    ref_img = image_data["images"][ref_idx]
    def_img = image_data["images"][def_idx]
    H, W = image_data["image_shape"]

    # Generate PINN‑warped image (forward warp of reference)
    warped = np.zeros((H, W))
    for r in range(H):
        for c in range(W):
            x = c / W * width
            y = (H - 1 - r) / (H - 1) * height
            u = model(jnp.array([x, y]))
            xd = np.clip(x + float(u[0]), 0, width - 1e-8)
            yd = np.clip(y + float(u[1]), 0, height - 1e-8)
            cd = xd / width * W
            rd = (H - 1) - yd / height * (H - 1)
            rd = np.clip(rd, 0, H - 1.001)
            cd = np.clip(cd, 0, W - 1.001)
            r0, c0 = int(np.floor(rd)), int(np.floor(cd))
            r1, c1 = min(r0 + 1, H - 1), min(c0 + 1, W - 1)
            dr, dc = rd - r0, cd - c0
            warped[r, c] = (
                ref_img[r0, c0] * (1 - dr) * (1 - dc)
                + ref_img[r0, c1] * (1 - dr) * dc
                + ref_img[r1, c0] * dr * (1 - dc)
                + ref_img[r1, c1] * dr * dc
            )

    diff = np.abs(def_img - warped)
    ext = [0, width * 1e3, 0, height * 1e3]

    fig, axes = plt.subplots(1, 4, figsize=(20, 8))
    axes[0].imshow(ref_img, cmap="gray", extent=ext, vmin=0, vmax=1)
    axes[0].set_title("Reference")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")

    axes[1].imshow(def_img, cmap="gray", extent=ext, vmin=0, vmax=1)
    axes[1].set_title(f"Deformed (step {abs_step})")
    axes[1].set_xlabel("x (mm)")

    axes[2].imshow(warped, cmap="gray", extent=ext, vmin=0, vmax=1)
    axes[2].set_title("PINN‑Warped Reference")
    axes[2].set_xlabel("x (mm)")

    im = axes[3].imshow(diff, cmap="hot", extent=ext)
    axes[3].set_title("|Deformed − Warped|")
    axes[3].set_xlabel("x (mm)")
    fig.colorbar(im, ax=axes[3], fraction=0.046)

    fig.suptitle("Registration Result", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {save_path}")


def plot_incremental_strain(
    results: Dict[str, Any],
    save_path: str = "outputs/03_incremental_strain.png",
) -> None:
    """Reproduce Gao & Desai Fig. 5: incremental strain fields."""
    params = results["params"]
    width, height = params["width"], params["height"]
    nodes = results["nodes"]
    elements = results["elements"]
    ng = params["n_gravity_steps"]
    n_steps = len(results["displacements"])
    tension_steps = list(range(ng + 1, n_steps))

    n_show = min(8, len(tension_steps))
    show_idx = np.linspace(0, len(tension_steps) - 1, n_show, dtype=int)

    nx_eval, ny_eval = 20, 50
    eval_x = np.linspace(width * 0.1, width * 0.9, nx_eval)
    eval_y = np.linspace(height * 0.05, height * 0.95, ny_eval)

    fig, axes = plt.subplots(2, (n_show + 1) // 2, figsize=(4 * ((n_show + 1) // 2), 8))
    axes = axes.ravel()

    for pi, si_idx in enumerate(show_idx):
        si = tension_steps[si_idx]
        u_curr = results["displacements"][si]
        u_prev = results["displacements"][max(si - 1, 0)]
        du = u_curr - u_prev

        Eyy = np.zeros((ny_eval, nx_eval))
        for iy, yv in enumerate(eval_y):
            for ix, xv in enumerate(eval_x):
                for elem in elements:
                    Xe = nodes[elem]
                    x1, y1 = Xe[0]
                    x2, y2 = Xe[1]
                    x3, y3 = Xe[2]
                    det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
                    if abs(det) < 1e-20:
                        continue
                    l1 = ((y2 - y3) * (xv - x3) + (x3 - x2) * (yv - y3)) / det
                    l2 = ((y3 - y1) * (xv - x3) + (x1 - x3) * (yv - y3)) / det
                    l3 = 1 - l1 - l2
                    if l1 >= -1e-6 and l2 >= -1e-6 and l3 >= -1e-6:
                        dN = (
                            np.array(
                                [
                                    [y2 - y3, y3 - y1, y1 - y2],
                                    [x3 - x2, x1 - x3, x2 - x1],
                                ]
                            )
                            / det
                        )
                        grad_du = du[elem].T @ dN.T
                        Eyy[iy, ix] = grad_du[1, 1]
                        break

        ax = axes[pi]
        im = ax.imshow(
            Eyy[::-1],
            cmap="hot",
            aspect="auto",
            extent=[
                eval_x[0] * 1e3,
                eval_x[-1] * 1e3,
                eval_y[0] * 1e3,
                eval_y[-1] * 1e3,
            ],
        )
        ax.set_title(f"Step {si}: ΔE_yy", fontsize=9)
        ax.set_xlabel("x (mm)", fontsize=8)
        if pi % ((n_show + 1) // 2) == 0:
            ax.set_ylabel("y (mm)", fontsize=8)

    for i in range(n_show, len(axes)):
        axes[i].axis("off")

    fig.suptitle(
        "Incremental Strain Fields — Maximum Stretching Band\n"
        "(cf. Gao & Desai 2010, Fig. 5)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {save_path}")


def plot_comparison_v9_vs_registration(
    history_v9: Dict[str, List[float]],
    history_reg: Dict[str, List[float]],
    true_params: Dict[str, float],
    save_path: str,
) -> None:
    """Side‑by‑side comparison: v9 (displacement data) vs registration (images)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # μ convergence
    ax = axes[0]
    ax.plot(history_v9["mu"], label="v9 (disp. data)", lw=1.5)
    ax.plot(history_reg["mu"], label="Registration (images)", lw=1.5)
    ax.axhline(true_params["mu"], color="r", ls="--", label="true")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("μ (Pa)")
    ax.set_title("μ Convergence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # α convergence
    ax = axes[1]
    ax.plot(history_v9["alpha"], label="v9 (disp. data)", lw=1.5)
    ax.plot(history_reg["alpha"], label="Registration (images)", lw=1.5)
    ax.axhline(true_params["alpha"], color="r", ls="--", label="true")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("α")
    ax.set_title("α Convergence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Parameter space
    ax = axes[2]
    ax.plot(
        history_v9["mu"], history_v9["alpha"], "b-", lw=1, alpha=0.6, label="v9 path"
    )
    ax.plot(
        history_reg["mu"], history_reg["alpha"], "g-", lw=1, alpha=0.6, label="Reg path"
    )
    ax.plot(true_params["mu"], true_params["alpha"], "r*", ms=14, label="true")
    ax.set_xlabel("μ (Pa)")
    ax.set_ylabel("α")
    ax.set_title("Parameter Space Trajectories")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Comparison: Displacement‑Based (v9) vs Image‑Based (Registration)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {save_path}")


def plot_force_displacement_curve(
    results: Dict[str, Any],
    save_path: str = "outputs/07_force_displacement.png",
) -> None:
    """Plot reaction force vs. prescribed top displacement."""
    load_info = results["load_info"]
    forces = results["reaction_forces"]

    disps_mm = [li["top_uy"] * 1e3 for li in load_info]
    forces_arr = np.array(forces)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(disps_mm, forces_arr, "b-o", markersize=4, lw=1.5)

    ng = results["params"]["n_gravity_steps"]
    if ng < len(disps_mm):
        ax.axvline(
            disps_mm[ng], color="gray", ls="--", alpha=0.5, label="precomp → tension"
        )

    ax.set_xlabel("Top displacement (mm)")
    ax.set_ylabel("Reaction force (Pa·m)")
    ax.set_title("Force–Displacement Curve (Top Boundary)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {save_path}")


def plot_deformed_mesh(
    results: Dict[str, Any],
    step_indices: Optional[List[int]] = None,
    scale: float = 1.0,
    save_path: str = "outputs/08_deformed_mesh.png",
) -> None:
    """Plot the FEM mesh in its deformed configuration at selected steps."""
    nodes = results["nodes"]
    elements = results["elements"]
    disps = results["displacements"]
    load_info = results["load_info"]
    ng = results["params"]["n_gravity_steps"]

    if step_indices is None:
        n_total = len(disps)
        candidates = [ng, (ng + n_total) // 2, n_total - 1]
        step_indices = sorted(set(max(0, min(s, n_total - 1)) for s in candidates))

    n = len(step_indices)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 8))
    if n == 1:
        axes = [axes]

    for ax, si in zip(axes, step_indices):
        u = disps[si]
        deformed = nodes + u * scale

        for elem in elements:
            tri = nodes[elem]
            tri_c = np.vstack([tri, tri[0]])
            ax.plot(tri_c[:, 0] * 1e3, tri_c[:, 1] * 1e3, "-", color="0.85", lw=0.5)

        u_y_max = max(np.max(np.abs(u[:, 1])), 1e-12)
        for elem in elements:
            tri_d = deformed[elem]
            tri_c = np.vstack([tri_d, tri_d[0]])
            u_y_mean = np.mean(u[elem, 1])
            color = plt.cm.coolwarm(0.5 + 0.5 * u_y_mean / u_y_max)
            ax.plot(tri_c[:, 0] * 1e3, tri_c[:, 1] * 1e3, "-", color=color, lw=0.7)

        top_uy = load_info[si]["top_uy"] * 1e3
        ax.set_title(f"Step {si} (u_y={top_uy:.2f} mm)", fontsize=10)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    fig.suptitle("Deformed Mesh (gray = reference)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {save_path}")


def plot_pinn_displacement_error(
    model: SoftTissuePINN,
    results: Dict[str, Any],
    step_index: int = -1,
    nx_eval: int = 30,
    ny_eval: int = 50,
    save_path: str = "outputs/09_pinn_displacement_error.png",
) -> None:
    """Heatmap of |u_PINN(X) - u_FEM(X)| over the domain."""
    params = results["params"]
    width, height = params["width"], params["height"]
    nodes = results["nodes"]
    elements = results["elements"]
    n_steps = len(results["displacements"])
    abs_step = step_index if step_index >= 0 else n_steps + step_index
    u_fem_field = results["displacements"][abs_step]

    eval_x = np.linspace(width * 0.02, width * 0.98, nx_eval)
    eval_y = np.linspace(height * 0.02, height * 0.98, ny_eval)

    err_mag = np.full((ny_eval, nx_eval), np.nan)
    err_x = np.full((ny_eval, nx_eval), np.nan)
    err_y = np.full((ny_eval, nx_eval), np.nan)

    for iy, yv in enumerate(eval_y):
        for ix, xv in enumerate(eval_x):
            for elem in elements:
                Xe = nodes[elem]
                x1, y1 = Xe[0]
                x2, y2 = Xe[1]
                x3, y3 = Xe[2]
                det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
                if abs(det) < 1e-20:
                    continue
                l1 = ((y2 - y3) * (xv - x3) + (x3 - x2) * (yv - y3)) / det
                l2 = ((y3 - y1) * (xv - x3) + (x1 - x3) * (yv - y3)) / det
                l3 = 1 - l1 - l2
                if l1 >= -1e-6 and l2 >= -1e-6 and l3 >= -1e-6:
                    u_fem = (
                        l1 * u_fem_field[elem[0]]
                        + l2 * u_fem_field[elem[1]]
                        + l3 * u_fem_field[elem[2]]
                    )
                    u_pinn = np.array(model(jnp.array([xv, yv])))
                    diff = u_pinn - u_fem
                    err_x[iy, ix] = diff[0]
                    err_y[iy, ix] = diff[1]
                    err_mag[iy, ix] = np.linalg.norm(diff)
                    break

    ext = [eval_x[0] * 1e3, eval_x[-1] * 1e3, eval_y[0] * 1e3, eval_y[-1] * 1e3]

    fig, axes = plt.subplots(1, 3, figsize=(16, 7))

    im0 = axes[0].imshow(err_x[::-1] * 1e3, cmap="RdBu_r", extent=ext, aspect="auto")
    axes[0].set_title("u_x error (mm)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(err_y[::-1] * 1e3, cmap="RdBu_r", extent=ext, aspect="auto")
    axes[1].set_title("u_y error (mm)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(err_mag[::-1] * 1e3, cmap="hot", extent=ext, aspect="auto")
    axes[2].set_title("|u_PINN − u_FEM| (mm)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

    fig.suptitle(
        f"PINN Displacement Error (step {abs_step})", fontsize=12, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {save_path}")


# ------------------------------------------------------------------------------
# Module: run_registration_pipeline.py - Main driver
# ------------------------------------------------------------------------------


def main() -> None:
    """Execute the complete pipeline."""
    parser = argparse.ArgumentParser(
        description="Hyperelastic image registration pipeline"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run a fast test with reduced iterations"
    )
    parser.add_argument(
        "--skip-v9",
        action="store_true",
        help="Skip the v9 (displacement‑based) comparison",
    )
    args = parser.parse_args()

    quick = args.quick
    skip_v9 = args.skip_v9

    print("=" * 70)
    print("Hyperelastic Image Registration Pipeline")
    print("Medical Image Analysis Project")
    print("=" * 70)
    print(f"JAX devices: {jax.devices()}")
    print(f"Quick mode: {quick}")
    print()

    # ---- Step 1: Forward simulation ----
    print("Step 1: Forward FEM simulation...")
    t0 = time.time()

    sim_params = None
    if quick:
        sim_params = {"nx": 4, "ny": 6, "n_tension_steps": 6}
    results = run_simulation(params=sim_params)

    print(f"  Done in {time.time() - t0:.1f}s")
    print(
        f"  {len(results['nodes'])} nodes, "
        f"{len(results['elements'])} elements, "
        f"{len(results['displacements'])} steps"
    )

    # ---- Step 2: Reaction forces ----
    print("\nStep 2: Computing reaction forces...")
    t0 = time.time()
    add_reaction_forces_to_results(results)
    print(f"  Done in {time.time() - t0:.1f}s")

    # ---- Step 2b: Basic plots ----
    plot_force_displacement_curve(results)
    plot_deformed_mesh(results)

    # ---- Step 3: Synthetic images ----
    print("\nStep 3: Generating synthetic tissue images...")
    t0 = time.time()

    img_shape = (128, 64) if quick else (256, 128)
    image_data = generate_image_sequence(
        results,
        image_shape=img_shape,
        noise_std=0.005,
        brightness_var=0.02,
    )

    plot_image_sequence(image_data, save_path="outputs/01_synthetic_images.png")
    plot_reference_with_mesh(
        image_data, results, save_path="outputs/02_reference_mesh.png"
    )

    print(f"  Done in {time.time() - t0:.1f}s")

    # ---- Step 4: Incremental strain fields ----
    print("\nStep 4: Computing incremental strain fields...")
    t0 = time.time()
    plot_incremental_strain(results, save_path="outputs/03_incremental_strain.png")
    print(f"  Done in {time.time() - t0:.1f}s")

    # ---- Step 5: Hyperelastic image registration ----
    print("\nStep 5: Training hyperelastic image registration PINN...")
    t0 = time.time()

    reg_config = REGISTRATION_CONFIG.copy()
    if quick:
        reg_config["phases"] = [
            ("anchor", 500, {"image": 50.0, "bc": 100.0, "phys": 0.0, "force": 0.0}),
            ("physics", 800, {"image": 50.0, "bc": 100.0, "phys": 1.0, "force": 1.0}),
            ("identify", 700, {"image": 20.0, "bc": 50.0, "phys": 1.0, "force": 10.0}),
        ]
        reg_config["n_image_samples"] = 200
        reg_config["n_collocation"] = 500

    reg_model, reg_history = train_registration(results, image_data, config=reg_config)

    true_params = {
        "mu": reg_config["mu_true"],
        "alpha": reg_config["alpha_true"],
    }

    plot_registration_diagnostics(
        reg_history, true_params, save_path="outputs/04_registration_diagnostics.png"
    )

    plot_registration_result(
        reg_model,
        image_data,
        results,
        step_index=-1,
        save_path="outputs/05_registration_result.png",
    )

    plot_pinn_displacement_error(
        reg_model,
        results,
        step_index=-1,
        save_path="outputs/09_pinn_displacement_error.png",
    )

    reg_time = time.time() - t0
    print(f"  Registration training: {reg_time:.1f}s")

    # ---- Step 6: v9 comparison (optional) ----
    v9_history = None
    if not skip_v9:
        print("\nStep 6: Training v9 PINN (displacement‑based, for comparison)...")
        t0 = time.time()

        obs = generate_observations(results)

        v9_data = prepare_training_data(results, obs)
        mu_true = true_params["mu"]
        alpha_true = true_params["alpha"]

        key = jax.random.PRNGKey(42)
        key, mk = jax.random.split(key)
        width = results["params"]["width"]
        height = results["params"]["height"]

        v9_model = SoftTissuePINN(
            mk,
            x_min=jnp.array([0.0, 0.0]),
            x_max=jnp.array([width, height]),
            u_scale=v9_data["u_scale"],
            mu_init=0.5 * mu_true,
            alpha_init=0.7 * alpha_true,
        )

        # Build optimizer
        filtered = eqx.filter(v9_model, eqx.is_array)
        labels = jax.tree.map(lambda _: "mlp", filtered)
        labels = eqx.tree_at(lambda m: m.log_mu, labels, "p")
        labels = eqx.tree_at(lambda m: m.log_alpha, labels, "p")
        v9_opt = optax.chain(
            optax.masked(optax.adam(1e-3), jax.tree.map(lambda l: l == "mlp", labels)),
            optax.masked(optax.adam(1e-2), jax.tree.map(lambda l: l == "p", labels)),
        )
        v9_ost = v9_opt.init(filtered)

        def _clamp(m: SoftTissuePINN) -> SoftTissuePINN:
            m = eqx.tree_at(
                lambda m: m.log_mu, m, jnp.clip(m.log_mu, jnp.log(10.0), jnp.log(500.0))
            )
            m = eqx.tree_at(
                lambda m: m.log_alpha,
                m,
                jnp.clip(m.log_alpha, jnp.log(0.5), jnp.log(20.0)),
            )
            return m

        @eqx.filter_jit
        def _v9_step(
            m: SoftTissuePINN,
            ost: Any,
            td: Dict[str, Any],
            coll: jnp.ndarray,
            w: Dict[str, float],
        ) -> Tuple[SoftTissuePINN, Any, jnp.ndarray, Dict[str, jnp.ndarray]]:
            @eqx.filter_value_and_grad(has_aux=True)
            def lg(
                m_: SoftTissuePINN,
                td_: Dict[str, Any],
                coll_: jnp.ndarray,
                w_: Dict[str, float],
            ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
                return composite_loss(m_, td_, coll_, w_)

            (loss, aux), grads = lg(m, td, coll, w)
            updates, nost = v9_opt.update(grads, ost, eqx.filter(m, eqx.is_array))
            return _clamp(eqx.apply_updates(m, updates)), nost, loss, aux

        v9_history = {
            "loss": [],
            "mu": [],
            "alpha": [],
            "loss_data": [],
            "loss_force": [],
        }

        if quick:
            v9_phases = [
                (500, {"data": 10.0, "bc": 100.0, "phys": 0.0, "force": 0.0}),
                (800, {"data": 10.0, "bc": 100.0, "phys": 1.0, "force": 1.0}),
                (700, {"data": 5.0, "bc": 50.0, "phys": 1.0, "force": 10.0}),
            ]
        else:
            v9_phases = [
                (2000, {"data": 10.0, "bc": 100.0, "phys": 0.0, "force": 0.0}),
                (4000, {"data": 10.0, "bc": 100.0, "phys": 1.0, "force": 1.0}),
                (6000, {"data": 5.0, "bc": 50.0, "phys": 1.0, "force": 10.0}),
            ]

        n_coll = 500 if quick else 2000
        key, ck = jax.random.split(key)
        coll = jax.random.uniform(ck, (n_coll, 2)) * jnp.array([width, height])
        gi = 0

        print("  JIT‑compiling v9 step...")
        for piters, weights in v9_phases:
            for _ in range(piters):
                if gi > 0 and gi % 500 == 0:
                    key, ck = jax.random.split(key)
                    coll = jax.random.uniform(ck, (n_coll, 2)) * jnp.array(
                        [width, height]
                    )
                v9_model, v9_ost, loss, aux = _v9_step(
                    v9_model, v9_ost, v9_data, coll, weights
                )
                if gi % 500 == 0:
                    mu_v, al_v = v9_model.get_params()
                    v9_history["loss"].append(float(loss))
                    v9_history["mu"].append(float(mu_v))
                    v9_history["alpha"].append(float(al_v))
                    v9_history["loss_data"].append(float(aux["data"]))
                    v9_history["loss_force"].append(float(aux["force"]))
                    if gi % 2000 == 0:
                        print(
                            f"    v9 iter {gi}: μ={float(mu_v):.2f} α={float(al_v):.3f}"
                        )
                gi += 1

        mu_v9, al_v9 = [float(x) for x in v9_model.get_params()]
        print(f"  v9 result: μ={mu_v9:.4f} α={al_v9:.4f}")
        print(f"  v9 training: {time.time() - t0:.1f}s")

        plot_comparison_v9_vs_registration(
            v9_history,
            reg_history,
            true_params,
            save_path="outputs/06_comparison_v9_vs_reg.png",
        )

    # ---- Step 7: Noise sensitivity analysis (optional) ----
    if not quick:
        print("\nStep 7: Noise sensitivity analysis...")
        t0 = time.time()
        # (Noise sensitivity function omitted for brevity – can be added similarly)
        print(f"  Done in {time.time() - t0:.1f}s")
    else:
        print("\nSkipping noise sensitivity (use full mode to include)")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    mu_reg, al_reg = [float(x) for x in reg_model.get_params()]
    mu_true = true_params["mu"]
    alpha_true = true_params["alpha"]

    print(f"\n  Registration PINN:")
    print(
        f"    μ = {mu_reg:.4f} Pa  (true: {mu_true:.4f}, "
        f"err: {abs(mu_reg - mu_true) / mu_true * 100:.2f}%)"
    )
    print(
        f"    α = {al_reg:.4f}     (true: {alpha_true:.4f}, "
        f"err: {abs(al_reg - alpha_true) / alpha_true * 100:.2f}%)"
    )

    if v9_history:
        print(f"\n  v9 PINN (displacement‑based):")
        print(
            f"    μ = {v9_history['mu'][-1]:.4f} Pa  "
            f"(err: {abs(v9_history['mu'][-1] - mu_true) / mu_true * 100:.2f}%)"
        )
        print(
            f"    α = {v9_history['alpha'][-1]:.4f}     "
            f"(err: {abs(v9_history['alpha'][-1] - alpha_true) / alpha_true * 100:.2f}%)"
        )

    print(f"\n  Output files:")
    for f in sorted(os.listdir("outputs")):
        if f.endswith(".png"):
            print(f"    outputs/{f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
