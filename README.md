# Quantum Chaos Simulation: OTOC and Local Instability in the Volcano Potential

This project explores quantum information scrambling and local instability by simulating the Out-of-Time-Ordered Correlator (OTOC) for a one-parameter family of potentials known as the "Volcano Potential." By varying the asymptotic structure of the potential, this analysis investigates how local geometric instability (negative curvature) drives the short-time exponential growth of the OTOC, acting as a quantum analogue to classical chaos.

The simulation and subsequent analyses are implemented in a Jupyter Notebook using standard Python scientific libraries, utilizing a Discrete Variable Representation (DVR) framework for matrix diagonalization and time evolution.

## Physics Background

### The Volcano Potential
The core system is defined by the one-parameter family of potentials:
$$V(x) = -\left(a_1 \cosh^{2\nu} x + a_2 \text{sech}^2 x \right)$$

The competition between the smooth attractive well ($\text{sech}^2 x$) and the asymptotic modifier ($\cosh^{2\nu} x$) yields distinct qualitative structures depending on $\nu$:
* **$\nu < 0$ (Modified Well):** The potential maintains a bounded well, decaying at infinity.
* **$\nu = 0$ (Shifted Pöschl-Teller):** The system reduces to the exactly solvable Pöschl-Teller potential.
* **$\nu > 0$ (Confining Walls):** The potential exhibits steep, exponentially growing confining walls (requiring $a_1 < 0$ for Hermiticity).

### The Out-of-Time-Ordered Correlator (OTOC)
The OTOC quantifies the sensitivity of quantum dynamics to small perturbations. We compute the expectation value of the squared commutator of the position and momentum operators in the energy eigenbasis:
$$C_n(t) = \langle n| [\hat{x}(t), \hat{p}(0)]^\dagger [\hat{x}(t), \hat{p}(0)]|n \rangle$$
The early-time exponential growth of $C_n(t)$ is a signature of local instability and information scrambling.

### Local Instability and Curvature
The short-time growth of the OTOC is tied to the local geometry of the potential, specifically the regions of negative curvature where $V''(x) < 0$. These regions act as locally inverted harmonic oscillators. The simulation computes the classical Lyapunov exponent ($\lambda_c$) derived from this curvature, alongside an Airy-corrected semiclassical exponent ($\lambda_{sc}$) that accounts for the wavefunction maximum shifting slightly inside the classical turning point.

## Features

* **DVR Framework Optimization:** Discretizes the spatial domain to construct and diagonalize the Hamiltonian $\hat{H} = \hat{T} + \hat{V}$, solving for exact energy eigenvalues and eigenvectors across all $\nu$ configurations.
* **Time Evolution & Scrambling:** Computes both the short-time exponential growth and the long-time saturation regimes of the OTOC for distinct energy states.
* **Curvature Diagnostics:** Automatically maps the second derivative $V''(x)$ to highlight and visualize unstable geometric regions.
* **Lyapunov Exponent Extraction:** Calculates and compares classical and Airy-corrected semi-classical Lyapunov exponents for varying potential asymptotes.
* **Publication-Ready Visualization:** Generates standardized, multi-panel figures for potentials, OTOC growth, stability regions, and exponent comparisons, styled for academic integration.

## Requirements

* Python 3.x
* Jupyter Notebook or JupyterLab
* Core Libraries: `numpy`, `scipy`, `matplotlib`, `pandas`

Install dependencies using pip:
```bash
pip install numpy scipy matplotlib pandas jupyter
