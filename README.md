# Hyperspectral Spectral Decomposition via Plug-and-Play Regularization

> A research-oriented notebook project on **hyperspectral image unmixing** using  
> **constrained optimization**, **synthetic data generation**, and **Plug-and-Play / GS-PnP priors**.  
> The objective is to recover physically meaningful **abundance maps** and, depending on the setting, **endmember spectra**, while preserving the interpretability of the inverse-problem formulation.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Mathematical Background](#3-mathematical-background)
   - 3.1 [Linear Mixing Model](#31-linear-mixing-model)
   - 3.2 [Physical Constraints](#32-physical-constraints)
   - 3.3 [Inverse Problem Formulation](#33-inverse-problem-formulation)
   - 3.4 [Plug-and-Play Regularization](#34-plug-and-play-regularization)
   - 3.5 [Gradient-Step Plug-and-Play (GS-PnP)](#35-gradient-step-plug-and-play-gs-pnp)
   - 3.6 [Synthetic Data Generation](#36-synthetic-data-generation)
4. [Notebook Walkthrough](#4-notebook-walkthrough)
   - 4.1 [`Hyperspectral_PnP.ipynb`](#41-project_finalipynb)
   - 4.2 [`images/`](#42-images)
5. [Installation & Usage](#5-installation--usage)
6. [Evaluation](#6-evaluation)
7. [References](#7-references)

---

## 1. Project Overview

Hyperspectral images (HSI) differ fundamentally from RGB images. Instead of 3 broad spectral bands, HSIs contain **tens to hundreds of narrow spectral bands**, often extending beyond the visible range. This richer spectral resolution makes hyperspectral imaging especially valuable for:

- remote sensing,
- material identification,
- environmental monitoring,
- agriculture,
- biomedical and scientific imaging.

However, because spatial resolution is limited, a single hyperspectral pixel often contains a **mixture of several materials**. The corresponding inverse problem is known as **hyperspectral unmixing**: given an observed spectrum, we aim to decompose it into:

- a set of **endmembers** (pure material spectra),
- and their **abundances** (proportions of each material in each pixel).

This repository studies unmixing through a **model-based optimization lens**. Instead of relying only on end-to-end neural networks, it explores how **Plug-and-Play priors** and **GS-PnP interpretations** can inject learned regularity into the reconstruction while preserving the physical structure of the problem.

---

## 2. Repository Structure

```text
HyperspectralImagingUnmixing/
│
├── project_final.ipynb      # Main notebook: derivations, experiments, implementation
├── images/                  # Figures, plots, and qualitative visual results
└── README.md                # Project description and mathematical overview
```

The repository is intentionally lightweight and notebook-centered. The main ideas, derivations, and experiments are concentrated in `project_final.ipynb`, while `images/` stores visual material used for analysis or illustration.

---

## 3. Mathematical Background

### 3.1 Linear Mixing Model

We model the observed hyperspectral image as:

$$X = AS + N$$

where:

- $X \in \mathbb{R}^{L \times P}$ is the observed hyperspectral image,
- $A \in \mathbb{R}^{L \times K}$ is the endmember matrix,
- $S \in \mathbb{R}^{K \times P}$ is the abundance matrix,
- $N$ is additive noise,
- $L$ is the number of spectral bands,
- $P$ is the number of pixels,
- $K$ is the number of materials.

Each pixel is therefore represented as a linear combination of $K$ endmember spectra.

---

### 3.2 Physical Constraints

The abundance vectors represent proportions, so they must satisfy standard physical constraints.

**Non-negativity:**

$$S \ge 0$$

**Sum-to-one constraint:**

$$\sum_{k=1}^{K} S_{k,p} = 1 \qquad \text{for every pixel } p$$

Equivalently, each abundance vector belongs to the simplex:

$$
\Delta^K = \{\, s \in \mathbb{R}^K \mid s \ge 0,\ \sum_{k=1}^{K} s_k = 1 \,\}
$$

These constraints are essential for interpretability and physical consistency.

---

### 3.3 Inverse Problem Formulation

In the simplest setting, hyperspectral unmixing can be formulated as:

$$\min_{A,S} \frac{1}{2}\|X - AS\|_F^2 + \lambda R(A,S)$$

where:

- the first term is the **data fidelity term**,
- $R(A,S)$ is a regularization term,
- $\lambda > 0$ balances reconstruction accuracy and prior knowledge.

Depending on the scenario, the project may consider:

| Setting | Unknowns | Description |
|---|---|---|
| **Non-blind** | $S$ only | Endmembers fixed, abundances estimated |
| **Semi-blind** | mostly $S$ | Endmembers known or partially fixed |
| **Blind** | $A$ and $S$ | Joint estimation of spectra and abundances |

The main difficulty is that this problem is generally:

- ill-posed,
- non-convex,
- not uniquely identifiable,
- sensitive to noise and model mismatch.

Without regularization, many decompositions can explain the same data.

---

### 3.4 Plug-and-Play Regularization

Classical regularizers such as sparsity, smoothness, or Total Variation are useful, but often too rigid to model realistic abundance maps.

Plug-and-Play (PnP) methods replace an explicit regularizer by a **denoising operator** $D_\sigma$. Instead of manually designing $R$, one alternates:

1. a data-consistency update,
2. a denoising step acting as an implicit prior.

This leads to the interpretation:

$$\min_Z f(Z) + \lambda R(Z)$$

where the prior is not written explicitly, but enforced through a learned denoiser.

In the hyperspectral setting, this is especially attractive because abundance maps often exhibit strong spatial structure that simple pixelwise penalties fail to capture.

---

### 3.5 Gradient-Step Plug-and-Play (GS-PnP)

A more structured interpretation of Plug-and-Play is given by **Gradient-Step Plug-and-Play**, where the denoiser can be written as:

$$D_\sigma(x) = x - \nabla g_\sigma(x)$$

for some implicit potential $g_\sigma$.

This yields an optimization-inspired decomposition:

$$F(Z) = f(Z) + \lambda g_\sigma(Z)$$

where:

- $f(Z)$ is the data fidelity term,
- $g_\sigma(Z)$ is induced by the denoiser.

This is important because it connects learned denoisers with variational optimization, making the method more theoretically grounded than a purely heuristic denoise-after-gradient strategy.

In this project, GS-PnP is particularly relevant when regularizing abundance maps while preserving simplex-type constraints and inverse-problem structure.

---

### 3.6 Synthetic Data Generation

Large annotated hyperspectral datasets are scarce, so synthetic data generation plays a central role in the project.

Several strategies are relevant:

#### 1. Statistical abundance modeling

- Dirichlet-distributed abundances,
- simplex-constrained random generation,
- controllable sparsity or mixing levels.

#### 2. Spatial structure simulation

- dead leaves model,
- blobs, rectangles, piecewise-constant regions,
- structured abundance maps with realistic geometry.

#### 3. Spectral library usage

- use known spectral signatures from spectral libraries,
- combine real endmembers with synthetic abundances.

#### 4. Noise and variability modeling

- additive Gaussian noise,
- spectral perturbations,
- controlled variability in endmembers.

Synthetic generation is useful because it provides access to the **ground truth**, which enables clearer benchmarking of abundance recovery and reconstruction quality.

---

## 4. Notebook Walkthrough

### 4.1 `Hyperspectral_PnP.ipynb`

This notebook is the main entry point of the repository. It is expected to contain the full experimental pipeline:

1. mathematical background on hyperspectral unmixing,
2. generation or loading of hyperspectral data,
3. implementation of constrained optimization baselines,
4. integration of Plug-and-Play or GS-PnP priors,
5. visualisation of spectra and abundance maps,
6. qualitative and quantitative analysis.

Conceptually, the notebook studies how learned denoising priors can improve abundance recovery while keeping the reconstruction pipeline interpretable and physically constrained.

---

### 4.2 `images/`

The `images/` folder stores figures used for visual analysis, including examples such as:

- abundance maps,
- spectral plots,
- comparison figures,
- qualitative reconstruction outputs.

This folder complements the notebook by providing static visual material for interpretation and reporting.

---

## 5. Installation & Usage

### Clone the repository

```bash
git clone https://github.com/KHOUTAIBI/HyperspectralImagingUnmixing.git
cd HyperspectralImagingUnmixing
```

### Install common dependencies

The exact dependencies depend on the notebook implementation, but a standard setup would typically include:

```bash
pip install numpy scipy matplotlib jupyter torch torchvision
```

Additional packages may be required depending on the denoiser architecture, data-loading utilities, or optimization routines used in the notebook.

### Launch the notebook

```bash
jupyter notebook Hyperspectral_PnP.ipynb
```

Then execute the cells step by step to reproduce the experiments and visualizations.

---

## 6. Evaluation

The project can be evaluated through both **quantitative** and **qualitative** criteria.

Typical quantitative metrics include:

- **MSE** on abundance maps,
- **PSNR** for reconstruction quality,
- **SAM** (Spectral Angle Mapper),
- reconstruction error $\|X - AS\|_F$.

A common reconstruction metric is:

$$\text{PSNR}(x,\hat{x}) = 10 \log_{10}\left(\frac{\mathrm{MAX}^2}{\mathrm{MSE}(x,\hat{x})}\right)$$

Qualitative analysis is also important, especially for:

- spatial smoothness of abundance maps,
- preservation of material boundaries,
- realism of reconstructed spectral patterns.

The main point of comparison is whether Plug-and-Play regularization produces abundance maps that are cleaner, more spatially coherent, and more physically meaningful than classical baselines.

---

## 7. References

| Topic | Reference |
|---|---|
| Hyperspectral unmixing | Classical linear mixture models and constrained matrix factorization |
| Plug-and-Play priors | Optimization with denoisers as implicit regularizers |
| GS-PnP | Gradient-step interpretation of denoising-based priors |
| Spectral unmixing constraints | Non-negativity and simplex-based formulations |
| Synthetic data generation | Dirichlet priors, dead-leaves-style structures, spectral libraries |

This repository sits at the intersection of:

- inverse problems,
- hyperspectral imaging,
- constrained optimization,
- learned priors,
- and denoising-based regularization.
