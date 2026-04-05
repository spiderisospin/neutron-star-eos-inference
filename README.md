# Neutron Star Equation of State Inference

This project studies an inverse inference problem: recovering the underlying
equation-of-state parameters of neutron star matter from observable quantities
(mass, radius, tidal deformability).

Our simulation pipeline generates ~260k noisy synthetic observations by sampling
candidate equations of state and solving the Tolman–Oppenheimer–Volkoff (TOV)
equations. Machine learning models are trained to infer physical parameters
from noisy observations and to quantify predictive uncertainty.

Technical stack
- Python, NumPy, SciPy
- PyTorch
- Scientific computing and numerical simulation

Machine learning
- Neural networks for classification and regression
- Bayesian regression for uncertainty estimation
- Feature importance analysis

Computational methods
- Numerical integration of differential equations (TOV solver)
- Synthetic dataset generation
- Large-scale simulation (~260k neutron star models per data set)

## Overview

Neutron stars connect observable astrophysical quantities with the physics of matter at various density regions. Their internal structure is determined by the equation of state (EoS), which relates pressure $p$ and energy density $\varepsilon$. If the EoS is known, global properties such as the stellar mass $M$, radius $R$, and tidal deformability $k_2$ can be obtained by solving the TOV equations.

The project reproduces and extends the deep-learning-based EoS inference framework introduced by Ventagli and Saltas. Parts of the pipeline are inspired by the NS_CC_ML repository (https://github.com/GiuliaVentagli/NS_CC_ML), while the data-generation pipeline and various other components are implemented independently.

## Method

**Data generation**

Candidate equations of state are sampled and used to generate neutron star models by numerically solving the TOV equations. From these solutions, observable quantities $(M,R,k_2)$ are sampled and Gaussian noise injections are added, representing measurement noise.

**Inference models**

Supervised neural networks are trained to infer properties of the neutron star equation of state from observable quantities $(M,R,k_2)$. Separate models identify properties of the low-density region and estimate parameters governing the higher-density regime. Feature-importance analysis is used to evaluate the contribution of each observable. Furthermore, a probabilistic model is implemented to estimate predictive uncertainties.

## Results

The classification model identifies the low-density equation of state with a test accuracy of around **92%**, exceeding the performance of around **87%** reported in the original study.

For the high-density region, (Bayesian) regression results are comparable to the reference work. The regression model recovers large-scale trends in the speed-of-sound profile but smooths out oscillatory structure, indicating limited information content in $(M,R,k_2)$ observations.

This reflects a fundamental limitation of the problem: the mapping from the EoS to observables compresses a large amount of microscopic information into a small set of macroscopic parameters $(M,R,k_2)$, leading to degeneracies in the inverse reconstruction. In astrophysics, this phenomenon is known as the inverse stellar structure problem.

## Repository structure

**data_generated/**:
    Folder containing generated synthetic neutron star datasets using generate_data.ipynb.

**data_reference/**:
    Folder containing reference EOS data required for the data-generation pipeline.

**src/**:
    Folder containing model implementations, plotting utilities and helper functions.

**generate_data.ipynb**:
    Pipeline for generating synthetic neutron star observations.

**eos_inference_pipeline.ipynb**:
    Machine learning pipeline for equation-of-state inference.
