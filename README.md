# Bandits for Algorithm Selection

## Project Overview

- **Title**: *Bayesian Bandits for Algorithm Selection: Latent-State Modeling and Spatial Reward Structures*
- **Supervisors**: Prof. Christian Brownlees, Prof. David Rossell  
- **Institution**: Universitat Pompeu Fabra & Barcelona School of Economics  
- **Objective**: To develop and evaluate bandit algorithms capable of dynamically selecting forecasting models under non-stationary and spatially-structured reward distributions.  
- **Team Members**: Marvin Ernst, Oriol Gelabert, Melisa Vadenja

---

## Motivation

Many real-world systems - from online platforms to adaptive control - require sequential decisions in uncertain environments where rewards evolve over time or depend on spatial features. This project explores how bandit algorithms can be enhanced to adapt to such non-stationarity, via hidden latent-state dynamics and spatial correlation among arms.

---

## Problem Formulation

- **Setting**: Each day (or time step) $t$, a decision-maker selects one arm (algorithm) out of $K$ options, receiving a stochastic reward.
- **Objective**: Minimize cumulative regret over a finite horizon $T$, compared to the best dynamic or spatially optimal policy.
- **Extensions**:
  - **Dynamic Bandits**: Rewards evolve due to unobserved latent states.
  - **Spatial Bandits**: Arms are embedded in a continuous space and share smooth reward structure.

---

## Contributions

1. **Dynamic Bandits**:
   - Implemented two Bayesian latent-state models:
     - **M1**: Arm-specific Hidden Markov Models (HMMs)
     - **M2**: Globally shared latent HMM
   - Compared to autoregressive (AR) and classical baselines.
   - Showed that **M1-TS** consistently outperforms others across settings, especially in adapting to hidden regime switches.

2. **Spatial Bandits**:
   - Benchmarked **GP-based UCB/TS** against:
     - **Zoom-In**: A tree-based region refinement strategy for Lipschitz bandits.
     - **Classical UCB** as a baseline.
   - Demonstrated the impact of kernel assumptions (noise, length scale) on GP performance.
   - Analyzed robustness of Zoom-In in irregular landscapes and its sensitivity to hyperparameters.

3. **Hybrid Strategies** (Future Direction):
   - Proposed combining GP-based exploration with Zoom-In-style regional refinement to balance accuracy and scalability.

---

## Evaluation Metrics

- **Cumulative Regret**
- **Instantaneous Regret**
- **Distance to the Best Arm** (for spatial scenarios)

---

## Key Algorithms Implemented

- **Bayesian Inference in JAGS** (for M1/M2)
- **Autoregressive Bandits**
- **GP-UCB, GP-TS**
- **Zoom-In Algorithm with multiple scoring rules (UCB, tuned-UCB, TS)**
- **Baseline UCB and Thompson Sampling**
---

## Literature

Our methods build on key contributions in bandit theory, Bayesian inference, and spatial modeling:

- **Desautels et al. (2014)** – *Gaussian Process Bandits with Exploration-Exploitation*  
  Introduced a framework for safe exploration using GPs under constraints, demonstrating GP-UCB effectiveness in structured spaces.

- **Chowdhury & Gopalan (2017)** – *Bandit Optimization with Gaussian Processes: Theory and Algorithms*  
  Provided theoretical regret bounds and adaptive confidence bounds for GP-UCB, enhancing its robustness under unknown noise conditions.

- **Salgia et al. (2021)** – *Lipschitz Bandits without the Lipschitz Constant*  
  Extended the Zooming algorithm to settings without prior knowledge of smoothness constants, leading to more adaptive exploration in spatial bandits.

- **Kandasamy et al. (2018)** – *Parallelised Bayesian Optimisation via Thompson Sampling*  
  Discussed Bayesian optimization with Gaussian Processes, including Thompson Sampling in high-dimensional input spaces.

- **Kleinberg et al. (2008)** – *Regret Bounds for Restless Bandits*  
  Introduced the theoretical foundations for the Zooming algorithm and regret minimization under Lipschitz continuity assumptions.

- **Lattimore & Szepesvári (2020)** – *Bandit Algorithms*  
  Comprehensive textbook that lays the theoretical groundwork for classical and modern bandit methods, including UCB, Thompson Sampling, and structured bandits.

- **Hamilton (1994)** – *Time Series Analysis*  
  Fundamental resource for understanding autoregressive models and hidden Markov models, which we adapted for non-stationary bandits in our dynamic modeling framework.

---

## Final Remarks

This project advances the understanding of structured bandit problems by providing detailed empirical evidence and implementation insights. We highlight the strengths of Bayesian models in non-stationary settings and reveal the computational trade-offs in spatial domains. Our findings advocate for adaptive, problem-driven strategy design, and motivate further research in hybrid and scalable bandit architectures.


