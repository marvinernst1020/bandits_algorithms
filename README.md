# Bandits for Algorithm Selection

## Project Overview

- **Title**: Bandits for Algorithm Selection  
- **Supervisors**: Prof. Christian Brownlees, Prof. David Rossell  
- **Institution**: Universitat Pompeu Fabra & Barcelona School of Economics  
- **Objective**: To develop and evaluate strategies for selecting forecasting algorithms that maximize the expected cumulative reward in a multi-armed bandit setting.  
- **Team Members**: Marvin Ernst, Oriol Gelabert, Melisa Vadenja

---

## Motivation

Web platforms and applications often face the challenge of selecting the best algorithm (or strategy) to present items (e.g., ads, recommendations) to users in order to maximize engagement. Using bandit algorithms allows dynamic and efficient allocation of forecasting models based on user interactions and rewards.

---

## Problem Formulation

- **Batch Setting**: Each day $t$, a batch of $n$ users arrives. Each is assigned to one of $K$ forecasting algorithms.
- **Reward Signal**: A reward $r_{ti}$ is observed when a user clicks (or engages), with a fixed reward value $c$ for each click.
- **Decision Vector**: $d_t \in \{1, ..., K\}^n$ assigns algorithms to users.
- **Goal**: Select a *policy* $p$ that determines $d_t$ each day to maximize  
  $$\mathbb{E} \left( \sum_t \sum_{i=1}^n r_{ti}^{(p)} \right)$$
  

---

## Key Components

### Policies Evaluated
- **Upper Confidence Bound (UCB)**
- **Thompson Sampling**
- **Gaussian Process Bandits**

### Evaluation Metric
- **Cumulative Reward** (or equivalently, **Regret Minimization**)

---

## Literature
We base our approach on foundational work in bandits and Gaussian Processes:
- Desautels et al. (2014)
- Chowdhury & Gopalan (2017)
- Salgia et al. (2021)
- Kandasamy et al. (2018)

---
Later we will add the structure of our project etc. here.
