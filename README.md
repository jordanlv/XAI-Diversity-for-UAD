# Explainability-Guided Diversity for Unsupervised Anomaly Detection Ensembles

<!-- [![HAL](https://img.shields.io/badge/HAL-Paper-blue.svg)](https://hal.science/hal-05485124) -->
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

Official implementation of the paper: **"Explainability-Guided Diversity for Unsupervised Anomaly Detection Ensembles"**.

> **Authors:** Jordan Levy, Paul Saves, Moncef Garouani, Nicolas Verstaevel, Benoit Gaudou & Vincent Talon <br>
> **Institution:** IRIT, Université Toulouse Capitole & TwinswHeel, Soben

## Abstract

Diversity in ensemble of Unsupervised Anomaly Detection (UAD) remains challenging due
to the absence of ground-truth labels. Traditionally, diversity in UAD is assessed through
output representations, such as outlier scores or reconstruction errors which fail to capture
differences in the underlying decision-making behavior of the base models. We propose a
novel approach to quantifying the diversity of the UAD ensemble by leveraging explainability
metrics. Rather than comparing output scores, we analyze feature attributions to charac-
terize how individual models reach their decisions. Across a comprehensive benchmark of
16 datasets and 14 diverse UAD models, we evaluated 8 diversity metrics derived from local
and global explainers to quantify model dissimilarity. Our empirical results demonstrate
that feature attribution-based explainability can effectively assess and encourage diversity
leading to superior ensemble performance. Specifically, attributions derived via SHAP and
ALE yield the most robust diversity measures. Finally, we demonstrate that, while diver-
sity can be beneficial, individual predictive performance of base models remains a strict
prerequisite for ensemble success.

## Datasets
You can access and download the datasets used in this paper directly from the GitHub repository **[ADBench](https://github.com/Minqi824/ADBench)**.