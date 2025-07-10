# Equlibrium-preserving Laplacian Renormalization Group of hypergraphs

This repository implements a Equlibrium-preserving Laplacian Renormalization Group (EqLRG) method for coarse-graining complex hypergraphs, including pairwise graphs. The main goal is to iteratively reduce network size while preserving key spectral and equilibrium properties.

---

## 🔧 File Overview

### 'lib.py'
Contains core utilities for:
- Laplacian Renormalization Group method and its functional processes
- Entropy and Partition function calculations

### 'net.py'
Generates a variety of graph topologies from string specs:
- 'BA', 'RN', ER', 'ST' (Barabasi-Albert, random tree from Prefur sequence, Erdős–Rényi, static model)
- 'BABI', 'RNBI', 'STBI', 'ERBI' (bipartite versions of above graphs)
- 'HBA', 'HRN', 'HER', 'HST' (hypergraphs versions)
- Uses deterministic seed for reproducibility

### 'rg.py'
Main driver script for running EqLRG:
- Reads network spec and parameters
- Performs repeated making supernodes and renormalization
- Saves intermediate results including:
  - Reduced size and mean degrees over EqLRG steps.
  - Degree and cardinality for each step
  - Partition function and entropy for each step

---

## ▶️ Usage

'''bash
python rg.py [NETWORK_SPEC] [f] [kappa]
