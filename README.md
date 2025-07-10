# Equlibrium-preserving Laplacian Renormalization Group of hypergraphs

This repository implements a Equlibrium-preserving Laplacian Renormalization Group (EqLRG) method for coarse-graining complex hypergraphs, including pairwise graphs. The main goal is to iteratively reduce network size while preserving key spectral and equilibrium properties.

---

## 🔧 File Overview


### 'lib.py'
Contains core utilities for:
- Laplacian Renormalization Group method and its functional processes
- Entropy and Partition function calculations


### 'net.py'
Parses a network specification string [NETWORK_SPEC] and generates a corresponding network or hypergraph.

This function interprets a string-based format to construct various types of networks, including:
- Pairwise networks (graphs)
- Hypergraphs
- Bipartite-lifted hypergraphs derived from pairwise networks

Each specification includes the model name, parameters (e.g., average degree or exponent), 
network size, and random seed, in a compact standardized format.

-------------------------------------------------------------------------------
Model categories and specification formats:
-------------------------------------------------------------------------------

Pairwise networks (Graph):
    - "BA_m{m}_N{N}I{seed}":
        Barabási–Albert scale-free graph with m edges per new node.
    - "RN_N{N}I{seed}":
        Random tree generated via Prufer sequence.
    - "ER_k{k}_N{N}I{seed}":
        Erdős–Rényi graph with average degree k.
    - "ST_k{k}_g{g}_N{N}I{seed}":
        Static model with P(k) ~ k^(-g), average degree k.

Hypergraphs (Explicit construction):
    - "HBA_m{m}_N{N}I{seed}":
        Hypergraph version of BA model.
    - "HRN_N{N}I{seed}":
        Hypergraph generated from a hyper-Prufer sequence.
    - "HER_k{k}_N{N}I{seed}":
        Hypergraph analog of ER graph.
    - "HST_k{k}_g{g}_N{N}I{seed}":
        Static hypergraph with degree P(k) ~ k^(-g), cardinality Poissonian.
    - "HST_k{k}_g{g}_g{g2}_N{N}I{seed}":
        Static hypergraph with P(k) ~ k^(-g), P(c) ~ c^(-g2)

Hypergraphs (Bipartite lifting of pairwise networks):
    - "BABI_m{m}_N{N}I{seed}":
        BA network interpreted as a hypergraph via bipartite mapping.
    - "RNBI_N{N}I{seed}":
        Prufer tree lifted to hypergraph.
    - "ERBI_k{k}_N{N}I{seed}":
        ER graph lifted to hypergraph.
    - "STBI_k{k}_g{g}_N{N}I{seed}":
        Static graph lifted to hypergraph.


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
