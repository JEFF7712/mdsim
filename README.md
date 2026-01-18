# Custom Molecular Dynamics Engine
A high-performance Molecular Dynamics (MD) engine written in C++ and CUDA. It uses a hybrid architecture where bond/angle forces are calculated on the CPU (OpenMP) and non-bonded interactions (Lennard-Jones + Coulomb) are accelerated on the GPU.

https://github.com/user-attachments/assets/e3d36321-2654-4c1b-83cc-ce1c16d7f594
---
## Features:

### Hybrid Parallelization
- **GPU (CUDA):** Handles the computationally expensive non-bonded pair interactions (Lennard-Jones + Coulombic). This is the dominant cost ($O(N \times \text{neighbors})$).
- **CPU (OpenMP):** Handles bonded interactions (harmonic bonds, harmonic angles) and integration.
    * Threads: Dynamic scheduling with `#pragma omp parallel for`.
    * Synchronization: Forces are computed independently on CPU/GPU and summed into a shared force array before the integration step.

### GPU Neighbor Searching (Cell-Linked List):
I used a spatial hashing algorithm on the GPU to avoid the $O(N^2)$ cost of checking every atom pair.

1. **Binning:** The simulation box is divided into cubic cells of side length $d \ge r_{cut} + \text{skin}$. Each atom is hashed to a 1D cell index $k = x_{grid} + y_{grid} \cdot D + z_{grid} \cdot D^2$.
2. **Sorting:** Atoms are sorted by cell ID using CUB Radix Sort.
3. **Indexing:** A lookup table (cell_start, cell_end) is generated to map cell IDs to start/end indices in the sorted atom array.
4. **Adjacency Build:** For every atom, the kernel searches only the 27 adjacent cells (3x3x3 block). Neighbors within $r < r_{cut} + \text{skin}$ are stored in a fixed-width adjacency list per atom.

### Systems
The engine simulates the following molecular systems:
- **Water (H₂O):** Includes proper bonding and angle parameters for water molecules
- **Argon (Ar):** Monatomic noble gas simulations
- **Methane (CH₄):** Multi-atom hydrocarbon simulations with bonding

# Core Equations:
## 1. Equations of Motion (Langevin Dynamics)
The engine uses Langevin Dynamics instead of standard Newtonian physics ($F=ma$). This simulates a system connected to a heat bath (thermostat) by adding friction and random thermal noise. $$ m_i \frac{d\mathbf{v}_i}{dt} = \underbrace{\mathbf{F}_{conservative}(\mathbf{r}_i)}_{\text{Forces from Atoms}} - \underbrace{\gamma \mathbf{v}_i}_{\text{Friction}} + \underbrace{\mathbf{R}_i(t)}_{\text{Random Noise}} $$

## 2. Non-Bonded Interactions (GPU)
Calculated between all pairs of atoms (excluding bonded neighbors) that fall within the cutoff radius ($r < r_{cut}$).
### A. Lennard-Jones Potential
$$V_{LJ}(r_{ij}) = 4\epsilon_{ij} \left[ \left( \frac{\sigma_{ij}}{r_{ij}} \right)^{12} - \left( \frac{\sigma_{ij}}{r_{ij}} \right)^{6} \right]$$
### B. Coulomb Potential (Electrostatics)
$$V_{Coulomb}(r_{ij}) = \frac{1}{4\pi\epsilon_0} \frac{q_i q_j}{r_{ij}}$$

## 3. Bonded Interactions (CPU)
### A. Harmonic Bonds (2-Body)
$$V_{bond}(r) = \frac{1}{2} k_b (r - r_0)^2$$
### B. Harmonic Angles (3-Body)
$$V_{angle}(\theta) = \frac{1}{2} k_\theta (\theta - \theta_0)^2$$

## 4. Integration Scheme
The engine solves these differential equations using a discrete time step ($\Delta t$).
1. **Half-Kick:** $\mathbf{v}(t + \frac{1}{2}\Delta t) = \mathbf{v}(t) + \frac{1}{2} \mathbf{a}(t) \Delta t$
2. **Drift:** $\mathbf{r}(t + \Delta t) = \mathbf{r}(t) + \mathbf{v}(t + \frac{1}{2}\Delta t) \Delta t$
3. **Compute Forces:** $\mathbf{a}(t + \Delta t) = \mathbf{F}(\mathbf{r}(t + \Delta t)) / m$
4. **Half-Kick:** $\mathbf{v}(t + \Delta t) = \mathbf{v}(t + \frac{1}{2}\Delta t) + \frac{1}{2} \mathbf{a}(t + \Delta t) \Delta t$

## Validation
### Radial Distribution Function (RDF)
The accuracy of the simulation was confirmed by computing the Radial Distribution Function (RDF) and comparing against reference data.

## Performance Improvements
### OpenMP Parallelization
**Parameters:** 1000 molecules, 31.0 Å box length, 0.5 fs time step, 50.0 fs simulation time 

| Configuration | Best Time | Improvement |
|---|---|---|
| Single-core (baseline) | 13.94 seconds | - |
| OpenMP parallelization | 2.46 seconds | **82% faster** |

### CUDA GPU Acceleration
**Parameters:** 1000 molecules, 31.0 Å box length, 0.5 fs time step, 5000.0 fs simulation time

| Configuration | Best Time | Improvement |
|---|---|---|
| OpenMP CPU | 245.12 seconds | - |
| CUDA GPU | 70.26 seconds | **71% faster** |

# Resources

Here are some resources I used to learn and help write the code.

- **Used this as a starting point:** [Molecular Dynamics Simulation Introduction](https://youtu.be/ChQbBqndwIA?si=14Vq-WGgh8yWk7DQ)
- **Textbook:** [Understanding Molecular Simulation - Frenkel & Smit](https://www.eng.uc.edu/~beaucag/Classes/AdvancedMaterialsThermodynamics/Books/%5BComputational%20science%20(San%20Diego,%20Calif.)%5D%20Daan%20Frenkel_%20Berend%20Smit%20-%20Understanding%20molecular%20simulation%20_%20from%20algorithms%20to%20applications%20(2002,%20Academic%20Press%20)%20-%20libgen.lc.pdf)
- **C++:** [LearnCpp.com](https://www.learncpp.com/)
- **Good Video on CUDA Basics:** [CUDA Programming Tutorial](https://youtu.be/xwbD6fL5qC8?si=ivaTzz3BVbgJdFsF)

# Future improvements

- Move bond/angle force computing onto GPU for improved performance
- Implement NPT (isothermal-isobaric) ensemble for pressure control
