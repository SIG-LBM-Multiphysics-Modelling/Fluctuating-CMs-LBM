# Kokkos-Based Fluctuating Lattice Boltzmann Framework

A high-performance fluctuating lattice Boltzmann framework implemented in modern C++ using the [Kokkos](https://kokkos.org/) programming model for portable on-node parallelism.

The codebase includes:

* **2D D2Q9 implementations**

  * BGK collision operator
  * Regularized collision operator
  * Central moments (CMS) collision operator
* **3D D3Q27 implementations**

  * BGK collision operator
  * Central moments (CMS) collision operator
* Support for **thermal fluctuations** and stochastic forcing
* Export utilities for **VTK/VTI visualisation**
* Modular structure suitable for benchmarking, validation, and future extensions

The project is designed for research in:

* Fluctuating hydrodynamics
* Soft matter and mesoscale physics
* Plasma-like and multiphase systems
* High-performance scientific computing
* GPU/CPU portable lattice Boltzmann simulations

---

# Repository Structure

```text
Kokkos/
├── CMakeLists.txt
├── include/
│   ├── VTKWriter.h
│   ├── VTKWriter2D.h
│   └── VTIWriter2D.h
├── src/
│   ├── 2D/
│   │   ├── D2Q9_BGK_Fluct_Tests.cpp
│   │   ├── D2Q9_Reg_Fluct_Tests.cpp
│   │   ├── D2Q9_CMS_Fluct_Tests.cpp
│   │   └── Old/
│   └── 3D/
│       └── Q27/
│           ├── D3Q27_BGK_Fluct_Tests.cpp
│           └── D3Q27_CMS_Fluct_Tests.cpp
└── build/
```

---

# Features

## Portable Parallelism with Kokkos

The framework uses Kokkos to enable execution across:

* Serial CPUs
* OpenMP multi-core CPUs
* CUDA-enabled GPUs
* HIP-enabled AMD GPUs
* Other supported Kokkos backends

This allows a single codebase to target multiple hardware architectures with minimal changes.

---

## Collision Models

### BGK (Bhatnagar–Gross–Krook)

Single-relaxation-time formulation commonly used in lattice Boltzmann methods due to its simplicity and computational efficiency.

### Central Moments Scheme (CMS)

Improves numerical stability and Galilean invariance by performing relaxation in central-moment space.

---

## Fluctuating Hydrodynamics

The framework includes stochastic forcing terms consistent with fluctuating hydrodynamics formulations, enabling:

* Thermal noise modelling
* Equilibrium fluctuation studies
* Mesoscopic statistical mechanics investigations
* Validation against fluctuation–dissipation behaviour

---

# Dependencies

## Required

* C++17 compatible compiler
* CMake ≥ 3.16
* [Kokkos](https://github.com/kokkos/kokkos)

## Recommended Compilers

* GCC
* Clang
* Intel oneAPI
* NVIDIA HPC SDK

---

# Installation

## 1. Install Kokkos

Download and install the latest version of Kokkos:

```bash
git clone https://github.com/kokkos/kokkos.git
cd kokkos
```

Follow the official installation instructions:

[https://kokkos.org/kokkos-core-wiki/get-started/building-from-source.html](https://kokkos.org/kokkos-core-wiki/get-started/building-from-source.html)

---

## 2. Clone the Repository

```bash
git clone https://github.com/SIG-LBM-Multiphysics-Modelling/Fluctuating-CMs-LBM.git
cd Fluctuating-CMs-LBM/Kokkos
```

---

## 3. Configure the Build

### CPU Build

```bash
cmake -B build -S .
```

### OpenMP Build

```bash
cmake -B build -S . \
  -DKokkos_ENABLE_OPENMP=ON
```

### Threads Build

```bash
cmake -B build -S . \
  -DKokkos_ENABLE_THREADS=ON
```

### CUDA Build

```bash
cmake -B build -S . \
  -DKokkos_ENABLE_CUDA=ON
```

---

## 4. Compile

```bash
cmake --build build -j
```

---

# Running Simulations

## 2D D2Q9 CMS Test

```bash
./build/D2Q9_CMS_Fluct_Tests -case 1
(use -case 2, -case 3, -case 4, -case 5, -case 6, -case 7 for the remaining tests)
```

## 2D D2Q9 BGK Test

```bash
./build/D2Q9_BGK_Fluct_Tests -case 1
```

## 3D D3Q27 CMS Test

```bash
./build/D3Q27_CMS_Fluct_Tests -case 1
```

## 3D D3Q27 BGK Test

```bash
./build/D3Q27_BGK_Fluct_Tests -case 1
```

---

# Output and Visualisation

The framework includes utilities for writing:

* `.vtk`
* `.vti`

files for post-processing and visualisation in:

* ParaView
* VisIt
* MayaVi

Example workflow:

```text
Simulation → VTK/VTI output → ParaView visualisation
```

---


# Build Cleanup

To remove the build directory:

```bash
rm -rf build
```

---

# Citation

If you use this framework in academic work, please cite the relevant associated publications and acknowledge the use of Kokkos.

---

# Acknowledgements

This project uses:

* [Kokkos](https://kokkos.org/) for performance portability
* VTK-compatible formats for scientific visualisation

---

# License

Add your preferred open-source license here:

* MIT
* BSD-3-Clause
* GPL
* Apache-2.0

Example:

```text
Copyright (c) 2026
```
