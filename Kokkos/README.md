# Lattice Boltzmann Method for Magnetic Hydrodynamics Flows

This repository contains the code to simulate magnetic hydrodynamics
flows using Lattice Boltzmann Method (LBM).


## Dependencies

There are two versions of the code. "src/magnetic.cpp" and
"src/mhd.cpp". The "src/magnetic.cpp" is a serial code implemented by
Dr. Alessandro Rosis, University of Manchester. "src/mhd.cpp" is the
port of the "src/magnetic.cpp" code to run on multi-platform using
[Kokkos](https://github.com/kokkos/kokkos/).


0. Setup Kokkos by downloading the latest version from
   https://github.com/kokkos/kokkos/releases

1. Follow the instructions for installation from,
https://kokkos.org/kokkos-core-wiki/get-started/building-from-source.html

2. Clone this repository
```
    $ git clone https://github.com/cfdemons/mhd-lbm
```

3. `cd` into the repository and configure cmake:
```
    $ cmake -B build -S .
```

4. Build the programs:
```
    $ cmake --build build -j
```

5. Run the programs:
```
    $ ./build/MHD && ./build/magnetic_serial
```
