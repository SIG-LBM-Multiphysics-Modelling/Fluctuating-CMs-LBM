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
    $ ./build/D3Q27_CMS_Fluct_Tests -case 1
```
