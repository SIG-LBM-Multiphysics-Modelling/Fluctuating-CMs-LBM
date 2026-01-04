#ifndef VTKWRITER_H
#define VTKWRITER_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <Kokkos_Core.hpp>

class VTKWriter {
public:
    // 2D MHD write function
    template <typename ViewType>
    static void write(const std::string& filename, const int nx, const int ny,
        ViewType& u, ViewType& v, 
        ViewType& rho, const double U_ref)
    {
        std::ofstream output_file(filename);
        if (!output_file.is_open()) return;

        writeHeader(output_file, nx, ny);
        writeCoordinates(output_file, nx, ny);
        writePointData(output_file, nx, ny, u, v, rho, U_ref);

        output_file.close();
    }

private:
    template<typename U>
    static const char* getVTKType(); // Declaration only

    static void writeHeader(std::ofstream& output_file, int nx, int ny) {
        output_file << "# vtk DataFile Version 3.0\n";
        output_file << "VTK output\n";
        output_file << "ASCII\n";
        output_file << "DATASET RECTILINEAR_GRID\n";
        output_file << "DIMENSIONS " << nx << " " << ny << " " << 1 << "\n";
    }

    static void writeCoordinates(std::ofstream& output_file, int nx, int ny) {
        output_file << "X_COORDINATES " << nx << " float\n";
        for (int i = 0; i < nx; ++i) output_file << i << ' ';
        output_file << "\nY_COORDINATES " << ny << " float\n";
        for (int j = 0; j < ny; ++j) output_file << j << ' ';
        output_file << "\nZ_COORDINATES " << 1 << " float\n";
        for (int k = 0; k < 1; ++k) output_file << k << ' ';
        output_file << '\n';
    }

    template <typename ViewType>
    static void writePointData(std::ofstream& output_file, const int nx, const int ny, 
        ViewType& u, ViewType& v,
        ViewType& rho, const double U_ref) {

        const int nz = 1;
        int total_points = nx * ny * nz;
        output_file << "POINT_DATA " << total_points << '\n';

        output_file << "VECTORS Velocity " << getVTKType<typename ViewType::value_type>() << '\n';
        for (int z = 0; z < nz; ++z) 
            for (int j = 0; j < ny; ++j) 
                for (int i = 0; i < nx; ++i) 
                output_file << u(i, j)/U_ref << ' ' << v(i, j)/U_ref << " 0\n";                
    }
};

// Specializations of getVTKType
template<>
const char* VTKWriter::getVTKType<float>() { return "float"; }
template<>
const char* VTKWriter::getVTKType<double>() { return "double"; }
template<>
const char* VTKWriter::getVTKType<int>() { return "int"; }

#endif // VTKWRITER_H