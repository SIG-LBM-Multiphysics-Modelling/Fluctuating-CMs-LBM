#ifndef VTIWRITER_2D_H
#define VTIWRITER_2D_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint> // for uint32_t
#include <Kokkos_Core.hpp>

class VTIWriter {
public:
    // 3D write function for VTI format with appended raw binary data
    template <typename ViewType>
    static void writeVTI(const std::string& filename,
                         const int nx, const int ny,
                         ViewType& u, ViewType& v,
                         ViewType& rho, ViewType& Ex, ViewType& Ey,
                         ViewType& q, ViewType& phi,
                         const double q0, const double phi_b, const double U_ref)
    {
        size_t nPoints = static_cast<size_t>(nx) * ny * 1;

        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Cannot open file: " << filename << "\n";
            return;
        }

        // Helper lambda to compute appended data block size (UInt32 header + float data)
        auto arraySize = [](size_t n) {
            return sizeof(uint32_t) + sizeof(float) * n;
        };

        // Sizes of all data arrays for offset calculation
        size_t chargeSize     = arraySize(nPoints);
        size_t potentialSize  = arraySize(nPoints);
        size_t velocitySize   = arraySize(3 * nPoints);
        size_t efieldSize     = arraySize(3 * nPoints);
        size_t densitySize    = arraySize(nPoints);

        // Offsets for each data array in appended data block
        size_t offset_charge    = 0;
        size_t offset_potential = offset_charge + chargeSize;
        size_t offset_velocity  = offset_potential + potentialSize;
        size_t offset_efield    = offset_velocity + velocitySize;
        size_t offset_density   = offset_efield + efieldSize;

        // Write XML header for VTI file
        file <<
            "<?xml version=\"1.0\"?>\n"
            "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt32\">\n"
            "  <ImageData WholeExtent=\"0 " << (nx-1) << " 0 " << (ny-1) << " 0 0" << "\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n"
            "    <Piece Extent=\"0 " << (nx-1) << " 0 " << (ny-1) << " 0 0" << "\">\n"
            "      <PointData Scalars=\"ChargeDensity\">\n"
            "        <DataArray type=\"Float32\" Name=\"ChargeDensity\" format=\"appended\" offset=\"" << offset_charge << "\"/>\n"
            "        <DataArray type=\"Float32\" Name=\"Potential\" format=\"appended\" offset=\"" << offset_potential << "\"/>\n"
            "        <DataArray type=\"Float32\" Name=\"Velocity\" NumberOfComponents=\"2\" format=\"appended\" offset=\"" << offset_velocity << "\"/>\n"
            "        <DataArray type=\"Float32\" Name=\"ElectricField\" NumberOfComponents=\"2\" format=\"appended\" offset=\"" << offset_efield << "\"/>\n"
            "        <DataArray type=\"Float32\" Name=\"Density\" format=\"appended\" offset=\"" << offset_density << "\"/>\n"
            "      </PointData>\n"
            "      <CellData/>\n"
            "    </Piece>\n"
            "  </ImageData>\n"
            "  <AppendedData encoding=\"raw\">\n"
            "    _";  // underscore marks start of appended data

        // Helper lambda to flatten 3D view into 1D std::vector<float>
        auto flatten = [&](ViewType& arr) {
            std::vector<float> flat(nPoints);
            for (int k = 0; k < 1; ++k)
                for (int j = 0; j < ny; ++j)
                    for (int i = 0; i < nx; ++i)
                        flat[k * ny * nx + j * nx + i] = static_cast<float>(arr(i,j));
            return flat;
        };

        // Helper lambda to interleave 3 component 3D views into single flat vector
        auto interleave2 = [&](ViewType& a, ViewType& b) {
            std::vector<float> result(3 * nPoints);
            for (int k = 0; k < 1; ++k)
                for (int j = 0; j < ny; ++j)
                    for (int i = 0; i < nx; ++i) {
                        size_t idx = k * ny * nx + j * nx + i;
                        result[2*idx+0] = static_cast<float>(a(i,j));
                        result[2*idx+1] = static_cast<float>(b(i,j));
                    }
            return result;
        };

        // Prepare all arrays to write, applying scalings where needed
        std::vector<float> flatCharge = flatten(q);
        for(auto& val : flatCharge) val /= static_cast<float>(q0);

        std::vector<float> flatPotential = flatten(phi);
        for(auto& val : flatPotential) val /= static_cast<float>(phi_b);

        std::vector<float> flatVelocity = interleave2(u, v);
        for(auto& val : flatVelocity) val /= static_cast<float>(U_ref);

        std::vector<float> flatEField = interleave2(Ex, Ey);
        // Assuming your original code scaled electric field by (ny/phi_b)
        for(auto& val : flatEField) val *= static_cast<float>(ny / phi_b);

        std::vector<float> flatDensity = flatten(rho);

        // Write appended data array helper
        auto writeArray = [&](const std::vector<float>& data) {
            uint32_t byteLength = static_cast<uint32_t>(data.size() * sizeof(float));
            file.write(reinterpret_cast<const char*>(&byteLength), sizeof(uint32_t));
            file.write(reinterpret_cast<const char*>(data.data()), byteLength);
        };

        // Write arrays in the same order as offsets were computed
        writeArray(flatCharge);
        writeArray(flatPotential);
        writeArray(flatVelocity);
        writeArray(flatEField);
        writeArray(flatDensity);

        // Close XML
        file <<
            "  </AppendedData>\n"
            "</VTKFile>\n";

        file.close();
    }
};

#endif // VTIWRITER_2D_H
