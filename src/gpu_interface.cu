#include "gpu_interface.h"
#include <cuda_runtime.h>
#include <cub/cub.cuh>

__global__ void calc_cell_id_kernel(
    int N, 
    const double* x, 
    const double* y, 
    const double* z, 
    int* cell_id, 
    int* particle_id,
    double L, 
    double cell_size, 
    int grid_dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Initialize particle_id array 
    particle_id[i] = i; 

    // Periodic wrap 
    double px = x[i]; 
    if (px < 0) px += L; 
    if (px >= L) px -= L;
    double py = y[i]; 
    if (py < 0) py += L; 
    if (py >= L) py -= L;
    double pz = z[i]; 
    if (pz < 0) pz += L; 
    if (pz >= L) pz -= L;

    // Grid coordinates
    int cx = static_cast<int>(px / cell_size);
    int cy = static_cast<int>(py / cell_size);
    int cz = static_cast<int>(pz / cell_size);

    // Clamp
    if (cx >= grid_dim) cx = grid_dim - 1;
    if (cy >= grid_dim) cy = grid_dim - 1;
    if (cz >= grid_dim) cz = grid_dim - 1;

    // Flat index 3D -> 1D
    cell_id[i] = cx + cy * grid_dim + cz * grid_dim * grid_dim;
}

__global__ void find_cell_bounds_kernel(
    int N, 
    const int* cell_id_sorted, 
    int* cell_start, 
    int* cell_end
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int cell_id = cell_id_sorted[i];
    int cell_id_prev = (i == 0) ? -1 : cell_id_sorted[i - 1]; // if first particle, cell = -1

    if (cell_id != cell_id_prev) {
        cell_start[cell_id] = i;
    }
    
    int cell_id_next = (i == N - 1) ? -1 : cell_id_sorted[i + 1]; // if last particle, cell = -1
    if (cell_id != cell_id_next) {
        cell_end[cell_id] = i + 1;
    }
}

__global__ void build_neighbors_kernel(
    int N,
    const double* x, 
    const double* y, 
    const double* z,
    const int* sorted_p_ids, 
    const int* cell_start, 
    const int* cell_end,
    int* neighbor_list, 
    int* num_neighbors,
    double L, 
    double VERLET_CUT_SQ, 
    double cell_size, 
    int grid_dim, 
    int max_neighbors
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int i_real = sorted_p_ids[idx];
    int mol_i = i_real / 3;

    double xi = x[i_real];
    double yi = y[i_real];
    double zi = z[i_real];

    // Get cell
    int cx = static_cast<int>(xi / cell_size);
    int cy = static_cast<int>(yi / cell_size);
    int cz = static_cast<int>(zi / cell_size);

    int count = 0;

    // Check 3x3x3 neighbor cells
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                
                // Periodic wrap
                int ncx = (cx + dx + grid_dim) % grid_dim;
                int ncy = (cy + dy + grid_dim) % grid_dim;
                int ncz = (cz + dz + grid_dim) % grid_dim;
                
                // Neighbor cell ID
                int neighbor_cell_id = ncx + ncy * grid_dim + ncz * grid_dim * grid_dim;
                
                // Look up bounds
                int start_idx = cell_start[neighbor_cell_id];
                int end_idx = cell_end[neighbor_cell_id];

                if (start_idx == -1 || end_idx == 0) continue; // Empty cell

                // Check atoms in neighbor cell
                for (int k = start_idx; k < end_idx; ++k) {
                    int j_real = sorted_p_ids[k];
                    
                    if (i_real == j_real) continue;

                    // Exclude same molecule
                    int mol_j = j_real / 3;
                    if (mol_i == mol_j) continue;

                    double dx = xi - x[j_real];
                    double dy = yi - y[j_real];
                    double dz = zi - z[j_real];

                    if (dx > L * 0.5) dx -= L; 
                    if (dx < -L * 0.5) dx += L;
                    if (dy > L * 0.5) dy -= L; 
                    if (dy < -L * 0.5) dy += L;
                    if (dz > L * 0.5) dz -= L; 
                    if (dz < -L * 0.5) dz += L;

                    // Within cutoff
                    if (dx*dx + dy*dy + dz*dz < VERLET_CUT_SQ) {
                         if (count < max_neighbors) {
                             neighbor_list[i_real * max_neighbors + count] = j_real;
                             count++;
                         }
                    }
                }
            }
        }
    }
    num_neighbors[i_real] = count;
}

__global__ void compute_forces_kernel(
    int N,
    const double* x, 
    const double* y, 
    const double* z,
    const int* type,
    const int* neighbor_list, 
    const int* num_neighbors,
    double* fx, 
    double* fy, 
    double* fz,
    double L, 
    double CUTOFF_SQ,
    int max_neighbors
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double xi = x[i];
    double yi = y[i];
    double zi = z[i];
    int type_i = type[i];

    double fxi = 0.0;
    double fyi = 0.0;
    double fzi = 0.0;

    double sigmas[2] = {0.4000, 3.1506};
    double epsilons[2] = {0.0460, 0.1521};
    double charges[2] = {0.417, -0.834};

    const double MAX_FORCE = 10000.0;

    int neighbor_count = num_neighbors[i];

    for (int k = 0; k < neighbor_count; ++k) {
        int j = neighbor_list[i * max_neighbors + k];

        double dx = xi - x[j];
        double dy = yi - y[j];
        double dz = zi - z[j];
        
        if (dx > L * 0.5) dx -= L; 
        if (dx < -L * 0.5) dx += L;
        if (dy > L * 0.5) dy -= L; 
        if (dy < -L * 0.5) dy += L;
        if (dz > L * 0.5) dz -= L; 
        if (dz < -L * 0.5) dz += L;

        double r2 = dx*dx + dy*dy + dz*dz;

        if (r2 < CUTOFF_SQ && r2 > 0.0) {
            int type_j = type[j];

            double sigma = 0.5 * (sigmas[type_i] + sigmas[type_j]);
            double epsilon = sqrt(epsilons[type_i] * epsilons[type_j]);

            double r = sqrt(r2 + 1e-10);
            double inv_r2 = 1 / (r2 + 1e-10);
            double sig_inv_r2 = sigma * sigma * inv_r2;
            double sig_inv_r6 = sig_inv_r2 * sig_inv_r2 * sig_inv_r2;
            double sig_inv_r12 = sig_inv_r6 * sig_inv_r6;
            double f_lj = (24.0 * epsilon * inv_r2) * (2.0 * sig_inv_r12 - sig_inv_r6);

            double q1 = charges[type_i];
            double q2 = charges[type_j];
            double coulomb_force = (332.06 * q1 * q2) / r2;
            double f_elec = coulomb_force * (1.0 / r);

            double f_total = f_lj + f_elec;

            if (f_total > MAX_FORCE) f_total = MAX_FORCE;
            if (f_total < -MAX_FORCE) f_total = -MAX_FORCE;

            // Local accumulation
            fxi += f_total * dx;
            fyi += f_total * dy;
            fzi += f_total * dz;
        }
    }
    // Write back to global memory
    fx[i] += fxi;
    fy[i] += fyi;
    fz[i] += fzi;
}

void build_and_compute_gpu(
    int N, const double* h_x, const double* h_y, const double* h_z, const int* h_type,
    double* h_fx, double* h_fy, double* h_fz,
    SystemGPU& gpu, double L, double CUTOFF_SQ, double GRID_CELL_SIZE, int GRID_DIM,
    bool rebuild
) {
    cudaSetDevice(0);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(gpu.d_x, h_x, N * sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(gpu.d_y, h_y, N * sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(gpu.d_z, h_z, N * sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(gpu.d_type, h_type, N * sizeof(int), cudaMemcpyHostToDevice, stream);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    if (rebuild) {
        // Calculate unsorted IDs
        calc_cell_id_kernel<<<blocks, threads, 0, stream>>>(
            N, gpu.d_x, gpu.d_y, gpu.d_z, gpu.d_cell_id, gpu.d_particle_id, L, GRID_CELL_SIZE, GRID_DIM
        );

        // Cub sort
        cub::DoubleBuffer<int> d_keys(gpu.d_cell_id, gpu.d_cell_id_alt);
        cub::DoubleBuffer<int> d_values(gpu.d_particle_id, gpu.d_particle_id_alt);

        void *d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;

        // Allocate temporary memory
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, N, 0, sizeof(int)*8, stream);
        cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream);

        // Run sort
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, N, 0, sizeof(int)*8, stream);
        
        int* d_cell_id_sorted = d_keys.Current();
        int* d_particle_id_sorted = d_values.Current();

        // Clear grid
        cudaMemsetAsync(gpu.d_cell_start, -1, gpu.num_cells * sizeof(int), stream);
        cudaMemsetAsync(gpu.d_cell_end, 0, gpu.num_cells * sizeof(int), stream);
        
        // Find bounds
        find_cell_bounds_kernel<<<blocks, threads, 0, stream>>>(N, d_cell_id_sorted, gpu.d_cell_start, gpu.d_cell_end);

        // Build neighbors
        double SKIN_CUTOFF_SQ = (sqrt(CUTOFF_SQ) + 2.0) * (sqrt(CUTOFF_SQ) + 2.0);
        build_neighbors_kernel<<<blocks, threads, 0, stream>>>(
            N, gpu.d_x, gpu.d_y, gpu.d_z, d_particle_id_sorted, 
            gpu.d_cell_start, gpu.d_cell_end, 
            gpu.d_neighbor_list, gpu.d_num_neighbors, 
            L, SKIN_CUTOFF_SQ, GRID_CELL_SIZE, GRID_DIM, gpu.max_neighbors
        );

        cudaFreeAsync(d_temp_storage, stream);
    }

    cudaMemsetAsync(gpu.d_fx, 0, N * sizeof(double), stream);
    cudaMemsetAsync(gpu.d_fy, 0, N * sizeof(double), stream);
    cudaMemsetAsync(gpu.d_fz, 0, N * sizeof(double), stream);

    compute_forces_kernel<<<blocks, threads, 0, stream>>>(
        N, gpu.d_x, gpu.d_y, gpu.d_z, gpu.d_type, 
        gpu.d_neighbor_list, gpu.d_num_neighbors, 
        gpu.d_fx, gpu.d_fy, gpu.d_fz, 
        L, CUTOFF_SQ, gpu.max_neighbors
    );
    
    cudaMemcpyAsync(h_fx, gpu.d_fx, N * sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_fy, gpu.d_fy, N * sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_fz, gpu.d_fz, N * sizeof(double), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

void SystemGPU::allocate(int num_atoms, int max_n, int grid_dim) {
    N = num_atoms;
    max_neighbors = max_n;
    num_cells = grid_dim * grid_dim * grid_dim;

    // Standard arrays
    cudaMalloc(&d_x, N * sizeof(double));
    cudaMalloc(&d_y, N * sizeof(double));
    cudaMalloc(&d_z, N * sizeof(double));
    cudaMalloc(&d_fx, N * sizeof(double));
    cudaMalloc(&d_fy, N * sizeof(double));
    cudaMalloc(&d_fz, N * sizeof(double));
    cudaMalloc(&d_type, N * sizeof(int));
    cudaMalloc(&d_neighbor_list, N * max_neighbors * sizeof(int));
    cudaMalloc(&d_num_neighbors, N * sizeof(int));

    // Sort arrays (main)
    cudaMalloc(&d_cell_id, N * sizeof(int));
    cudaMalloc(&d_particle_id, N * sizeof(int));
    
    // Sort arrays (alternate)
    cudaMalloc(&d_cell_id_alt, N * sizeof(int));
    cudaMalloc(&d_particle_id_alt, N * sizeof(int));

    // Grid arrays
    cudaMalloc(&d_cell_start, num_cells * sizeof(int));
    cudaMalloc(&d_cell_end, num_cells * sizeof(int));
}

void SystemGPU::cleanup() {
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    cudaFree(d_fx); cudaFree(d_fy); cudaFree(d_fz);
    cudaFree(d_type); cudaFree(d_neighbor_list); cudaFree(d_num_neighbors);
    cudaFree(d_cell_id); cudaFree(d_particle_id);
    cudaFree(d_cell_id_alt); cudaFree(d_particle_id_alt);
    cudaFree(d_cell_start); cudaFree(d_cell_end);
}