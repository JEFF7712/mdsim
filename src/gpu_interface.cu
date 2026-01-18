#include "gpu_interface.h"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <vector>
#include <cstdio>

__global__ void calc_cell_id_kernel(
    int N, 
    const float* x, 
    const float* y, 
    const float* z, 
    int* cell_id, 
    int* particle_id,
    float L, 
    float cell_size, 
    int grid_dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Initialize particle_id array 
    particle_id[i] = i; 

    // Periodic wrap 
    float px = x[i]; 
    if (px < 0) px += L; 
    if (px >= L) px -= L;
    float py = y[i]; 
    if (py < 0) py += L; 
    if (py >= L) py -= L;
    float pz = z[i]; 
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
    const float* x, 
    const float* y, 
    const float* z,
    const int* sorted_p_ids, 
    const int* cell_start, 
    const int* cell_end,
    int* neighbor_list, 
    int* num_neighbors,
    float L, 
    float VERLET_CUT_SQ, 
    float cell_size, 
    int grid_dim, 
    int max_neighbors,
    int atoms_per_molecule
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int i_real = sorted_p_ids[idx];

    float xi = x[i_real];
    float yi = y[i_real];
    float zi = z[i_real];

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

                    float dx = xi - x[j_real];
                    float dy = yi - y[j_real];
                    float dz = zi - z[j_real];

                    if (dx > L * 0.5f) dx -= L; 
                    if (dx < -L * 0.5f) dx += L;
                    if (dy > L * 0.5f) dy -= L; 
                    if (dy < -L * 0.5f) dy += L;
                    if (dz > L * 0.5f) dz -= L; 
                    if (dz < -L * 0.5f) dz += L;

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
    const float* x, 
    const float* y, 
    const float* z,
    const int* type,
    const int* neighbor_list, 
    const int* num_neighbors,
    float* fx, 
    float* fy, 
    float* fz,
    float L, 
    float CUTOFF_SQ,
    int max_neighbors,
    float* d_energy,
    const float* __restrict__ param_sigma,
    const float* __restrict__ param_epsilon,
    const float* __restrict__ param_charge,
    const int* __restrict__ d_exclusion_list,
    const int* __restrict__ d_exclusion_start,
    const int* __restrict__ d_exclusion_count
) {
    __shared__ float shared_energy[256];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float local_energy = 0.0f;
    float fxi = 0.0f;
    float fyi = 0.0f;
    float fzi = 0.0f;

    const float MAX_FORCE = 10000.0f;

    if (i < N) {
        float xi = x[i];
        float yi = y[i];
        float zi = z[i];
        int type_i = type[i];
        int neighbor_count = num_neighbors[i];

        for (int k = 0; k < neighbor_count; ++k) {
            int j = neighbor_list[i * max_neighbors + k];

            // Check if j is in exclusion list for i
            bool excluded = false;
            int excl_start = d_exclusion_start[i];
            int excl_count = d_exclusion_count[i];
            for (int e = 0; e < excl_count; ++e) {
                if (d_exclusion_list[excl_start + e] == j) {
                    excluded = true;
                    break;
                }
            }
            if (excluded) continue;

            float dx = xi - x[j];
            float dy = yi - y[j];
            float dz = zi - z[j];
            
            // Periodic wrap
            if (dx > L * 0.5f) dx -= L; 
            if (dx < -L * 0.5f) dx += L;
            if (dy > L * 0.5f) dy -= L; 
            if (dy < -L * 0.5f) dy += L;
            if (dz > L * 0.5f) dz -= L; 
            if (dz < -L * 0.5f) dz += L;

            float r2 = dx*dx + dy*dy + dz*dz;

            if (r2 < 1e-6f) continue;

            if (r2 < CUTOFF_SQ && r2 > 0.0f) {
                int type_j = type[j];

                float sigma_i = param_sigma[type_i];
                float sigma_j = param_sigma[type_j];
                float eps_i = param_epsilon[type_i];
                float eps_j = param_epsilon[type_j];
                float q_i = param_charge[type_i];
                float q_j = param_charge[type_j];

                float sigma = 0.5f * (sigma_i + sigma_j);
                float epsilon = sqrtf(eps_i * eps_j);

                float r = sqrtf(r2 + 1e-10f);
                float inv_r2 = 1.0f / (r2 + 1e-10f);
                float sig_inv_r2 = sigma * sigma * inv_r2;
                float sig_inv_r6 = sig_inv_r2 * sig_inv_r2 * sig_inv_r2;
                float sig_inv_r12 = sig_inv_r6 * sig_inv_r6;
                
                float f_lj = (24.0f * epsilon * inv_r2) * (2.0f * sig_inv_r12 - sig_inv_r6);
                float f_elec = (332.0637f * q_i * q_j) / (r2 * r);
                
                float f_total = f_lj + f_elec;

                float pair_energy = 4.0f * epsilon * (sig_inv_r12 - sig_inv_r6) + (332.06f * q_i * q_j) / r;
                
                float r_cut_sq = CUTOFF_SQ; 
                float r_switch_sq = 0.81f * r_cut_sq;

                if (r2 > r_switch_sq) {
                    float r_cut = sqrtf(r_cut_sq);
                    float r_switch = sqrtf(r_switch_sq);
                    
                    float dist = r_cut - r_switch;
                    float u = (r - r_switch) / dist;
                    
                    float s_val = 1.0f - u*u * (3.0f - 2.0f * u);
                    
                    float ds_dr = (-6.0f * u + 6.0f * u * u) / dist;

                    f_total = f_total * s_val - (pair_energy * ds_dr * (1.0f/r));
                    
                    pair_energy *= s_val;
                }

                if (f_total > MAX_FORCE) f_total = MAX_FORCE;
                if (f_total < -MAX_FORCE) f_total = -MAX_FORCE;

                fxi += f_total * dx;
                fyi += f_total * dy;
                fzi += f_total * dz;
                
                local_energy += 0.5f * pair_energy;
            }
        }
        fx[i] += fxi;
        fy[i] += fyi;
        fz[i] += fzi;
    }

    shared_energy[tid] = local_energy;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_energy[tid] += shared_energy[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(d_energy, shared_energy[0]);
    }
}

void build_and_compute_gpu(
    int N, const float* h_x, const float* h_y, const float* h_z, const int* h_type,
    float* h_fx, float* h_fy, float* h_fz,
    SystemGPU& gpu, float L, float CUTOFF_SQ, float VERLET_CUTOFF_SQ, float GRID_CELL_SIZE, int GRID_DIM,
    int atoms_per_molecule, float* h_gpu_pe, bool rebuild, const std::vector<std::vector<int>>& exclusions
) {
    int current_device;
    cudaError_t err = cudaGetDevice(&current_device);
    
    if (err != cudaSuccess) {
        err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            printf("CUDA ERROR: %s\n", cudaGetErrorString(err));
            exit(1);
        }
    }
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(gpu.d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(gpu.d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(gpu.d_z, h_z, N * sizeof(float), cudaMemcpyHostToDevice, stream);
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
        build_neighbors_kernel<<<blocks, threads, 0, stream>>>(
            N, gpu.d_x, gpu.d_y, gpu.d_z, d_particle_id_sorted, 
            gpu.d_cell_start, gpu.d_cell_end, 
            gpu.d_neighbor_list, gpu.d_num_neighbors, 
            L, VERLET_CUTOFF_SQ, GRID_CELL_SIZE, GRID_DIM, gpu.max_neighbors, atoms_per_molecule
        );

        cudaFreeAsync(d_temp_storage, stream);
    }

    cudaMemsetAsync(gpu.d_fx, 0, N * sizeof(float), stream);
    cudaMemsetAsync(gpu.d_fy, 0, N * sizeof(float), stream);
    cudaMemsetAsync(gpu.d_fz, 0, N * sizeof(float), stream);
    cudaMemsetAsync(gpu.d_energy, 0, sizeof(float), stream);

    compute_forces_kernel<<<blocks, threads, 0, stream>>>(
        N, gpu.d_x, gpu.d_y, gpu.d_z, gpu.d_type, 
        gpu.d_neighbor_list, gpu.d_num_neighbors, 
        gpu.d_fx, gpu.d_fy, gpu.d_fz, 
        L, CUTOFF_SQ, gpu.max_neighbors, gpu.d_energy,
        gpu.d_sigma, gpu.d_epsilon, gpu.d_charge,
        gpu.d_exclusion_list, gpu.d_exclusion_start, gpu.d_exclusion_count
    );
    
    cudaMemcpyAsync(h_fx, gpu.d_fx, N * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_fy, gpu.d_fy, N * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_fz, gpu.d_fz, N * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_gpu_pe, gpu.d_energy, sizeof(float), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

void SystemGPU::allocate(int num_atoms, int max_n, int grid_dim) {
    cudaDeviceReset(); 
    
    N = num_atoms;
    max_neighbors = max_n;
    num_cells = grid_dim * grid_dim * grid_dim;
    max_exclusions_per_atom = 20;
    
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMalloc(&d_z, N * sizeof(float));
    cudaMalloc(&d_fx, N * sizeof(float));
    cudaMalloc(&d_fy, N * sizeof(float));
    cudaMalloc(&d_fz, N * sizeof(float));
    cudaMalloc(&d_type, N * sizeof(int));
    cudaMalloc(&d_neighbor_list, N * max_neighbors * sizeof(int));
    cudaMalloc(&d_num_neighbors, N * sizeof(int));
    cudaMalloc(&d_cell_id, N * sizeof(int));
    cudaMalloc(&d_particle_id, N * sizeof(int));
    cudaMalloc(&d_cell_id_alt, N * sizeof(int));
    cudaMalloc(&d_particle_id_alt, N * sizeof(int));
    cudaMalloc(&d_cell_start, num_cells * sizeof(int));
    cudaMalloc(&d_cell_end, num_cells * sizeof(int));
    cudaMalloc(&d_energy, sizeof(float));
    cudaMalloc(&d_exclusion_list, N * max_exclusions_per_atom * sizeof(int));
    cudaMalloc(&d_exclusion_start, N * sizeof(int));
    cudaMalloc(&d_exclusion_count, N * sizeof(int));
    cudaMemset(d_exclusion_count, 0, N * sizeof(int));

}

void SystemGPU::set_atom_params(int num_types, const float* h_sigma, const float* h_epsilon, const float* h_charge) {
    cudaMalloc(&d_sigma, num_types * sizeof(float));
    cudaMalloc(&d_epsilon, num_types * sizeof(float));
    cudaMalloc(&d_charge, num_types * sizeof(float));

    cudaMemcpy(d_sigma, h_sigma, num_types * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_epsilon, h_epsilon, num_types * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_charge, h_charge, num_types * sizeof(float), cudaMemcpyHostToDevice);
}

void SystemGPU::set_exclusions(int num_atoms, const std::vector<std::vector<int>>& exclusion_list) {
    // Flatten the exclusion list
    std::vector<int> h_exclusion_list;
    std::vector<int> h_exclusion_start(num_atoms);
    std::vector<int> h_exclusion_count(num_atoms);
    
    int total_exclusions = 0;
    for (int i = 0; i < num_atoms; ++i) {
        h_exclusion_start[i] = h_exclusion_list.size();
        h_exclusion_count[i] = exclusion_list[i].size();
        total_exclusions += h_exclusion_count[i];
        
        if (h_exclusion_count[i] > max_exclusions_per_atom) {
            h_exclusion_count[i] = max_exclusions_per_atom;
        }
        
        for (int j = 0; j < h_exclusion_count[i]; ++j) {
            h_exclusion_list.push_back(exclusion_list[i][j]);
        }
    }
    
    // Copy to GPU
    cudaMemcpy(d_exclusion_list, h_exclusion_list.data(), h_exclusion_list.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_exclusion_start, h_exclusion_start.data(), num_atoms * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_exclusion_count, h_exclusion_count.data(), num_atoms * sizeof(int), cudaMemcpyHostToDevice);
}

void SystemGPU::cleanup() {
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    cudaFree(d_fx); cudaFree(d_fy); cudaFree(d_fz);
    cudaFree(d_type); cudaFree(d_neighbor_list); cudaFree(d_num_neighbors);
    cudaFree(d_cell_id); cudaFree(d_particle_id);
    cudaFree(d_cell_id_alt); cudaFree(d_particle_id_alt);
    cudaFree(d_cell_start); cudaFree(d_cell_end);
    cudaFree(d_energy);
    cudaFree(d_sigma); cudaFree(d_epsilon); cudaFree(d_charge);
    cudaFree(d_exclusion_list); cudaFree(d_exclusion_start); cudaFree(d_exclusion_count);
}