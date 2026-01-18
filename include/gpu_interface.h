#ifndef GPU_INTERFACE_H
#define GPU_INTERFACE_H

#include <vector>
#include <cuda_runtime.h>

struct SystemGPU {
    float *d_x, *d_y, *d_z;
    float *d_fx, *d_fy, *d_fz;
    int *d_type;
    
    int *d_neighbor_list; 
    int *d_num_neighbors; 

    int *d_cell_id;
    int *d_particle_id;
    int *d_cell_id_alt;    
    int *d_particle_id_alt;
    
    int *d_cell_start;
    int *d_cell_end;

    int N;
    int max_neighbors;
    int num_cells;

    float *d_sigma;
    float *d_epsilon;
    float *d_charge;

    float *d_energy;

    int *d_exclusion_list;
    int *d_exclusion_start;
    int *d_exclusion_count;
    int max_exclusions_per_atom;

    void allocate(int num_atoms, int max_n, int grid_dim);
    void cleanup();
    void set_atom_params(int num_types, const float* h_sigma, const float* h_epsilon, const float* h_charge);
    void set_exclusions(int num_atoms, const std::vector<std::vector<int>>& exclusion_list);
};

void build_and_compute_gpu(
    int N,
    const float* h_x, 
    const float* h_y, 
    const float* h_z, 
    const int* h_type,
    float* h_fx, 
    float* h_fy, 
    float* h_fz,
    SystemGPU& gpu,
    float L, 
    float CUTOFF_SQ,
    float VERLET_CUTOFF_SQ,
    float GRID_CELL_SIZE, 
    int GRID_DIM,
    int atoms_per_molecule,
    float* h_gpu_pe,
    bool rebuild,
    const std::vector<std::vector<int>>& exclusions
);

#endif