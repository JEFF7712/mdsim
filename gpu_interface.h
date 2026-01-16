#ifndef GPU_INTERFACE_H
#define GPU_INTERFACE_H

#include <cuda_runtime.h>

struct SystemGPU {
    // Atom Data
    double *d_x, *d_y, *d_z;
    double *d_fx, *d_fy, *d_fz;
    int *d_type;
    
    int *d_neighbor_list; 
    int *d_num_neighbors; 

    int *d_cell_id;
    int *d_particle_id;
    int *d_cell_id_alt;    
    int *d_particle_id_alt;
    
    int *d_cell_start;
    int *d_cell_end;

    int N; // Number of atoms
    int max_neighbors;
    int num_cells;

    void allocate(int num_atoms, int max_n, int grid_dim);
    void cleanup();
};

void build_and_compute_gpu(
    int N,
    const double* h_x, 
    const double* h_y, 
    const double* h_z, 
    const int* h_type,
    double* h_fx, 
    double* h_fy, 
    double* h_fz,
    SystemGPU& gpu,
    double L, 
    double CUTOFF_SQ,
    double GRID_CELL_SIZE, 
    int GRID_DIM,
    bool rebuild
);

#endif