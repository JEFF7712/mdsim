#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <omp.h>

// Distance: Angstroms (Å)
// Time: femtoseconds (fs)
// Mass: atomic mass units (amu)
// Energy: kcal/mol
// Temperature: Kelvin (K)
// Force: kcal/mol/Å

// Parameters
const int num_molecules = 1000; // Number of molecules
const double L = 31.0; // Box length (Angstroms)
const double dt = 0.5; // Time step (fs)
const double t_max = 2500.0; // Maximum simulation time (fs)
enum class Ensemble { NVE, NVT };
const Ensemble ensemble = Ensemble::NVT; // Ensemble type
const double kb = 0.001987204; // Boltzmann constant (kcal/mol/K)
const double target_temp = 300.0; // Target temperature (K)

// Conversion Factor:
const double F_TO_ACC = 4.184e-4; // [kcal/mol/Angstrom] -> [amu * (A/fs^2)]
const double KE_TO_KCAL = 2390.057; // [amu * (A/fs)^2] -> [kcal/mol]

// Force cutoff
const double CUTOFF = 10.0; // Cutoff distance (Angstroms)
const double CUTOFF_SQ = CUTOFF * CUTOFF;
const double SKIN = 2.0;
const double VERLET_CUTOFF = CUTOFF + SKIN;
const double VERLET_CUTOFF_SQ = VERLET_CUTOFF * VERLET_CUTOFF;

// Random number generator
std::mt19937& get_rng() {
    // static unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    static unsigned seed = 42;
    static std::mt19937 engine(seed);
    return engine;
}

// Atom type structure
struct AtomType {
    std::string name;
    double mass; // amu
    double sigma; // Angstroms
    double epsilon; // kcal/mol
    double charge; 
};

// Bond structure
struct Bond {
    int atom1;
    int atom2;
    double k; // bond strength (kcal/mol/Angstrom^2)
    double rb; // bond length (Angstroms)
};

struct Angle {
    int atom1; 
    int atom2;
    int atom3;
    double k; // Stiffness (kcal/mol/rad^2)
    double theta0; // Equilibrium angle (radians)
};

const std::vector<AtomType> ATOM_TYPES = {
    {"H", 1.008, 0.4000, 0.0460, 0.417},    // Type 0: Hydrogen
    {"O", 15.999, 3.1506, 0.1521, -0.834}   // Type 1: Oxygen
};


struct System {
    std::vector<double> x, y, z;
    std::vector<double> vx, vy, vz;
    std::vector<double> fx, fy, fz;
    std::vector<int> type;
    std::vector<int> id;
    std::vector<std::vector<int>> exclusions;
    std::vector<std::vector<int>> neighbors;

    void resize(int n) {
        x.resize(n); y.resize(n); z.resize(n);
        vx.resize(n); vy.resize(n); vz.resize(n);
        fx.resize(n); fy.resize(n); fz.resize(n);
        type.resize(n);
        id.resize(n);
        exclusions.resize(n);
        neighbors.resize(n);
    }
};

void init_water_box(System& sys, std::vector<Bond>& bonds, std::vector<Angle>& angles) {
    int per_side = std::ceil(std::cbrt(num_molecules));
    double spacing = L / per_side;
    sys.resize(num_molecules * 3);
    int p_idx = 0;
    
    for (int x = 0; x < per_side; ++x) {
        for (int y = 0; y < per_side; ++y) {
            for (int z = 0; z < per_side; ++z) {
                // Check if there is enough particles for another water molecule
                if (p_idx + 2 >= sys.x.size()) return;

                double cx = x * spacing + spacing/2;
                double cy = y * spacing + spacing/2;
                double cz = z * spacing + spacing/2;

                // O

                sys.id[p_idx] = p_idx;
                sys.type[p_idx] = 1;
                sys.x[p_idx] = cx;
                sys.y[p_idx] = cy;
                sys.z[p_idx] = cz;
                
                // H1
                sys.id[p_idx + 1] = p_idx + 1;
                sys.type[p_idx + 1] = 0;
                sys.x[p_idx + 1] = cx + 0.6;
                sys.y[p_idx + 1] = cy + 0.6;
                sys.z[p_idx + 1] = cz;

                // H2
                sys.id[p_idx + 2] = p_idx + 2;
                sys.type[p_idx + 2] = 0;
                sys.x[p_idx + 2] = cx - 0.6;
                sys.y[p_idx + 2] = cy + 0.6;
                sys.z[p_idx + 2] = cz;

                // Create Bonds
                bonds.push_back({sys.id[p_idx], sys.id[p_idx + 1], 450.0, 0.9572});
                bonds.push_back({sys.id[p_idx], sys.id[p_idx + 2], 450.0, 0.9572});

                // Create Angle
                double theta_eq = 104.52 * 3.14159 / 180.0;
                angles.push_back({sys.id[p_idx + 1], sys.id[p_idx], sys.id[p_idx + 2], 100.0, theta_eq});

                // Add Exclusions
                sys.exclusions[sys.id[p_idx]].push_back(sys.id[p_idx + 1]);
                sys.exclusions[sys.id[p_idx]].push_back(sys.id[p_idx + 2]);
                sys.exclusions[sys.id[p_idx + 1]].push_back(sys.id[p_idx]);
                sys.exclusions[sys.id[p_idx + 2]].push_back(sys.id[p_idx]);
                sys.exclusions[sys.id[p_idx + 1]].push_back(sys.id[p_idx + 2]);
                sys.exclusions[sys.id[p_idx + 2]].push_back(sys.id[p_idx + 1]);

                p_idx += 3;
            }
        }
    }
}

struct CellList {
    double cell_size;
    int grid_dim;
    int num_cells;

    std::vector<int> head;
    std::vector<int> next;

    CellList(double box_length, double cutoff, int num_particles) {
        // How many cells fit in the box
        grid_dim = static_cast<int>(std::floor(box_length / cutoff));
        // At least one cell
        if (grid_dim == 0) grid_dim = 1; 
        
        // Actual size of cell
        cell_size = box_length / grid_dim;

        // Total number of cells 
        num_cells = grid_dim * grid_dim * grid_dim;

        // Resize arrays to match number of cells and particles
        head.resize(num_cells);
        next.resize(num_particles);
    }

    // Convert position to cell index
    int get_cell_index(double x, double y, double z) const {

        // Wrap positions to be within box using mod
        x = std::fmod(x, L);
        if (x < 0.0) x += L;
        
        y = std::fmod(y, L);
        if (y < 0.0) y += L;
        
        z = std::fmod(z, L);
        if (z < 0.0) z += L;

        // Find cell coordinates
        int cx = static_cast<int>(x / cell_size);
        int cy = static_cast<int>(y / cell_size);
        int cz = static_cast<int>(z / cell_size);
        
        // Ensure cell indices are within bounds
        cx = std::min(cx, grid_dim - 1);
        cy = std::min(cy, grid_dim - 1);
        cz = std::min(cz, grid_dim - 1);

        // Flatten 3D index to 1D
        return cx + cy * grid_dim + cz * grid_dim * grid_dim;
    }

    void build(const System& sys) {

        // Fill head with -1 (-1 indicates last particle in cell)
        std::fill(head.begin(), head.end(), -1);

        // Assign each particle to a cell
        for (int i = 0; i < sys.x.size(); ++i) {
            int cell = get_cell_index(sys.x[i], sys.y[i], sys.z[i]);

            next[i] = head[cell];
            head[cell] = i;
        }
    }


};

void build_verlet_lists(System& sys, CellList& cells) {
    cells.build(sys);
    int dim = cells.grid_dim;

    #pragma omp parallel for schedule(dynamic) collapse(3)
    for (int cx = 0; cx < dim; ++cx) {
        for (int cy = 0; cy < dim; ++cy) {
            for (int cz = 0; cz < dim; ++cz) {
                int cell_index = cx + cy * dim + cz * dim * dim;
                int i = cells.head[cell_index];

                while (i != -1) {
                    // Clear old list
                    sys.neighbors[i].clear();
                    
                    // Guess ~100 neighbors
                    sys.neighbors[i].reserve(100); 

                    // Search 3x3x3 block
                    for (int nx = cx - 1; nx <= cx + 1; ++nx) {
                        for (int ny = cy - 1; ny <= cy + 1; ++ny) {
                            for (int nz = cz - 1; nz <= cz + 1; ++nz) {
                                int wrapped_nx = (nx + dim) % dim;
                                int wrapped_ny = (ny + dim) % dim;
                                int wrapped_nz = (nz + dim) % dim;
                                int neighbor_cell = wrapped_nx + wrapped_ny * dim + wrapped_nz * dim * dim;

                                int j = cells.head[neighbor_cell];
                                while (j != -1) {
                                    if (i != j) {
                                        
                                        // Squared Distance Check
                                        double dx = sys.x[i] - sys.x[j];
                                        double dy = sys.y[i] - sys.y[j];
                                        double dz = sys.z[i] - sys.z[j];
                                        
                                        if (dx > L/2) dx -= L; if (dx < -L/2) dx += L;
                                        if (dy > L/2) dy -= L; if (dy < -L/2) dy += L;
                                        if (dz > L/2) dz -= L; if (dz < -L/2) dz += L;

                                        double r2 = dx*dx + dy*dy + dz*dz;

                                        if (r2 < VERLET_CUTOFF_SQ) {
                                            bool excluded = false;
                                            for (int ex : sys.exclusions[i]) {
                                                if (ex == j) { excluded = true; break; }
                                            }
                                            if (!excluded) {
                                                sys.neighbors[i].push_back(j);
                                            }
                                        }
                                    }
                                    j = cells.next[j];
                                }
                            }
                        }
                    }
                    i = cells.next[i];
                }
            }
        }
    }
}


void compute_bond_forces(System& sys, const std::vector<Bond>& bonds, double& pe_total) {
    double local_pe = 0.0;
    
    #pragma omp parallel for reduction(+:local_pe)
    for (const auto& b : bonds) {

        double dx = sys.x[b.atom1] - sys.x[b.atom2];
        double dy = sys.y[b.atom1] - sys.y[b.atom2];
        double dz = sys.z[b.atom1] - sys.z[b.atom2];

        if (dx > L/2) dx -= L; 
        if (dx < -L/2) dx += L;
        if (dy > L/2) dy -= L; 
        if (dy < -L/2) dy += L;
        if (dz > L/2) dz -= L; 
        if (dz < -L/2) dz += L;

        double r2 = dx*dx + dy*dy + dz*dz;
        double r = std::sqrt(r2);

        double delta = r - b.rb;
        double f_mag = -b.k * delta;
        local_pe += 0.5 * b.k * delta * delta;

        double fx = f_mag * (dx / r);
        double fy = f_mag * (dy / r);
        double fz = f_mag * (dz / r);

        #pragma omp atomic
        sys.fx[b.atom1] += fx;
        #pragma omp atomic 
        sys.fy[b.atom1] += fy; 
        #pragma omp atomic
        sys.fz[b.atom1] += fz;
        #pragma omp atomic
        sys.fx[b.atom2] -= fx; 
        #pragma omp atomic
        sys.fy[b.atom2] -= fy;
        #pragma omp atomic 
        sys.fz[b.atom2] -= fz;
    }
    pe_total += local_pe;
}

void compute_angle_forces(System& sys, const std::vector<Angle>& angles, double& pe_total) {
    double local_pe = 0.0;
    
    #pragma omp parallel for reduction(+:local_pe)
    for (const auto& ang : angles) {

        double r21x = sys.x[ang.atom1] - sys.x[ang.atom2];
        double r21y = sys.y[ang.atom1] - sys.y[ang.atom2];
        double r21z = sys.z[ang.atom1] - sys.z[ang.atom2];

        double r23x = sys.x[ang.atom3] - sys.x[ang.atom2];
        double r23y = sys.y[ang.atom3] - sys.y[ang.atom2];
        double r23z = sys.z[ang.atom3] - sys.z[ang.atom2];

        // Periodic Boundaries
        if (r21x > L/2) r21x -= L; 
        if (r21x < -L/2) r21x += L;
        if (r21y > L/2) r21y -= L; 
        if (r21y < -L/2) r21y += L;
        if (r21z > L/2) r21z -= L; 
        if (r21z < -L/2) r21z += L;

        if (r23x > L/2) r23x -= L; 
        if (r23x < -L/2) r23x += L;
        if (r23y > L/2) r23y -= L; 
        if (r23y < -L/2) r23y += L;
        if (r23z > L/2) r23z -= L; 
        if (r23z < -L/2) r23z += L;

        // Magnitudes squared
        double r21_sq = r21x*r21x + r21y*r21y + r21z*r21z;
        double r23_sq = r23x*r23x + r23y*r23y + r23z*r23z;
        double r21 = std::sqrt(r21_sq);
        double r23 = std::sqrt(r23_sq);

        // Dot product
        double dot = r21x*r23x + r21y*r23y + r21z*r23z;
        double cos_theta = dot / (r21 * r23);

        // Make sure cos_theta is in valid range for acos
        if (cos_theta > 1.0) cos_theta = 1.0;
        if (cos_theta < -1.0) cos_theta = -1.0;

        double theta = std::acos(cos_theta);
        double d_theta = theta - ang.theta0;
        
        // Potential Energy
        local_pe += 0.5 * ang.k * d_theta * d_theta;

        // Force Magnitude
        double force_factor = -ang.k * d_theta;
        double sin_theta = std::sin(theta);
        if (std::abs(sin_theta) < 1e-6) sin_theta = 1e-6; // Avoid div by zero

        double coef1 = force_factor / (r21 * sin_theta);
        double coef2 = force_factor / (r23 * sin_theta);

        double c1 = force_factor / (r21_sq * sin_theta);
        double c2 = force_factor / (r23_sq * sin_theta);

        double f1x = c1 * (r21x * cos_theta - r23x * (r21/r23));
        double f1y = c1 * (r21y * cos_theta - r23y * (r21/r23));
        double f1z = c1 * (r21z * cos_theta - r23z * (r21/r23));

        double f3x = c2 * (r23x * cos_theta - r21x * (r23/r21));
        double f3y = c2 * (r23y * cos_theta - r21y * (r23/r21));
        double f3z = c2 * (r23z * cos_theta - r21z * (r23/r21));

        // Apply forces
        #pragma omp atomic
        sys.fx[ang.atom1] += f1x; 
        #pragma omp atomic
        sys.fy[ang.atom1] += f1y; 
        #pragma omp atomic
        sys.fz[ang.atom1] += f1z;
        #pragma omp atomic
        sys.fx[ang.atom3] += f3x; 
        #pragma omp atomic
        sys.fy[ang.atom3] += f3y; 
        #pragma omp atomic
        sys.fz[ang.atom3] += f3z;
        
        // Center Atom
        #pragma omp atomic
        sys.fx[ang.atom2] -= (f1x + f3x);
        #pragma omp atomic
        sys.fy[ang.atom2] -= (f1y + f3y);
        #pragma omp atomic
        sys.fz[ang.atom2] -= (f1z + f3z);
    }
    pe_total += local_pe;
} 

void compute_forces(System& sys, double& pe_total) {
    double global_pe = 0.0;
    
    // Loop over all cells
    #pragma omp parallel for schedule(dynamic) reduction(+:global_pe)
    for (int i = 0; i < sys.x.size(); ++i) {
        double f1x = 0.0;
        double f1y = 0.0;
        double f1z = 0.0;
        double local_pe = 0.0;
        const AtomType& type1 = ATOM_TYPES[sys.type[i]];
        for (int j : sys.neighbors[i]) {
            double dx = sys.x[i] - sys.x[j];
            double dy = sys.y[i] - sys.y[j];
            double dz = sys.z[i] - sys.z[j];

            // Periodic boundary
            if (dx > L/2) dx -= L; 
            if (dx < -L/2) dx += L;
            if (dy > L/2) dy -= L; 
            if (dy < -L/2) dy += L;
            if (dz > L/2) dz -= L; 
            if (dz < -L/2) dz += L;

            double r2 = dx*dx + dy*dy + dz*dz;

            // Apply LJ and Coulomb forces
            if (r2 < CUTOFF_SQ && r2 > 0.0) {
                const AtomType& type1 = ATOM_TYPES[sys.type[i]];
                const AtomType& type2 = ATOM_TYPES[sys.type[j]];

                double sigma = 0.5 * (type1.sigma + type2.sigma);
                double epsilon = std::sqrt(type1.epsilon * type2.epsilon);
                
                double inv_r = 1.0 / std::sqrt(r2);
                double inv_r2 = inv_r * inv_r;
                double r = r2 * inv_r;
                double sig_inv_r2 = sigma * sigma * inv_r2;
                double sig_inv_r6 = sig_inv_r2 * sig_inv_r2 * sig_inv_r2;
                double sig_inv_r12 = sig_inv_r6 * sig_inv_r6;
                double f_lj = (24.0 * epsilon * inv_r2) * (2.0 * sig_inv_r12 - sig_inv_r6);

                double q1 = type1.charge;
                double q2 = type2.charge;
                double coulomb_force = (332.06 * q1 * q2) / r2;
                double f_elec = coulomb_force * (1.0 / r);

                double f_total = f_lj + f_elec;

                f1x += f_total * dx;
                f1y += f_total * dy;
                f1z += f_total * dz;

                double u_lj = 4.0 * epsilon * (sig_inv_r12 - sig_inv_r6);
                double u_elec = (332.06 * type1.charge * type2.charge) / r;

                double inv_rc2 = 1.0 / CUTOFF_SQ;
                double sig_rc2 = sigma * sigma * inv_rc2;
                double sig_rc6 = sig_rc2 * sig_rc2 * sig_rc2;
                double sig_rc12 = sig_rc6 * sig_rc6;
                double u_shift = 4.0 * epsilon * (sig_rc12 - sig_rc6); 
                double u_elec_shift = (332.06 * type1.charge * type2.charge) / CUTOFF;

                local_pe += 0.5 * (u_lj - u_shift);
                local_pe += 0.5 * (u_elec - u_elec_shift);
            }
        }
        sys.fx[i] += f1x;
        sys.fy[i] += f1y;
        sys.fz[i] += f1z;
        global_pe += local_pe;
    }
    pe_total += global_pe;
}

void verlet_first_step(System& sys, double dt) {
    #pragma omp parallel for
    for (size_t i = 0; i < sys.x.size(); ++i) {
        double m = ATOM_TYPES[sys.type[i]].mass;
        double acc_factor = F_TO_ACC / m; 

        // Update Velocities
        sys.vx[i] += 0.5 * sys.fx[i] * acc_factor * dt;
        sys.vy[i] += 0.5 * sys.fy[i] * acc_factor * dt;
        sys.vz[i] += 0.5 * sys.fz[i] * acc_factor * dt;

        // Update Positions
        sys.x[i] += sys.vx[i] * dt;
        sys.y[i] += sys.vy[i] * dt;
        sys.z[i] += sys.vz[i] * dt;

        // Periodic Boundaries
        if (sys.x[i] < 0) sys.x[i] += L; 
        if (sys.x[i] >= L) sys.x[i] -= L;
        if (sys.y[i] < 0) sys.y[i] += L; 
        if (sys.y[i] >= L) sys.y[i] -= L;
        if (sys.z[i] < 0) sys.z[i] += L; 
        if (sys.z[i] >= L) sys.z[i] -= L;
    }
}

void verlet_second_step(System& sys, double dt) {
    #pragma omp parallel for
    for (size_t i = 0; i < sys.x.size(); ++i) {
        double m = ATOM_TYPES[sys.type[i]].mass;
        double acc_factor = F_TO_ACC / m;

        sys.vx[i] += 0.5 * acc_factor * sys.fx[i] * dt;
        sys.vy[i] += 0.5 * acc_factor * sys.fy[i] * dt;
        sys.vz[i] += 0.5 * acc_factor * sys.fz[i] * dt;
    }
}

void save_frame(int step, const System& sys) {
    std::string filename = "dumps/t" + std::to_string(step) + ".dump";
    std::ofstream data(filename);
    
    data << "ITEM: TIMESTEP\n" << step << "\n";
    data << "ITEM: NUMBER OF ATOMS\n" << sys.x.size() << "\n";
    data << "ITEM: BOX BOUNDS pp pp pp\n";
    data << "0.0 " << L << "\n0.0 " << L << "\n0.0 " << L << "\n";
    data << "ITEM: ATOMS id type x y z vx vy vz\n";

    for (size_t i = 0; i < sys.x.size(); ++i) {
        data << i << " " << sys.type[i] << " " 
             << sys.x[i] << " " << sys.y[i] << " " << sys.z[i] << " "
             << sys.vx[i] << " " << sys.vy[i] << " " << sys.vz[i] << "\n";
    }
    
    data.close();
}

double compute_kinetic_energy(const System& sys) {
    double ke = 0.0;
    for (size_t i = 0; i < sys.x.size(); ++i) {
        double v2 = sys.vx[i]*sys.vx[i] + 
                    sys.vy[i]*sys.vy[i] + 
                    sys.vz[i]*sys.vz[i];
        double m = ATOM_TYPES[sys.type[i]].mass;
        ke += 0.5 * m * v2;
    }
    double ke_cal = ke * KE_TO_KCAL;
    return ke_cal;
}

void rescale_velocities(System& sys, double target_temp) {
    double ke = compute_kinetic_energy(sys);
    double dof = 3.0 * (sys.x.size() - 1.0);
    double current_temp = 2.0 * ke / (dof * kb);
    double scale_factor = std::sqrt(target_temp / current_temp);

    for (size_t i = 0; i < sys.x.size(); ++i) {
        sys.vx[i] *= scale_factor;
        sys.vy[i] *= scale_factor;
        sys.vz[i] *= scale_factor;
    }
}

int main() {
    std::filesystem::create_directory("dumps");
    std::ofstream energy_file("energy.csv");
    energy_file << "Time,Kinetic,Potential,Total,Temperature\n";

    System sys;
    std::vector<Bond> bonds;
    std::vector<Angle> angles;
    init_water_box(sys, bonds, angles);

    CellList cells(L, VERLET_CUTOFF, sys.x.size());
    std::cout << "Starting Simulation with " << sys.x.size() << " particles.\n";
    
    double pe = 0.0;

    // Warmup phase
    double dt_warmup = 0.01;
    double max_force = 100.0;
    build_verlet_lists(sys, cells);
    for (int i = 0; i < 100; ++i) {
        verlet_first_step(sys, dt_warmup);
        std::fill(sys.fx.begin(), sys.fx.end(), 0.0);
        std::fill(sys.fy.begin(), sys.fy.end(), 0.0);
        std::fill(sys.fz.begin(), sys.fz.end(), 0.0);
        double pe_dummy = 0;
        compute_bond_forces(sys, bonds, pe_dummy);
        compute_angle_forces(sys, angles, pe_dummy);
        compute_forces(sys, pe_dummy);
        // Cap forces to avoid instability
        for (size_t i = 0; i < sys.x.size(); ++i) {
            if (sys.fx[i] > max_force) sys.fx[i] = max_force;
            if (sys.fx[i] < -max_force) sys.fx[i] = -max_force;
            if (sys.fy[i] > max_force) sys.fy[i] = max_force;
            if (sys.fy[i] < -max_force) sys.fy[i] = -max_force;
            if (sys.fz[i] > max_force) sys.fz[i] = max_force;
            if (sys.fz[i] < -max_force) sys.fz[i] = -max_force;
        }
        verlet_second_step(sys, dt_warmup);
        if (i % 50 == 0) max_force += 50.0;
    }
    std::cout << "Warmup done. Starting main run.\n";

    std::fill(sys.fx.begin(), sys.fx.end(), 0.0);
    std::fill(sys.fy.begin(), sys.fy.end(), 0.0);
    std::fill(sys.fz.begin(), sys.fz.end(), 0.0);
    rescale_velocities(sys, target_temp);

    auto start_time = std::chrono::high_resolution_clock::now();
    int steps = static_cast<int>(t_max / dt);
    for (int step = 0; step < steps; ++step) {
        verlet_first_step(sys, dt);
        std::fill(sys.fx.begin(), sys.fx.end(), 0.0);
        std::fill(sys.fy.begin(), sys.fy.end(), 0.0);
        std::fill(sys.fz.begin(), sys.fz.end(), 0.0);
        if (step % 20 == 0) {
            build_verlet_lists(sys, cells);
        }
        pe = 0.0;
        compute_bond_forces(sys, bonds, pe);
        compute_angle_forces(sys, angles, pe);
        compute_forces(sys, pe);
        verlet_second_step(sys, dt);

        if (step % 10 == 0) {

            if (ensemble == Ensemble::NVT) {
                rescale_velocities(sys, target_temp);
            }

            save_frame(step, sys);

            double ke = compute_kinetic_energy(sys);
            double total = ke + pe;

            double dof = 3.0 * (sys.x.size() - 1.0);
            double current_temp = 2.0 * ke / (dof * kb);

            energy_file << step * dt << "," << ke << "," << pe << "," << total << "," << current_temp << "\n";
            
            if (step % 100 == 0) {
                std::cout << "Step " << step << " / " << steps << "\r" << std::flush;
            }
        }

    }
    energy_file.close();

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    std::cout << "Simulation completed in " << duration.count() << " seconds.\n";
    
    return 0;
}