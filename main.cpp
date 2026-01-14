#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <random>
#include <chrono>
#include <string>
#include <span>

enum class BoundaryCondition { PERIODIC, REFLECTIVE };
enum class Ensemble { NVE, NVT };

// Parameters
const int num_particles = 100; // Number of particles
const double L = 8.0; // Box length for periodic boundaries 
const double dt = 0.001; // Time step
const double t_max = 10.0; // Maximum simulation time
const BoundaryCondition BC = BoundaryCondition::PERIODIC; // Boundary condition type
const Ensemble ensemble = Ensemble::NVT; // Ensemble type
double kb = 1.0; // Boltzmann constant
double target_temp = 5.0; // Target temperature for NVT

// Force cutoff
const double CUTOFF = 3.5 * 1.2;
const double CUTOFF_SQ = CUTOFF * CUTOFF;

// Random number generator
std::mt19937& get_rng() {
    static unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    static std::mt19937 engine(seed);
    return engine;
}

// Atom type structure
struct AtomType {
    double mass;
    double sigma;
    double epsilon;
};

const std::vector<AtomType> ATOM_TYPES = {
    {1.0, 1.0, 1.0},   // Type 0: Small, Light
    {10.0, 1.2, 1.5}    // Type 1: Bigger (1.2x), Heavier (10.0x), Stickier (1.5x)
};

// Particle structure
struct Particle {
    std::array<double, 3> position;
    std::array<double, 3> velocity;
    std::array<double, 3> force;
    int type;

    Particle() : position{0,0,0}, velocity{0,0,0}, force{0,0,0}, type(0) {}
};

void init_lattice(std::vector<Particle>& particles, double density) {
    int n = particles.size();
    int particles_per_side = std::ceil(std::cbrt(n));
    double spacing = L / particles_per_side;

    int idx = 0;
    for (int x = 0; x < particles_per_side; ++x) {
        for (int y = 0; y < particles_per_side; ++y) {
            for (int z = 0; z < particles_per_side; ++z) {
                if (idx >= n) return;

                // Place on grid
                particles[idx].position[0] = x * spacing + (spacing * 0.5);
                particles[idx].position[1] = y * spacing + (spacing * 0.5);
                particles[idx].position[2] = z * spacing + (spacing * 0.5);

                static std::uniform_int_distribution<int> type_dist(0, ATOM_TYPES.size() - 1);
                particles[idx].type = type_dist(get_rng());

                // Random initial velocity
                static std::uniform_real_distribution<double> v_dist(-1.0, 1.0);
                auto& rng = get_rng();
                particles[idx].velocity = {v_dist(rng), v_dist(rng), v_dist(rng)};
                
                idx++;
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

    void build(const std::span<Particle> particles) {

        // Fill head with -1 (-1 indicates last particle in cell)
        std::fill(head.begin(), head.end(), -1);

        // Assign each particle to a cell
        for (int i = 0; i < particles.size(); ++i) {
            int cell = get_cell_index(particles[i].position[0], particles[i].position[1], particles[i].position[2]);

            next[i] = head[cell];
            head[cell] = i;
        }
    }


};

double compute_forces(std::span<Particle> particles, CellList& cells) {
    // Reset Forces
    for (auto& p : particles) {
        p.force = {0.0, 0.0, 0.0};
    }

    // Build cell list with current positions
    cells.build(particles);

    // Get grid dimension
    int dim = cells.grid_dim;

    double total_potential = 0.0;
    
    // Loop over all cells
    for (int cx = 0; cx < dim; ++cx) {
        for (int cy = 0; cy < dim; ++cy) {
            for (int cz = 0; cz < dim; ++cz) {
                
                // 1D index of current cell
                int cell_index = cx + cy * dim + cz * dim * dim;

                // Loop over particles in this cell starting at head
                int i = cells.head[cell_index];
                while (i != -1) { 
                    
                    // Check a every cell around current cell (3x3x3 block)
                    for (int nx = cx - 1; nx <= cx + 1; ++nx) {
                        for (int ny = cy - 1; ny <= cy + 1; ++ny) {
                            for (int nz = cz - 1; nz <= cz + 1; ++nz) {
                                
                                // Periodic wrap for cell indices
                                int wrapped_nx = (nx + dim) % dim;
                                int wrapped_ny = (ny + dim) % dim;
                                int wrapped_nz = (nz + dim) % dim;
                                
                                // 1D index of neighbor cell
                                int neighbor_cell_index = wrapped_nx + wrapped_ny * dim + wrapped_nz * dim * dim;

                                // Loop over particles in neighbor cell
                                int j = cells.head[neighbor_cell_index];
                                while (j != -1) {
                                    
                                    // Avoid double counting and self-interaction
                                    if (i < j) {
                                        Particle& p1 = particles[i];
                                        Particle& p2 = particles[j];

                                        const AtomType& type1 = ATOM_TYPES[p1.type];
                                        const AtomType& type2 = ATOM_TYPES[p2.type];

                                        double sigma = 0.5 * (type1.sigma + type2.sigma);
                                        double epsilon = std::sqrt(type1.epsilon * type2.epsilon);

                                        double dx = p1.position[0] - p2.position[0];
                                        double dy = p1.position[1] - p2.position[1];
                                        double dz = p1.position[2] - p2.position[2];

                                        // Periodic boundary
                                        if (dx > L/2) dx -= L; 
                                        if (dx < -L/2) dx += L;
                                        if (dy > L/2) dy -= L; 
                                        if (dy < -L/2) dy += L;
                                        if (dz > L/2) dz -= L; 
                                        if (dz < -L/2) dz += L;

                                        double r2 = dx*dx + dy*dy + dz*dz;

                                        if (r2 < CUTOFF_SQ && r2 > 0.0) {
                                            double inv_r2 = 1.0 / r2;
                                            double sig_inv_r2 = sigma * sigma * inv_r2;
                                            double sig_inv_r6 = sig_inv_r2 * sig_inv_r2 * sig_inv_r2;
                                            double sig_inv_r12 = sig_inv_r6 * sig_inv_r6;
                                            double factor = (24.0 * epsilon * inv_r2) * (2.0 * sig_inv_r12 - sig_inv_r6);

                                            double fx = factor * dx;
                                            double fy = factor * dy;
                                            double fz = factor * dz;

                                            p1.force[0] += fx; p1.force[1] += fy; p1.force[2] += fz;
                                            p2.force[0] -= fx; p2.force[1] -= fy; p2.force[2] -= fz;

                                            double u_r = 4.0 * epsilon * (sig_inv_r12 - sig_inv_r6);

                                            // Pre-calculate inverse cutoff terms based on sigma
                                            double inv_rc2 = 1.0 / CUTOFF_SQ;
                                            double sig_rc2 = sigma * sigma * inv_rc2;
                                            double sig_rc6 = sig_rc2 * sig_rc2 * sig_rc2;
                                            double sig_rc12 = sig_rc6 * sig_rc6;
                                            double u_shift = 4.0 * epsilon * (sig_rc12 - sig_rc6);

                                            // Add shifted potential
                                            total_potential += (u_r - u_shift);
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
    return total_potential;
}

// Periodic boundary condition
void apply_periodic_bc(Particle& p) {
    for (int i = 0; i < 3; ++i) {
        if (p.position[i] < 0.0) {
            p.position[i] += L;
        } else if (p.position[i] >= L) {
            p.position[i] -= L;
        }
    }
}

// Reflective boundary condition
void apply_reflective_bc(Particle& p) {
    for (int i = 0; i < 3; ++i) {
        if (p.position[i] < 0.0) {
            p.position[i] = -p.position[i];
            p.velocity[i] = -p.velocity[i];
        } else if (p.position[i] > L) {
            p.position[i] = 2 * L - p.position[i];
            p.velocity[i] = -p.velocity[i];
        }
    }
}

void verlet_first_step(std::span<Particle> particles, double dt) {
    for (auto& p : particles) {
        double m = ATOM_TYPES[p.type].mass;
        std::array<double, 3> acceleration = {p.force[0]/m, p.force[1]/m, p.force[2]/m};

        for (int i = 0; i < 3; ++i) {
            p.velocity[i] += 0.5 * acceleration[i] * dt;
            p.position[i] += p.velocity[i] * dt;
        }

        if (BC == BoundaryCondition::PERIODIC) {
            apply_periodic_bc(p);
        }
        else {
            apply_reflective_bc(p);
        }
    }
}

void verlet_second_step(std::span<Particle> particles, double dt) {
    for (auto& p : particles) {
        double m = ATOM_TYPES[p.type].mass;
        std::array<double, 3> acceleration = {p.force[0]/m, p.force[1]/m, p.force[2]/m};

        for (int i = 0; i < 3; ++i) {
            p.velocity[i] += 0.5 * acceleration[i] * dt;
        }
    }
}

void save_frame(int step, const std::vector<Particle>& particles) {
    std::string filename = "dumps/t" + std::to_string(step) + ".dump";
    std::ofstream data(filename);
    
    data << "ITEM: TIMESTEP\n" << step << "\n";
    data << "ITEM: NUMBER OF ATOMS\n" << particles.size() << "\n";
    data << "ITEM: BOX BOUNDS pp pp pp\n";
    data << "0.0 " << L << "\n0.0 " << L << "\n0.0 " << L << "\n";
    data << "ITEM: ATOMS id type x y z vx vy vz\n";

    for (size_t i = 0; i < particles.size(); ++i) {
        const auto& p = particles[i];
        data << i << " " << p.type << " " 
             << p.position[0] << " " << p.position[1] << " " << p.position[2] << " "
             << p.velocity[0] << " " << p.velocity[1] << " " << p.velocity[2] << "\n";
    }
    
    data.close();
}

double compute_kinetic_energy(const std::vector<Particle>& particles) {
    double ke = 0.0;
    for (const auto& p : particles) {
        double v2 = p.velocity[0]*p.velocity[0] + 
                    p.velocity[1]*p.velocity[1] + 
                    p.velocity[2]*p.velocity[2];
        double m = ATOM_TYPES[p.type].mass;
        ke += 0.5 * m * v2;
    }
    return ke;
}

void rescale_velocities(std::vector<Particle>& particles, double target_temp) {
    double ke = compute_kinetic_energy(particles);
    double dof = 3.0 * (particles.size() - 1.0);
    double current_temp = 2.0 * ke / (dof * kb);
    double scale_factor = std::sqrt(target_temp / current_temp);

    for (auto& p : particles) {
        for (int i = 0; i < 3; ++i) {
            p.velocity[i] *= scale_factor;
        }
    }
}

int main() {
    std::filesystem::create_directory("dumps");

    std::ofstream energy_file("energy.csv");
    energy_file << "Time,Kinetic,Potential,Total,Temperature\n";

    std::vector<Particle> particles(num_particles);
    init_lattice(particles, 0.8);
    CellList cells(L, CUTOFF, num_particles);
    std::cout << "Starting Simulation with " << particles.size() << " particles.\n";
    
    double pe = compute_forces(particles, cells);

    int steps = static_cast<int>(t_max / dt);
    for (int step = 0; step < steps; ++step) {
        verlet_first_step(particles, dt);
        pe = compute_forces(particles, cells);
        verlet_second_step(particles, dt);


        if (step % 10 == 0) {

            if (ensemble == Ensemble::NVT) {
                rescale_velocities(particles, target_temp);
            }

            save_frame(step, particles);

            double ke = compute_kinetic_energy(particles);
            double total = ke + pe;

            double dof = 3.0 * (particles.size() - 1.0);
            double current_temp = 2.0 * ke / (dof * kb);

            energy_file << step * dt << "," << ke << "," << pe << "," << total << "," << current_temp << "\n";

            if (step % 100 == 0) {
                std::cout << "Step " << step << " / " << steps << "\r" << std::flush;
            }
        }

    }
    energy_file.close();
    std::cout << "\nDone.\n";
    return 0;
}