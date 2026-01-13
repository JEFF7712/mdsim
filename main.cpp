#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <random>
#include <chrono>
#include <string>
#include <span>

// Parameters
const int num_particles = 100; // Number of particles
const double L = 100.0; // Box length for periodic boundaries 
const double dt = 0.1; // Time step
const double t_max = 100.0; // Maximum simulation time
const double m = 1.0; // Particle mass
enum class BoundaryCondition { PERIODIC, REFLECTIVE };
const BoundaryCondition BC = BoundaryCondition::PERIODIC; // Boundary condition type

// Lenenard Jones potential parameters
const double epsilon = 1.0;
const double sigma = 1.0;
const double CUTOFF = 2.5 * sigma; // Ignore atoms too far away
const double CUTOFF_SQ = CUTOFF * CUTOFF;

// Random number generator
std::mt19937& get_rng() {
    static unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    static std::mt19937 engine(seed);
    return engine;
}

// Particle structure
struct Particle {
    std::array<double, 3> position;
    std::array<double, 3> velocity;
    std::array<double, 3> force;

    Particle() {

        // Initialize position and velocity with random values
        static std::uniform_real_distribution<double> pos_dist(0.0, L);
        static std::uniform_real_distribution<double> vel_dist(-10.0, 10.0);
        auto& rng = get_rng();

        position = {pos_dist(rng), pos_dist(rng), pos_dist(rng)};
        velocity = {vel_dist(rng), vel_dist(rng), vel_dist(rng)};

        // Initialize force to zero
        force = {0.0, 0.0, 0.0};

    }
};

void compute_forces(std::span<Particle> particles) {
    // Reset forces
    for (auto& p : particles) {
        p.force = {0.0, 0.0, 0.0};
    }

    for (size_t i = 0; i < particles.size(); ++i) {
        for (size_t j = i + 1; j < particles.size(); ++j) {

            Particle& p1 = particles[i];
            Particle& p2 = particles[j];

            // Distances
            double dx = p1.position[0] - p2.position[0];
            double dy = p1.position[1] - p2.position[1];
            double dz = p1.position[2] - p2.position[2];

            // Periodic boundary conditions
            if (dx > L/2) dx -= L;
            if (dx < -L/2) dx += L;
            if (dy > L/2) dy -= L;
            if (dy < -L/2) dy += L;
            if (dz > L/2) dz -= L;
            if (dz < -L/2) dz += L;

            double r2 = dx*dx + dy*dy + dz*dz;

            if (r2 < CUTOFF_SQ) {
                double inv_r2 = 1.0 / r2;
                double inv_r6 = inv_r2 * inv_r2 * inv_r2;
                double inv_r12 = inv_r6 * inv_r6;

                // Simplified assuming sigma=1 for now
                double factor = (24.0 * epsilon * inv_r2) * (2.0 * inv_r12 - inv_r6);

                // Distribute force to vectors
                double fx = factor * dx;
                double fy = factor * dy;
                double fz = factor * dz;

                // Apply forces
                p1.force[0] += fx;
                p1.force[1] += fy;
                p1.force[2] += fz;

                p2.force[0] -= fx;
                p2.force[1] -= fy;
                p2.force[2] -= fz;
            }
        }
    }
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
    data << "ITEM: ATOMS id x y z vx vy vz\n";

    for (size_t i = 0; i < particles.size(); ++i) {
        const auto& p = particles[i];
        data << i << " " 
             << p.position[0] << " " << p.position[1] << " " << p.position[2] << " "
             << p.velocity[0] << " " << p.velocity[1] << " " << p.velocity[2] << "\n";
    }
    
    data.close();
}


int main() {
    std::filesystem::create_directory("dumps");

    std::vector<Particle> particles(num_particles);
    std::cout << "Starting Simulation with " << particles.size() << " particles.\n";

    compute_forces(particles);
    int steps = static_cast<int>(t_max / dt);
    
    for (int step = 0; step < steps; ++step) {
        verlet_first_step(particles, dt);
        compute_forces(particles);
        verlet_second_step(particles, dt);
        save_frame(step, particles);
    }
    
    std::cout << "\nDone.\n";
    return 0;
}