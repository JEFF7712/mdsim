#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <cmath>
#include <random>
#include <chrono>
#include <string>

std::mt19937& get_rng() {
    static unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    static std::mt19937 engine(seed);
    return engine;
}

struct Particle {
    std::array<double, 3> position;
    std::array<double, 3> velocity;

    Particle() {

        // Initialize position and velocity with random values
        static std::uniform_real_distribution<double> pos_dist(0.0, 100.0);
        position[0] = pos_dist(get_rng());
        position[1] = pos_dist(get_rng());
        position[2] = pos_dist(get_rng());

        static std::uniform_real_distribution<double> vel_dist(0.0, 50.0);
        velocity[0] = vel_dist(get_rng());
        velocity[1] = vel_dist(get_rng());
        velocity[2] = vel_dist(get_rng());

    }
};

const double L = 100.0; // Box length for periodic boundaries
const std::string BC = "reflective"; // Boundary condition type

Particle update_particle_periodic(const Particle& p, double dt) {
    Particle out = p;
    out.position[0] = std::fmod(p.position[0] + p.velocity[0] * dt, L);
    out.position[1] = std::fmod(p.position[1] + p.velocity[1] * dt, L);
    out.position[2] = std::fmod(p.position[2] + p.velocity[2] * dt, L);
    return out;
}

Particle update_particle_reflective(const Particle& p, double dt) {
    Particle out = p;
    for (int i = 0; i < 3; ++i) {
        out.position[i] += p.velocity[i] * dt;
        if (out.position[i] < 0.0) {
            out.position[i] = -out.position[i];
            out.velocity[i] = -out.velocity[i];
        } else if (out.position[i] > L) {
            out.position[i] = 2 * L - out.position[i];
            out.velocity[i] = -out.velocity[i];
        }
    }
    return out;
}

Particle update_particle(const Particle& p, double dt) {
    Particle out = p;
    if (BC == "periodic") {
        out = update_particle_periodic(p, dt);
    } else if (BC == "reflective") {
        out = update_particle_reflective(p, dt);
    }
    return out;
}

int main() {
    
    std::vector<Particle> particles(5);
    std::cout << "Initialized " << particles.size() << " particles.\n";

    double dt = 0.1; // Time step
    double t_max = 10.0; // Maximum simulation time

    // Create output directory for dump files
    std::filesystem::create_directory("dumps");

    int timestep_index = 0;
    for (double t = 0; t < t_max; t += dt) {
        // Create dump file for this timestep
        std::string filename = "dumps/t" + std::to_string(timestep_index) + ".dump";
        std::ofstream data(filename);
        
        data << "ITEM: TIMESTEP\n";
        data << timestep_index << "\n";
        data << "ITEM: NUMBER OF ATOMS\n";
        data << particles.size() << "\n";
        data << "ITEM: BOX BOUNDS ff ff ff\n";
        data << "0.0 " << L << "\n";
        data << "0.0 " << L << "\n";
        data << "0.0 " << L << "\n";
        data << "ITEM: ATOMS id x y z\n";

        for (size_t i = 0; i < particles.size(); ++i) {
            data << i << " "
                 << particles[i].position[0] << " "
                 << particles[i].position[1] << " "
                 << particles[i].position[2] << "\n";

            particles[i] = update_particle(particles[i], dt);
        }
        
        data.close();
        timestep_index++;
    }

    std::cout << "Simulation complete. Data saved to dumps/ folder\n";

    return 0;

}
