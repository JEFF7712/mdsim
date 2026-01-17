#pragma once
#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <iostream>

class Config {
private:
    std::map<std::string, std::string> data;

public:
    void load(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open config file \n";
            exit(1);
        }

        std::string line;
        while (std::getline(file, line)) {
            // Remove comments (lines starting with #)
            if (line.empty() || line[0] == '#') continue;

            std::istringstream iss(line);
            std::string key, eq, val;
            if (iss >> key >> eq >> val) {
                if (eq == "=") {
                    data[key] = val;
                }
            }
        }
    }

    double get_double(const std::string& key, double default_val) {
        if (data.count(key)) return std::stod(data.at(key));
        return default_val;
    }

    int get_int(const std::string& key, int default_val) {
        if (data.count(key)) return std::stoi(data.at(key));
        return default_val;
    }

    std::string get_string(const std::string& key, const std::string& default_val) {
        if (data.count(key)) return data.at(key);
        return default_val;
    }
};