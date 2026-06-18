#include <exception>
#include <iostream>

#include "imagenn/version.hpp"

/// @file main.cpp
/// @brief Точка входа командной строки.

int main(int argc, char** argv) {
    try {
        std::cout << "ImageCNN " << imagenn::project_version() << "\n";
        if (argc > 1) {
            std::cout << "Arguments are not handled yet.\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
