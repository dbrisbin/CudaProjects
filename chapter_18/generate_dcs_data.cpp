/// @file generate_fhd_data.cpp
/// @brief This is a simple program that generates a file with random data
/// for the FHD algorithm to use.

#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

int main(int argc, char* argv[])
{
    std::vector<std::string> args(argv, argv + argc);
    if (args.size() != 6)
    {
        std::cout << "Usage: " << argv[0] << " <output_file> <N> <X> <Y> <Z>" << std::endl;
        return 1;
    }

    std::ofstream file_ptr(args[1]);
    if (!file_ptr.is_open())
    {
        std::cout << "Could not open file " << args[1] << "." << std::endl;
        return 1;
    }

    const int N = std::stoi(args[2]);
    const int X = std::stoi(args[3]);
    const int Y = std::stoi(args[4]);
    const int Z = std::stoi(args[5]);

    file_ptr << X << " " << Y << " " << Z << std::endl;
    file_ptr << N << std::endl;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 5.0);

    // Write N atoms with random positions and charges
    for (int i{0}; i < N; ++i)
    {
        for (int j{0}; j < 4; ++j)
        {
            file_ptr << dis(gen) << " ";
        }
        file_ptr << std::endl;
    }

    return 0;
}
