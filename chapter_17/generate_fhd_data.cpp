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
    if (args.size() != 4)
    {
        std::cout << "Usage: " << argv[0] << " <output_file> <M> <N>" << std::endl;
        return 1;
    }

    std::ofstream file_ptr(args[1]);
    if (!file_ptr.is_open())
    {
        std::cout << "Could not open file " << args[1] << "." << std::endl;
        return 1;
    }

    const int M = std::stoi(args[2]);
    const int N = std::stoi(args[3]);

    file_ptr << M << " " << N << std::endl;
    constexpr int kNumVarsWithMElements = 7;
    constexpr int kNumVarsWithNElements = 3;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    for (int i{0}; i < kNumVarsWithMElements; ++i)
    {
        for (int j{0}; j < M; ++j)
        {
            file_ptr << dis(gen) << " ";
        }
        file_ptr << std::endl;
    }

    for (int i{0}; i < kNumVarsWithNElements; ++i)
    {
        for (int j{0}; j < N; ++j)
        {
            file_ptr << dis(gen) << " ";
        }
        file_ptr << std::endl;
    }

    return 0;
}
