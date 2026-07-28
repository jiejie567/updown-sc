#include "scan_context.hpp"

#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;
namespace sc = fast_lio::scan_context;

int main(int argc, char **argv)
{
    if (argc < 2 || argc > 3 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")
    {
        std::cout
            << "Usage: scan_context_convert INPUT.scd [OUTPUT.scd]\n"
            << "Load a V1/V2/V3/V4/V5/V6/V7 Scan Context database and save it in V7 ground-height bitset format.\n"
            << "If OUTPUT is omitted, INPUT is replaced atomically after it is fully loaded.\n";
        return argc == 2 ? 0 : 2;
    }

    const fs::path input = fs::absolute(argv[1]).lexically_normal();
    const fs::path output =
        argc == 3 ? fs::absolute(argv[2]).lexically_normal() : input;
    if (!fs::exists(input))
    {
        std::cerr << "Input database does not exist: " << input << '\n';
        return 1;
    }

    std::error_code size_error;
    const std::uintmax_t input_size = fs::file_size(input, size_error);
    if (size_error)
    {
        std::cerr << "Failed to read input size: " << size_error.message() << '\n';
        return 1;
    }

    sc::Database database;
    std::string error;
    if (!database.load(input.string(), &error))
    {
        std::cerr << "Failed to load Scan Context database: " << error << '\n';
        return 1;
    }
    if (!database.save(output.string(), &error))
    {
        std::cerr << "Failed to save V7 Scan Context database: " << error << '\n';
        return 1;
    }

    const std::uintmax_t output_size = fs::file_size(output, size_error);
    if (size_error)
    {
        std::cerr << "Failed to read output size: " << size_error.message() << '\n';
        return 1;
    }

    std::cout << "Converted Scan Context database to V7 ground-relative heights and bitset masks:\n"
              << "  input: " << input << '\n'
              << "  output: " << output << '\n'
              << "  entries: " << database.size() << '\n'
              << "  bytes: " << input_size << " -> " << output_size << '\n';
    return 0;
}
