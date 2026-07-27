#include "scan_context.hpp"

#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace sc = fast_lio::scan_context;

namespace
{

bool readIndices(const std::string &path, std::vector<std::size_t> *indices)
{
    std::ifstream input(path);
    if (!input)
    {
        std::cerr << "Failed to open index file: " << path << '\n';
        return false;
    }
    long long value = -1;
    while (input >> value)
    {
        if (value < 0)
        {
            std::cerr << "Negative index in " << path << ": " << value << '\n';
            return false;
        }
        indices->push_back(static_cast<std::size_t>(value));
    }
    if (!input.eof() || indices->empty())
    {
        std::cerr << "Invalid or empty index file: " << path << '\n';
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        std::cerr << "Usage: scan_context_subset INPUT.scd INDICES.txt OUTPUT.scd\n";
        return 2;
    }

    sc::Database input;
    std::string error;
    if (!input.load(argv[1], &error))
    {
        std::cerr << "Failed to load input database: " << error << '\n';
        return 1;
    }

    std::vector<std::size_t> indices;
    if (!readIndices(argv[2], &indices))
    {
        return 1;
    }

    sc::Database output(input.config());
    const auto &entries = input.entries();
    std::size_t previous = std::numeric_limits<std::size_t>::max();
    for (const std::size_t index : indices)
    {
        if (index >= entries.size())
        {
            std::cerr << "Index " << index << " exceeds database size "
                      << entries.size() << '\n';
            return 1;
        }
        if (previous != std::numeric_limits<std::size_t>::max() &&
            index <= previous)
        {
            std::cerr << "Indices must be strictly increasing\n";
            return 1;
        }
        const auto &entry = entries[index];
        output.addEntry(entry.stamp, entry.pose, entry.descriptor);
        previous = index;
    }

    if (!output.save(argv[3], &error))
    {
        std::cerr << "Failed to save output database: " << error << '\n';
        return 1;
    }
    std::cout << "Wrote " << output.size() << '/' << input.size()
              << " Scan Context entries to " << argv[3] << '\n';
    return 0;
}
