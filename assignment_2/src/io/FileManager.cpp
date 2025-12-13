#include "io/FileManager.hpp"
#include "io/AsciiWriter.hpp"
#include "io/HDF5Writer.hpp"

namespace io
{

    std::unique_ptr<FileManager> FileManager::create(
        OutputFormat format,
        std::optional<int> mpiRank,
        std::optional<int> mpiSize)
    {
        switch (format)
        {
        case OutputFormat::Ascii:
            if (mpiRank.has_value())
            {
                return std::make_unique<AsciiWriter>(*mpiRank);
            }
            return std::make_unique<AsciiWriter>();

        case OutputFormat::HDF5_XDMF:
            if (mpiRank.has_value() && mpiSize.has_value())
            {
                return std::make_unique<HDF5Writer>(*mpiRank, *mpiSize);
            }
            else if (mpiRank.has_value())
            {
                return std::make_unique<HDF5Writer>(*mpiRank, 1);
            }
            return std::make_unique<HDF5Writer>();
        }

        // Should never reach here
        return std::make_unique<AsciiWriter>();
    }

    std::unique_ptr<FileManager> FileManager::createParallel(
        OutputFormat format,
        int mpiRank,
        int mpiSize,
        MPI_Comm comm)
    {
        switch (format)
        {
        case OutputFormat::Ascii:
            // ASCII doesn't support parallel I/O, fall back to per-rank files
            return std::make_unique<AsciiWriter>(mpiRank);

        case OutputFormat::HDF5_XDMF:
            return std::make_unique<HDF5Writer>(mpiRank, mpiSize, comm);
        }

        return std::make_unique<AsciiWriter>();
    }

} // namespace io
