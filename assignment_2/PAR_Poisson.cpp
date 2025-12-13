/*
 * PAR_Poisson.cpp
 * Parallel 2D Poisson equation solver using MPI
 *
 * Refactored OOP version using modern C++17
 */

#include "poisson/Solver.hpp"
#include "poisson/Config.hpp"
#include "poisson/Timer.hpp"
#include "io/FileManager.hpp"
#include "io/AsciiWriter.hpp"
#include "io/HDF5Writer.hpp"
#include "io/IterationRecorder.hpp"
#include "mpi/MPIContext.hpp"

#include <iostream>
#include <iomanip>
#include <filesystem>
#include <cstring>

// Debug flag
constexpr bool DEBUG_MODE = false;

// Command line options
struct ProgramOptions
{
  std::filesystem::path configPath = "input.dat";
  std::filesystem::path outputPath = "/home/pascal/Documents/HighPerformance/assignment_2/output/solution";
  bool useHDF5 = false;
  bool recordIterations = false;
  int saveInterval = 20; // Save every iteration
};

void printUsage(const char *progName)
{
  std::cout << "Usage: " << progName << " [options] [config_file]\n"
            << "\nOptions:\n"
            << "  -h, --help              Show this help message\n"
            << "  -o, --output PATH       Output file base path (default: output)\n"
            << "  --hdf5                  Use HDF5 output format (requires HDF5 support)\n"
            << "  --record-iterations     Record intermediate iterations (implies --hdf5)\n"
            << "  --save-interval N       Save every N iterations (default: 100)\n"
            << "\nExamples:\n"
            << "  " << progName << " input.dat\n"
            << "  " << progName << " --hdf5 -o solution input.dat\n"
            << "  " << progName << " --record-iterations --save-interval 50 input.dat\n";
}

ProgramOptions parseArgs(int argc, char **argv)
{
  ProgramOptions opts;

  for (int i = 1; i < argc; ++i)
  {
    std::string arg = argv[i];

    if (arg == "-h" || arg == "--help")
    {
      printUsage(argv[0]);
      std::exit(0);
    }
    else if (arg == "-o" || arg == "--output")
    {
      if (i + 1 < argc)
      {
        opts.outputPath = argv[++i];
      }
    }
    else if (arg == "--hdf5")
    {
      opts.useHDF5 = true;
    }
    else if (arg == "--record-iterations")
    {
      opts.recordIterations = true;
      opts.useHDF5 = true; // Iteration recording requires HDF5
    }
    else if (arg == "--save-interval")
    {
      if (i + 1 < argc)
      {
        opts.saveInterval = std::stoi(argv[++i]);
      }
    }
    else if (arg[0] != '-')
    {
      opts.configPath = arg;
    }
  }

  return opts;
}

void printDebug(const std::string &msg)
{
  if (DEBUG_MODE)
  {
    std::cout << msg << std::endl;
  }
}

int main(int argc, char **argv)
{
  using namespace poisson;

  // Initialize MPI
  auto mpi = std::make_shared<poisson::MPIContext>(argc, argv);
  mpi->createCartesian({0, 0}, {false, false}); // Let MPI decide decomposition. non periodic boundaries


  try
  {
    // Parse command line options
    auto opts = parseArgs(argc, argv);
    printf("Process %d/%d starting...\n", mpi->rank(), mpi->size());
    // Determine input file path
    std::filesystem::path configPath = opts.configPath;
    if (!std::filesystem::exists(configPath))
    {
      // Try relative to executable
      auto execPath = std::filesystem::path(__FILE__).parent_path() / configPath;
      if (std::filesystem::exists(execPath))
      {
        configPath = execPath;
      }
    }

    printDebug("Loading config from: " + configPath.string());

    // Load configuration
    Config config;
    if (mpi->isRoot())
    {
      config = Config::fromFile(configPath);
      std::cout << "Grid size: " << config.nx() << " x " << config.ny() << std::endl;
      std::cout << "Precision goal: " << config.precisionGoal() << std::endl;
      std::cout << "Max iterations: " << config.maxIterations() << std::endl;
      std::cout << "Source points: " << config.sources().size() << std::endl;

      if (opts.useHDF5)
      {
        std::cout << "Output format: HDF5+XDMF" << std::endl;
        if (opts.recordIterations)
        {
          std::cout << "Iteration recording: enabled (interval=" << opts.saveInterval << ")" << std::endl;
        }
      }
      else
      {
        std::cout << "Output format: ASCII" << std::endl;
      }
    }

    mpi->broadcast(config);
    printDebug("Config received by rank " + std::to_string(mpi->rank()));
    printDebug("Config precision goal: " + std::to_string(config.precisionGoal()) + " on rank " + std::to_string(mpi->rank()));

    // Create solver (serial for now)
    Solver solver(config, mpi);
    std::string outputPathPerRank = opts.outputPath.string() + "_rank" + std::to_string(mpi->rank());

    // Set up iteration recorder if requested
    std::unique_ptr<io::IterationRecorder> recorder;
    if (opts.recordIterations)
    {
      recorder = std::make_unique<io::IterationRecorder>(
          config, outputPathPerRank
      );

      if (!recorder->enable(opts.saveInterval))
      {
        if (mpi->isRoot())
        {
          std::cerr << "Warning: Failed to enable iteration recording: "
                    << recorder->lastError() << std::endl;
          std::cerr << "Falling back to final-only output." << std::endl;
        }
        recorder.reset();
      }
      else
      {
        // Set the progress callback for iteration recording
        solver.setProgressCallback(recorder->createCallback());
      }
    }

    printDebug("Setup complete, starting solve...");

    // Solve with timing
    Timer timer;
    timer.start();
    

    auto result = solver.solve();
    
    timer.stop();

    // Finalize iteration recording
    if (recorder && recorder->isEnabled())
    {
      recorder->finalize();
    }


    // Print results (only from root in parallel case)
    std::cout << "Rank " << mpi->rank() << " completed solve using " << result.iterations << " iterations, taking " << timer.elapsedSeconds()*1000 << " milliseconds." << std::endl;
    mpi->barrier(); // Synchronize before printing final results
    if (mpi->isRoot())
    {
      std::cout << "\n=== Results ===" << std::endl;
      std::cout << "Number of iterations: " << result.iterations << std::endl;
      std::cout << "Converged: " << (result.converged ? "yes" : "no") << std::endl;
      std::cout << "Final residual: " << std::scientific << std::setprecision(6)
                << result.finalResidual << std::endl;
      std::cout << "Elapsed time: " << std::fixed << std::setprecision(6)
                << timer.elapsedSeconds() << " s" << std::endl;

      if (recorder && recorder->isEnabled())
      {
        std::cout << "Iterations recorded: " << recorder->recordedCount() << std::endl;
      }
    }

    // Write final output (skip if iterations were recorded - they include final state)
    if (!opts.recordIterations)
    {
      printDebug("Writing output...");

      if (opts.useHDF5)
      {
        // Use parallel I/O - all ranks write to a single file
        auto writer = io::FileManager::createParallel(
            io::OutputFormat::HDF5_XDMF, 
            mpi->rank(), 
            mpi->size(), 
            mpi->cartComm());
        
        if (!writer->write(solver.grid(), config, opts.outputPath))
        {
          if (mpi->isRoot())
          {
            std::cerr << "Warning: Failed to write HDF5 output: " << writer->lastError() << std::endl;
          }
        }
        else if (mpi->isRoot())
        {
          std::cout << "Output written to: " << opts.outputPath.string() << ".h5" << std::endl;
        }
      }
      else
      {
        auto writer = io::FileManager::create(io::OutputFormat::Ascii);
        std::string outputPathPerRank = opts.outputPath.string() + "_rank" + std::to_string(mpi->rank());
        if (!writer->write(solver.grid(), config, outputPathPerRank))
        {
          if (mpi->isRoot())
          {
            std::cerr << "Warning: Failed to write output: " << writer->lastError() << std::endl;
          }
        }
        else if (mpi->isRoot())
        {
          std::cout << "Output written to: " << outputPathPerRank << ".txt" << std::endl;
        }
      }
    }
    else if (mpi->isRoot())
    {
      std::cout << "Output written to: " << opts.outputPath.string() << ".h5 and .xdmf" << std::endl;
    }

    printDebug("Done!");
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
