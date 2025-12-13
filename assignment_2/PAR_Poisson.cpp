/*
 * PAR_Poisson.cpp
 * Parallel 2D Poisson equation solver using MPI
 *
 * Refactored OOP version using modern C++17
 * Supports multiple solver methods: Gauss-Seidel, SOR, Conjugate Gradient
 */

#include "poisson/Solver.hpp"
#include "poisson/Config.hpp"
#include "poisson/Timer.hpp"
#include "poisson/Telemetry.hpp"
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
  bool useHDF5 = true;
  bool recordIterations = false;
  int saveInterval = 20;

  // Solver options (exercises 1.2.x)
  std::string solverMethod = "sor";           // gs, sor, cg
  double omega = 1.8;                        // SOR relaxation parameter
  int errorCheckInterval = 1;                // Check convergence every N iterations
  int sweepsPerExchange = 1;                 // Sweeps between border exchanges
  bool optimizedLoop = false;                // Use stride-2 loop optimization

  // Telemetry options
  bool verboseTiming = false;                // Print detailed timing summary
  std::filesystem::path timingOutputPath;    // CSV output path for timing data
  
  // Grid size override (for easy benchmarking)
  int gridNx = 100;                           // Override grid X size (-1 = use config file)
  int gridNy = 100;                           // Override grid Y size (-1 = use config file)
  int maxIterations = 100000;                    // Override max iterations (-1 = use config file)
  
  // Processor topology options
  int procX = 0;                              // Processor grid X dimension (0 = auto)
  int procY = 0;                              // Processor grid Y dimension (0 = auto)
};

void printUsage(const char *progName)
{
  std::cout << "Usage: " << progName << " [options] [config_file]\n"
            << "\nOutput Options:\n"
            << "  -h, --help              Show this help message\n"
            << "  -o, --output PATH       Output file base path (default: output/solution)\n"
            << "  --hdf5                  Use HDF5 output format (requires HDF5 support)\n"
            << "  --record-iterations     Record intermediate iterations (implies --hdf5)\n"
            << "  --save-interval N       Save every N iterations (default: 20)\n"
            << "\nSolver Options (for exercises 1.2.x):\n"
            << "  --solver METHOD         Solver method: gs (Gauss-Seidel), sor, cg (default: gs)\n"
            << "  --omega VALUE           SOR relaxation parameter (default: 1.0, typical: 1.90-1.99)\n"
            << "  --error-check-interval N  Check convergence every N iterations (default: 1)\n"
            << "  --sweeps-per-exchange N Number of sweeps between border exchanges (default: 1)\n"
            << "  --optimized-loop        Use stride-2 loop optimization (exercise 1.2.9)\n"
            << "\nGrid Override Options (for easy benchmarking):\n"
            << "  --nx N                  Override grid X size\n"
            << "  --ny N                  Override grid Y size\n"
            << "  --grid-size N           Set both nx and ny to N\n"
            << "  --max-iter N            Override maximum iterations\n"
            << "\nProcessor Topology Options:\n"
            << "  --px N                  Processor grid X dimension (0 = auto)\n"
            << "  --py N                  Processor grid Y dimension (0 = auto)\n"
            << "  --topology PxQ          Set processor topology (e.g., 4x1, 2x2)\n"
            << "\nTelemetry Options:\n"
            << "  --verbose-timing        Print detailed timing summary\n"
            << "  --timing-output PATH    Write per-iteration timing data to CSV file\n"
            << "\nExamples:\n"
            << "  # Basic run with Gauss-Seidel\n"
            << "  " << progName << " input.dat\n"
            << "\n  # Exercise 1.2.1-1.2.2: SOR with omega sweep\n"
            << "  " << progName << " --solver sor --omega 1.95 --verbose-timing input.dat\n"
            << "\n  # Exercise 1.2.3: Fixed iterations timing study\n"
            << "  " << progName << " --solver sor --omega 1.95 --grid-size 400 --max-iter 100 \\\n"
            << "                    --timing-output timing_400.csv input.dat\n"
            << "\n  # Exercise 1.2.7: Reduced error checking\n"
            << "  " << progName << " --solver sor --omega 1.95 --error-check-interval 10 input.dat\n"
            << "\n  # Exercise 1.2.8: Multiple sweeps per exchange\n"
            << "  " << progName << " --solver sor --omega 1.95 --sweeps-per-exchange 3 input.dat\n"
            << "\n  # Exercise 1.2.9: Optimized loop\n"
            << "  " << progName << " --solver sor --omega 1.95 --optimized-loop input.dat\n"
            << "\n  # Exercise 1.2.13: Conjugate Gradient\n"
            << "  " << progName << " --solver cg --verbose-timing input.dat\n";
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
      opts.useHDF5 = true;
    }
    else if (arg == "--save-interval")
    {
      if (i + 1 < argc)
      {
        opts.saveInterval = std::stoi(argv[++i]);
      }
    }
    // Solver options
    else if (arg == "--solver")
    {
      if (i + 1 < argc)
      {
        opts.solverMethod = argv[++i];
      }
    }
    else if (arg == "--omega")
    {
      if (i + 1 < argc)
      {
        opts.omega = std::stod(argv[++i]);
      }
    }
    else if (arg == "--error-check-interval")
    {
      if (i + 1 < argc)
      {
        opts.errorCheckInterval = std::stoi(argv[++i]);
      }
    }
    else if (arg == "--sweeps-per-exchange")
    {
      if (i + 1 < argc)
      {
        opts.sweepsPerExchange = std::stoi(argv[++i]);
      }
    }
    else if (arg == "--optimized-loop")
    {
      opts.optimizedLoop = true;
    }
    // Grid override options
    else if (arg == "--nx")
    {
      if (i + 1 < argc)
      {
        opts.gridNx = std::stoi(argv[++i]);
      }
    }
    else if (arg == "--ny")
    {
      if (i + 1 < argc)
      {
        opts.gridNy = std::stoi(argv[++i]);
      }
    }
    else if (arg == "--grid-size")
    {
      if (i + 1 < argc)
      {
        int size = std::stoi(argv[++i]);
        opts.gridNx = size;
        opts.gridNy = size;
      }
    }
    else if (arg == "--max-iter")
    {
      if (i + 1 < argc)
      {
        opts.maxIterations = std::stoi(argv[++i]);
      }
    }
    // Processor topology options
    else if (arg == "--px")
    {
      if (i + 1 < argc)
      {
        opts.procX = std::stoi(argv[++i]);
      }
    }
    else if (arg == "--py")
    {
      if (i + 1 < argc)
      {
        opts.procY = std::stoi(argv[++i]);
      }
    }
    else if (arg == "--topology")
    {
      if (i + 1 < argc)
      {
        std::string topo = argv[++i];
        size_t xPos = topo.find('x');
        if (xPos != std::string::npos)
        {
          opts.procX = std::stoi(topo.substr(0, xPos));
          opts.procY = std::stoi(topo.substr(xPos + 1));
        }
      }
    }
    // Telemetry options
    else if (arg == "--verbose-timing")
    {
      opts.verboseTiming = true;
    }
    else if (arg == "--timing-output")
    {
      if (i + 1 < argc)
      {
        opts.timingOutputPath = argv[++i];
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

  // Parse command line options early (before MPI topology setup)
  auto opts = parseArgs(argc, argv);

  // Initialize MPI
  auto mpi = std::make_shared<poisson::MPIContext>(argc, argv);
  
  // Set up Cartesian topology based on command line options
  // If procX and procY are 0, MPI will decide the decomposition
  mpi->createCartesian({opts.procX, opts.procY}, {false, false});


  try
  {
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
      
      // Apply command-line overrides
      if (opts.gridNx > 0 && opts.gridNy > 0)
      {
        config.setGridSize(opts.gridNx, opts.gridNy);
      }
      else if (opts.gridNx > 0)
      {
        config.setGridSize(opts.gridNx, config.ny());
      }
      else if (opts.gridNy > 0)
      {
        config.setGridSize(config.nx(), opts.gridNy);
      }
      
      if (opts.maxIterations > 0)
      {
        config.setMaxIterations(opts.maxIterations);
      }
      
      // Apply solver options
      config.setSolverMethod(Config::parseMethod(opts.solverMethod))
            .setOmega(opts.omega)
            .setErrorCheckInterval(opts.errorCheckInterval)
            .setSweepsPerExchange(opts.sweepsPerExchange)
            .setUseOptimizedLoop(opts.optimizedLoop)
            .setVerboseTiming(opts.verboseTiming)
            .setTimingOutputPath(opts.timingOutputPath);

      std::cout << "\n=== Configuration ===" << std::endl;
      std::cout << "Grid size: " << config.nx() << " x " << config.ny() << std::endl;
      std::cout << "Precision goal: " << config.precisionGoal() << std::endl;
      std::cout << "Max iterations: " << config.maxIterations() << std::endl;
      std::cout << "Source points: " << config.sources().size() << std::endl;
      std::cout << "Solver method: " << Config::methodToString(config.solverMethod()) << std::endl;
      
      if (config.solverMethod() == SolverMethod::SOR)
      {
        std::cout << "Omega (ω): " << config.omega() << std::endl;
      }
      
      std::cout << "Error check interval: " << config.errorCheckInterval() << std::endl;
      std::cout << "Sweeps per exchange: " << config.sweepsPerExchange() << std::endl;
      std::cout << "Optimized loop: " << (config.useOptimizedLoop() ? "yes" : "no") << std::endl;
      std::cout << "Processor topology: " << mpi->size() << " = " 
                << mpi->cartDims()[0] << " x " << mpi->cartDims()[1] << std::endl;

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

      // Print verbose timing summary if requested
      if (opts.verboseTiming && result.telemetry)
      {
        result.telemetry->printSummary(std::cout,
                                       config.nx(), config.ny(),
                                       mpi->size(),
                                       mpi->cartDims()[0], mpi->cartDims()[1],
                                       Config::methodToString(config.solverMethod()),
                                       config.omega());
      }

      // Write timing CSV if requested
      if (!opts.timingOutputPath.empty() && result.telemetry)
      {
        if (result.telemetry->writeCSV(opts.timingOutputPath))
        {
          std::cout << "Timing data written to: " << opts.timingOutputPath << std::endl;
        }
        else
        {
          std::cerr << "Warning: Failed to write timing CSV to " << opts.timingOutputPath << std::endl;
        }
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
