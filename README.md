<img width="450" alt="bora_logo" src="https://github.com/user-attachments/assets/5b11c2dc-a78d-4fba-a0b4-c5c7d2f55158">

| **Documentation** | **Code style** | **Coverage** |
|:--------:|:----------------:|:----------------:|
| [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://cosmologicalemulators.github.io/EmulatorsTrainer.jl/dev) [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://cosmologicalemulators.github.io/EmulatorsTrainer.jl/stable) | [![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle) [![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac) | [![codecov](https://codecov.io/github/CosmologicalEmulators/EmulatorsTrainer.jl/graph/badge.svg?token=7IPK963YOW)](https://codecov.io/github/CosmologicalEmulators/EmulatorsTrainer.jl) |

# EmulatorsTrainer.jl

A Julia package for training and validating surrogate models (emulators) in the `CosmologicalEmulators` organization. This package provides utilities for dataset creation, training data management, and comprehensive validation of emulator performance.

## Key Features

### 🚀 Smart Auto-Detection
- **Automatic dimension inference**: No manual feature counting required
- **Dataset size detection**: Automatically determines number of validation samples
- **Consistent validation**: Ensures data consistency across files

### ⚡ Flexible Parallelization
- **Distributed computing**: Scale across multiple processes
- **Multi-threading**: Efficient shared-memory parallelism
- **Serial execution**: Debugging and small dataset support

### 📊 Comprehensive Validation
- **Customizable percentiles**: Compute any error percentiles
- **Automatic residual computation**: With optional uncertainties
- **NaN handling**: Robust data loading with automatic filtering

## Installation

```julia
using Pkg
Pkg.add("EmulatorsTrainer")
```

## Quick Start

### Creating Training Datasets

Generate parameter samples for training using Latin Hypercube Sampling:

```julia
using EmulatorsTrainer

# Define parameter bounds
lower_bounds = [0.1, 0.5, 60.0]  # e.g., [Ωm_min, σ8_min, H0_min]
upper_bounds = [0.5, 1.0, 80.0]  # e.g., [Ωm_max, σ8_max, H0_max]

# Generate 1000 training samples
training_matrix = create_training_dataset(1000, lower_bounds, upper_bounds)

# Create parameter dictionary for a specific sample
params = ["omega_m", "sigma_8", "H0"]
param_dict = create_training_dict(training_matrix, 1, params)
```

### Computing Datasets

Generate datasets with flexible parallelization:

```julia
# Define computation function
function compute_simulation(params_dict, output_dir)
    # Your simulation code here
    # Save results to output_dir
end

# Choose parallelization mode
compute_dataset(training_matrix, params, "/data/simulations", compute_simulation, :distributed)
compute_dataset(training_matrix, params, "/data/simulations", compute_simulation, :threads)  
compute_dataset(training_matrix, params, "/data/simulations", compute_simulation, :serial)
```

### Loading Training Data

Load simulation outputs into a DataFrame for training:

```julia
using DataFrames

# Create empty DataFrame with appropriate columns
df = DataFrame()

# Define how to extract features from your data
function get_tuple(params, observable)
    return (
        omega_m = params["omega_m"],
        sigma_8 = params["sigma_8"],
        H0 = params["H0"],
        power_spectrum = observable
    )
end

# Load all data from a directory
add_obs_func = (df, root) -> add_observable_df!(
    df, root, "params.json", "power_spectrum.npy", get_tuple
)
load_df_directory!(df, "/path/to/simulations", add_obs_func)

# Extract arrays - dimensions detected automatically!
X, y = extract_input_output_df(df)
```

### Validating Emulators

Streamlined validation with full auto-detection:

```julia
# Define functions to get ground truth and emulator predictions
function get_ground_truth(location)
    # Load your ground truth data
    npzread(location * "/output.npy")
end

function get_emu_prediction(params)
    # Get prediction from your emulator
    emulator(params)
end

# Validate emulator performance - everything is auto-detected!
pars_array = ["omega_m", "sigma_8", "H0", "w0", "wa"]
sorted_residuals = evaluate_sorted_residuals(
    "/path/to/validation/data",  # Directory with validation samples
    "params.json",                # Parameter file name
    pars_array,                   # Parameters to extract
    get_ground_truth,             # Function to load truth
    get_emu_prediction            # Function to get prediction
)

# Optional: Specify custom percentiles
sorted_residuals = evaluate_sorted_residuals(
    validation_dir, "params.json", pars_array,
    get_ground_truth, get_emu_prediction;
    percentiles = [2.5, 16.0, 50.0, 84.0, 97.5]  # Custom percentiles
)
```

## What's New in v0.3.0

### 🎯 Complete Auto-Detection
All major functions now automatically detect dimensions:
- `extract_input_output_df(df)` - No manual feature counting
- `get_minmax_out(array)` - Auto-detects output dimensions
- `evaluate_residuals(...)` - Finds all validation samples automatically
- `sort_residuals(...)` - Infers matrix dimensions
- `getdata(df)` - Automatically splits train/test with dimension detection

### 🔧 Flexible Parallelization
New `compute_dataset` modes:
- `:distributed` - Multi-process computing
- `:threads` - Shared-memory parallelism  
- `:serial` - Sequential execution

### 📊 Enhanced Data Handling
- **NaN checking**: Both `add_observable_df!` methods now filter NaN values
- **Robust loading**: Automatic data validation during import
- **Smart defaults**: Sensible percentiles ([68.0, 95.0, 99.7]) for validation

## Complete Example

```julia
using EmulatorsTrainer
using NPZ

# Define parameter names
pars_array = ["ln10As", "ns", "H0", "ombh2", "omch2", "τ", "Mν", "w0", "wa"]

# Load your trained emulator
emulator = load_emulator("path/to/weights")

# Define data access functions
get_ground_truth(loc) = npzread(loc * "/Cl.npy")[2:3001]
get_emu_prediction(p) = emulator.predict(p)

# Run validation with automatic detection
results = evaluate_sorted_residuals(
    "/cosmology/validation/data",
    "cosmology_params.json",
    pars_array,
    get_ground_truth,
    get_emu_prediction;
    percentiles = [16.0, 50.0, 84.0]  # 1-sigma and median
)

# Results matrix has shape (n_percentiles, n_output_features)
println("Median relative error: ", results[2, :])
```

## API Reference

### Dataset Creation

- `create_training_dataset(n, lb, ub)`: Generate quasi-Monte Carlo samples using Latin Hypercube
- `create_training_dict(matrix, idx, params)`: Create parameter dictionary for a specific sample
- `prepare_dataset_directory(path; force=false)`: Safely create dataset directory with backup options
- `compute_dataset(matrix, params, dir, func, mode; force)`: Compute dataset with parallelization
  - Modes: `:distributed` (default), `:threads`, `:serial`

### Data Loading

- `add_observable_df!(df, location, param_file, obs_file, get_tuple)`: Add single observation with NaN checking
- `add_observable_df!(df, location, param_file, obs_file, first_idx, last_idx, get_tuple)`: Add observation slice with NaN checking
- `load_df_directory!(df, dir, add_func)`: Load all observations from directory
- `extract_input_output_df(df)`: Extract training arrays with automatic dimension detection
- `get_minmax_in(df, params)`: Get min/max values for input features
- `get_minmax_out(array_out)`: Get min/max values for output features with automatic detection
- `getdata(df)`: Split DataFrame into train/test sets with automatic dimension detection

### Validation

- `evaluate_residuals(dir, dict_file, params, get_truth, get_pred; get_σ)`: Compute residuals
- `evaluate_sorted_residuals(dir, dict_file, params, get_truth, get_pred; get_σ, percentiles)`: Compute sorted residuals at specified percentiles
- `sort_residuals(residuals; percentiles)`: Sort and extract percentiles

## Roadmap

| Feature | Status | Version |
|:--------|:-------|:--------|
| Auto-detection for all functions | ✅ | v0.3.0 |
| Flexible parallelization modes | ✅ | v0.3.0 |
| Smart validation utilities | ✅ | v0.3.0 |
| Active learning | 🚧 | Planned |

## Authors

- Marco Bonici, PostDoctoral Researcher at Waterloo Centre for Astrophysics
- Federico Bianchini, PostDoctoral researcher at Stanford

## License

MIT License - see LICENSE file for details.

## Citation

If you use EmulatorsTrainer.jl in your research, please cite:

```bibtex
@software{EmulatorsTrainer,
  author = {Bonici, Marco and Bianchini, Federico},
  title = {EmulatorsTrainer.jl: Training and Validation Tools for Cosmological Emulators},
  url = {https://github.com/CosmologicalEmulators/EmulatorsTrainer.jl},
  version = {0.3.0},
  year = {2024}
}
```
