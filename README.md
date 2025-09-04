<img width="450" alt="bora_logo" src="https://github.com/user-attachments/assets/5b11c2dc-a78d-4fba-a0b4-c5c7d2f55158">

| **Documentation** | **Code style** | **Coverage** |
|:--------:|:----------------:|:----------------:|
| [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://cosmologicalemulators.github.io/EmulatorsTrainer.jl/dev) [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://cosmologicalemulators.github.io/EmulatorsTrainer.jl/stable) | [![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle) [![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac) | [![codecov](https://codecov.io/github/CosmologicalEmulators/EmulatorsTrainer.jl/graph/badge.svg?token=7IPK963YOW)](https://codecov.io/github/CosmologicalEmulators/EmulatorsTrainer.jl) |

# EmulatorsTrainer.jl

A Julia package for training and validating surrogate models (emulators) in the `CosmologicalEmulators` organization. This package provides utilities for dataset creation, training data management, and comprehensive validation of emulator performance.

## Features

- **Dataset Creation**: Generate training datasets using quasi-Monte Carlo sampling
- **Data Loading**: Efficient loading and preprocessing of training data from distributed simulations
- **Validation Tools**: Comprehensive validation utilities
- **Flexible API**: Intelligent defaults with customizable options

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

# Extract input and output arrays for training
X, y = extract_input_output_df(df, 3, 100)  # 3 inputs, 100 outputs
```

### Validating Emulators

The validation module provides a streamlined API with automatic dimension detection:

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

## Key Features of the Validation API

The validation module now features intelligent auto-detection:

- **Automatic sample counting**: No need to specify the number of validation samples
- **Automatic dimension inference**: Output dimensions are inferred from the first data file
- **Consistency validation**: Ensures all files have consistent dimensions
- **Flexible percentiles**: Customize which percentiles to compute (default: 68%, 95%, 99.7%)
- **Smart directory handling**: Correctly handles nested directory structures

### Example: Complete Validation Pipeline

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

- `create_training_dataset(n, lb, ub)`: Generate quasi-Monte Carlo samples
- `create_training_dict(matrix, idx, params)`: Create parameter dictionary
- `prepare_dataset_directory(path; force=false)`: Safely create dataset directory

### Data Loading

- `add_observable_df!(df, location, param_file, obs_file, get_tuple)`: Add single observation
- `load_df_directory!(df, dir, add_func)`: Load all observations from directory
- `extract_input_output_df(df, n_in, n_out)`: Extract training arrays

### Validation

- `evaluate_residuals(dir, dict_file, params, get_truth, get_pred; get_σ)`: Compute residuals
- `evaluate_sorted_residuals(dir, dict_file, params, get_truth, get_pred; get_σ, percentiles)`: Compute sorted residuals at specified percentiles
- `sort_residuals(residuals; percentiles)`: Sort and extract percentiles

## Roadmap to v1.0.0

| Step | Status | Comment |
|:------------ | :-------------|:-------------|
| Utils for training | ✅ | Complete with auto-detection |
| Distributed dataset creation | ✅ | Implemented and tested |
| Utils for validation | ✅ | Enhanced with intelligent defaults |
| Active learning | 🚧 | Work in progress |

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
