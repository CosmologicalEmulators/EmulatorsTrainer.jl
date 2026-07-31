module EmulatorsTrainer

using DataFrames
using DataFrames: AbstractDataFrame
using ADTypes
using Dates
using Distributions
using Distributed
using HDF5
using JSON3
using Lux
using NPZ
using Optimisers
using QuasiMonteCarlo
using Random
using SimpleChains
using Zygote

# Export dataset creation functions
export create_training_dataset, create_training_dict
export prepare_dataset_directory, compute_dataset
export compute_dataset_hdf5, merge_hdf5_shards, load_hdf5_dataset

# Export data loading and training functions
export add_observable_df!, load_df_directory!
export extract_input_output_df, get_minmax_in, get_minmax_out
export maximin_df!, split_indices, splitdf, traintest_split, getdata
export DatasetLoadFailure, DatasetLoadReport

# Export neural-network training functions
export SimpleChainsTrainingConfig, SimpleChainsTrainingResult
export train_simplechains, save_training_result
export LuxTrainingConfig, LuxTrainingResult, train_lux

# Export validation functions
export evaluate_residuals, evaluate_sorted_residuals, sort_residuals

include("trainer.jl")
include("farmer.jl")
include("hdf5.jl")
include("validator.jl")
include("simplechains.jl")
include("lux.jl")

end # module EmulatorsTrainer
