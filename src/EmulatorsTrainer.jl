module EmulatorsTrainer

using DataFrames
using DataFrames: AbstractDataFrame
using Dates
using Distributions
using Distributed
using HDF5
using JSON3
using NPZ
using QuasiMonteCarlo
using Random
using SimpleChains

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

# Export validation functions
export evaluate_residuals, evaluate_sorted_residuals, sort_residuals

include("trainer.jl")
include("farmer.jl")
include("hdf5.jl")
include("validator.jl")
include("simplechains.jl")

end # module EmulatorsTrainer
