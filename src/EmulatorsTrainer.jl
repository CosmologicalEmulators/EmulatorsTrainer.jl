module EmulatorsTrainer

using DataFrames
using DataFrames: AbstractDataFrame
using Dates
using Distributions
using Distributed
using JSON3
using NPZ
using QuasiMonteCarlo
using Random

# Export dataset creation functions
export create_training_dataset, create_training_dict
export prepare_dataset_directory, compute_dataset

# Export data loading and training functions
export add_observable_df!, load_df_directory!
export extract_input_output_df, get_minmax_in, get_minmax_out
export maximin_df!, splitdf, traintest_split, getdata

# Export validation functions
export evaluate_residuals, evaluate_sorted_residuals, sort_residuals

include("trainer.jl")
include("farmer.jl")
include("validator.jl")

end # module EmulatorsTrainer
