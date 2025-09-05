"""
    create_training_dataset(n::Int, lb::Array, ub::Array)

Generate quasi-Monte Carlo samples using Latin Hypercube Sampling.

# Arguments
- `n::Int`: Number of samples to generate
- `lb::Array`: Lower bounds for each parameter
- `ub::Array`: Upper bounds for each parameter

# Returns
- `Matrix{Float64}`: Matrix of shape (n_params, n_samples) with parameter combinations

# Example
```julia
lb = [0.1, 0.5, 60.0]
ub = [0.5, 1.0, 80.0]
samples = create_training_dataset(1000, lb, ub)
```
"""
function create_training_dataset(n::Int, lb::AbstractArray{<:Real}, ub::AbstractArray{<:Real})
    # Input validation
    if n <= 0
        throw(ArgumentError("Number of samples must be positive, got n=$n"))
    end

    if length(lb) != length(ub)
        throw(ArgumentError("Lower and upper bounds must have same length. Got length(lb)=$(length(lb)), length(ub)=$(length(ub))"))
    end

    if isempty(lb)
        throw(ArgumentError("Bounds arrays cannot be empty"))
    end

    for i in 1:length(lb)
        if lb[i] >= ub[i]
            throw(ArgumentError("Lower bound must be less than upper bound for parameter $i. Got lb[$i]=$(lb[i]) >= ub[$i]=$(ub[i])"))
        end
        if !isfinite(lb[i]) || !isfinite(ub[i])
            throw(ArgumentError("Bounds must be finite. Got lb[$i]=$(lb[i]), ub[$i]=$(ub[i])"))
        end
    end

    # Convert to concrete Vector types for QuasiMonteCarlo.sample
    lb_vec = Float64[lb[i] for i in 1:length(lb)]
    ub_vec = Float64[ub[i] for i in 1:length(ub)]
    return QuasiMonteCarlo.sample(n, lb_vec, ub_vec, LatinHypercubeSample())
end

"""
    create_training_dict(training_matrix::Matrix, idx_comb::Int, params::Vector{String})

Create parameter dictionary for a specific sample from the training matrix.

# Arguments
- `training_matrix::Matrix`: Matrix of parameter combinations
- `idx_comb::Int`: Column index of the desired combination
- `params::Vector{String}`: Parameter names

# Returns
- `Dict{String, Float64}`: Dictionary mapping parameter names to values
"""
function create_training_dict(training_matrix::AbstractMatrix{T}, idx_comb::Int, params::AbstractVector{<:AbstractString}) where T<:Real
    return Dict([(value, training_matrix[idx_par, idx_comb])
    for (idx_par, value) in enumerate(params)])
end

"""
    prepare_dataset_directory(root_dir::String; force::Bool=false)

Safely create a dataset directory with existence checking and metadata tracking.

# Arguments
- `root_dir::String`: Path to the dataset directory
- `force::Bool=false`: If true, backs up existing directory; if false, throws error
"""
function prepare_dataset_directory(root_dir::String; force::Bool=false)
    if isdir(root_dir)
        if force
            # Create timestamped backup
            timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
            backup_dir = root_dir * "_backup_" * timestamp
            @warn "Directory exists. Creating backup before proceeding" existing=root_dir backup=backup_dir
            mv(root_dir, backup_dir)
            mkdir(root_dir)
        else
            error("""
                  Dataset directory already exists: $root_dir

                  This safety check prevents accidentally mixing datasets from different runs.

                  Options:
                  1. Choose a different directory name
                  2. Delete the existing directory manually if you're sure: rm("$root_dir", recursive=true)
                  3. Call with force=true to automatically backup existing data

                  Existing directory created: $(stat(root_dir).mtime)
                  """)
        end
    else
        mkdir(root_dir)
    end

    # Create metadata file to track dataset generation
    metadata = Dict(
        "created_at" => now(),
        "julia_version" => string(VERSION)
    )

    metadata_path = joinpath(root_dir, ".dataset_metadata.json")
    open(metadata_path, "w") do io
        JSON3.write(io, metadata)
    end

    return root_dir
end

"""
    validate_compute_inputs(training_matrix::AbstractMatrix, params::AbstractVector{String})

Validate inputs for compute_dataset functions.

# Arguments
- `training_matrix::AbstractMatrix`: Matrix of parameter combinations
- `params::AbstractVector{String}`: Parameter names

# Returns
- `(n_pars::Int, n_combs::Int)`: Tuple of matrix dimensions

# Throws
- `ArgumentError`: If inputs are invalid
"""
function validate_compute_inputs(training_matrix::AbstractMatrix, params::AbstractVector{<:AbstractString})
    n_pars, n_combs = size(training_matrix)

    # Input validation
    if n_pars != length(params)
        throw(ArgumentError("Number of parameters ($(length(params))) must match rows in training_matrix ($n_pars)"))
    end

    if n_combs == 0
        throw(ArgumentError("Training matrix must have at least one combination (column)"))
    end

    if isempty(params)
        throw(ArgumentError("Parameter names array cannot be empty"))
    end

    if any(isempty, params)
        throw(ArgumentError("Parameter names cannot be empty strings"))
    end

    if length(unique(params)) != length(params)
        throw(ArgumentError("Parameter names must be unique. Found duplicates in: $params"))
    end

    # Check for non-finite values in training matrix
    if any(!isfinite, training_matrix)
        throw(ArgumentError("Training matrix contains non-finite values (NaN or Inf)"))
    end

    return n_pars, n_combs
end

"""
    compute_dataset(training_matrix, params, root_dir, script_func, mode; force=false)

Compute dataset using specified parallelization mode with optional force override.

# Arguments
- `training_matrix::AbstractMatrix`: Matrix of parameter combinations
- `params::AbstractVector{String}`: Parameter names
- `root_dir::String`: Root directory for dataset
- `script_func::Function`: Function to compute data for each parameter combination
- `mode::Symbol`: Computation mode (:distributed, :threads, or :serial)
- `force::Bool=false`: Force overwrite of existing directory

# Modes
- `:distributed`: Use distributed computing across multiple processes
- `:threads`: Use multi-threading on shared memory
- `:serial`: Sequential execution (useful for debugging)
"""
function compute_dataset(training_matrix::AbstractMatrix, params::AbstractVector{<:AbstractString},
                        root_dir::String, script_func::Function,
                        mode::Symbol; force::Bool=false)

    # Validate inputs
    n_pars, n_combs = validate_compute_inputs(training_matrix, params)

    # Safely prepare directory
    actual_dir = prepare_dataset_directory(root_dir; force=force)

    # Dispatch based on mode
    if mode == :distributed
        @sync @distributed for idx in 1:n_combs
            train_dict = create_training_dict(training_matrix, idx, params)
            script_func(train_dict, actual_dir)
        end
    elseif mode == :threads
        Threads.@threads for idx in 1:n_combs
            train_dict = create_training_dict(training_matrix, idx, params)
            script_func(train_dict, actual_dir)
        end
    elseif mode == :serial
        for idx in 1:n_combs
            train_dict = create_training_dict(training_matrix, idx, params)
            script_func(train_dict, actual_dir)
        end
    else
        throw(ArgumentError("Invalid mode: $mode. Must be :distributed, :threads, or :serial"))
    end

    return actual_dir
end
