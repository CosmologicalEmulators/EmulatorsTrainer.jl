function create_training_dataset(n::Int, lb::Array, ub::Array)
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
    
    return QuasiMonteCarlo.sample(n, lb, ub, LatinHypercubeSample())
end

function create_training_dict(training_matrix::Matrix, idx_comb::Int, params::Array{String})
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

function compute_dataset(training_matrix::Matrix, params::Array{String}, root_dir::String, script_func::Function; force::Bool=false)
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
    
    # Safely prepare directory
    actual_dir = prepare_dataset_directory(root_dir; force=force)
    
    @sync @distributed for idx in 1:n_combs
        train_dict = create_training_dict(training_matrix, idx, params)
        script_func(train_dict, actual_dir)
    end
    
    return actual_dir
end

# Keep original function signature for backward compatibility
function compute_dataset(training_matrix::Matrix, params::Array{String}, root_dir::String, script_func::Function)
    compute_dataset(training_matrix, params, root_dir, script_func; force=false)
end
