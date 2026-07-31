"""A sample directory and the reason it was skipped while loading a dataset."""
struct DatasetLoadFailure
    directory::String
    message::String
end

"""Counts and failure details returned by `load_df_directory!`."""
struct DatasetLoadReport
    discovered::Int
    loaded::Int
    skipped::Int
    failures::Vector{DatasetLoadFailure}
end

_dataset_path(location, filename) = joinpath(location, lstrip(filename, ('/', '\\')))

function _read_observable(location, param_file, observable_file; first_idx=nothing,
    last_idx=nothing, validate_observable=nothing)
    parameter_path = _dataset_path(location, param_file)
    observable_path = _dataset_path(location, observable_file)
    isfile(parameter_path) || throw(ArgumentError("Parameter file does not exist: $parameter_path"))
    isfile(observable_path) || throw(ArgumentError("Observable file does not exist: $observable_path"))

    cosmo_pars = JSON3.read(read(parameter_path, String))
    observable = npzread(observable_path)
    if first_idx !== nothing
        checkbounds(Bool, observable, first_idx:last_idx) ||
            throw(ArgumentError("Observable slice $first_idx:$last_idx is outside $(axes(observable, 1))"))
        observable = observable[first_idx:last_idx]
    end
    isempty(observable) && throw(ArgumentError("Observable is empty: $observable_path"))
    all(isfinite, observable) || throw(ArgumentError("Observable contains NaN or Inf: $observable_path"))
    if validate_observable !== nothing
        valid = validate_observable(cosmo_pars, observable, location)
        valid === false && throw(ArgumentError("Observable failed domain-specific validation: $observable_path"))
    end
    return cosmo_pars, observable
end

"""
    add_observable_df!(df::DataFrame, location::String, param_file::String,
                      observable_file::String, first_idx::Int, last_idx::Int, get_tuple::Function)

Add an observation slice to a DataFrame after checking file existence, bounds,
and finite values. An optional callback performs domain-specific validation.

# Arguments
- `df::DataFrame`: Target DataFrame
- `location::String`: Directory containing files
- `param_file::String`: JSON file with parameters
- `observable_file::String`: NPY file with observables
- `first_idx::Int`: Start index for slice
- `last_idx::Int`: End index for slice
- `get_tuple::Function`: Function to process (params, observable) into tuple
"""
function add_observable_df!(df::DataFrames.DataFrame, location::String, param_file::String,
    observable_file::String, first_idx::Int, last_idx::Int, get_tuple::Function;
    validate_observable=nothing)
    cosmo_pars, observable = _read_observable(
        location, param_file, observable_file;
        first_idx, last_idx, validate_observable,
    )
    processed_observable = get_tuple(cosmo_pars, observable)
    push!(df, processed_observable)
    return true
end

"""
    add_observable_df!(df::DataFrame, location::String, param_file::String,
                      observable_file::String, get_tuple::Function)

Add a complete observation to a DataFrame after generic integrity validation.

# Arguments
- `df::DataFrame`: Target DataFrame
- `location::String`: Directory containing files
- `param_file::String`: JSON file with parameters
- `observable_file::String`: NPY file with observables
- `get_tuple::Function`: Function to process (params, observable) into tuple
"""
function add_observable_df!(df::DataFrames.DataFrame, location::String, param_file::String,
    observable_file::String, get_tuple::Function; validate_observable=nothing)
    cosmo_pars, observable = _read_observable(
        location, param_file, observable_file; validate_observable,
    )
    processed_observable = get_tuple(cosmo_pars, observable)
    push!(df, processed_observable)
    return true
end

"""
    load_df_directory!(df, directory, parameter_file, add_observable_function;
                       skip_invalid=true)

Load each sample directory containing the exact parameter filename once. Invalid
samples are skipped by default and recorded in the returned `DatasetLoadReport`.

# Arguments
- `df::DataFrame`: Target DataFrame
- `Directory::String`: Root directory to search
- `parameter_file`: Exact marker filename identifying a sample directory
- `add_observable_function`: Function to add one sample to the DataFrame
"""
function load_df_directory!(df::DataFrames.DataFrame, Directory::String,
    parameter_file::AbstractString, add_observable_function::Function; skip_invalid::Bool=true)
    if !isdir(Directory)
        throw(ArgumentError("Directory does not exist: $Directory"))
    end

    discovered = 0
    loaded = 0
    failures = DatasetLoadFailure[]
    for (root, dirs, files) in walkdir(Directory)
        sort!(dirs)
        sort!(files)
        if parameter_file in files
            discovered += 1
            try
                result = add_observable_function(df, root)
                if result === false
                    push!(failures, DatasetLoadFailure(root, "observation loader rejected sample"))
                else
                    loaded += 1
                end
            catch error
                skip_invalid || rethrow()
                message = sprint(showerror, error)
                push!(failures, DatasetLoadFailure(root, message))
                @warn "Skipping invalid sample directory" root exception=(error, catch_backtrace())
            end
        end
    end
    return DatasetLoadReport(discovered, loaded, discovered - loaded, failures)
end

"""
    extract_input_output_df(df; input_columns=nothing, observable_column=:observable)

Automatically detect and extract input and output features from a DataFrame.
The observable column may appear anywhere. Input columns can be given explicitly
to guarantee their order; otherwise every column except the observable is used.

# Returns
- `array_input::Matrix{Float64}`: Input features matrix (n_input_features × n_samples)
- `array_output::Matrix{Float64}`: Output features matrix (n_output_features × n_samples)
"""
function extract_input_output_df(df::AbstractDataFrame; input_columns=nothing,
    observable_column::Symbol=:observable)
    # Input validation
    if nrow(df) == 0
        throw(ArgumentError("DataFrame cannot be empty"))
    end
    
    if !hasproperty(df, observable_column)
        throw(ArgumentError("DataFrame must have an '$observable_column' column"))
    end

    columns = if input_columns === nothing
        filter(!=(observable_column), propertynames(df))
    else
        Symbol.(input_columns)
    end
    isempty(columns) && throw(ArgumentError("At least one input feature column is required"))
    length(unique(columns)) == length(columns) || throw(ArgumentError("Input columns must be unique"))
    observable_column in columns && throw(ArgumentError("Observable column cannot also be an input column"))
    for column in columns
        hasproperty(df, column) || throw(ArgumentError("Input column '$column' not found"))
        all(value -> value isa Real && isfinite(value), df[!, column]) ||
            throw(ArgumentError("Input column '$column' must contain finite Real values"))
    end

    n_input_features = length(columns)
    n_samples = nrow(df)
    observable_col = df[!, observable_column]
    first_observable = observable_col[1]
    n_output_features = length(first_observable)
    
    if n_output_features == 0
        throw(ArgumentError("Observable arrays cannot be empty"))
    end
    
    array_input = Matrix{Float64}(undef, n_input_features, n_samples)
    for (index, column) in enumerate(columns)
        array_input[index, :] = df[!, column]
    end

    array_output = Matrix{Float64}(undef, n_output_features, n_samples)
    for i in 1:n_samples
        obs = observable_col[i]
        if length(obs) != n_output_features
            throw(ArgumentError("Observable at row $i has wrong size: expected $n_output_features, got $(length(obs))"))
        end
        all(value -> value isa Real && isfinite(value), obs) ||
            throw(ArgumentError("Observable at row $i must contain finite Real values"))
        array_output[:, i] = obs
    end

    return array_input, array_output
end

"""
    get_minmax_in(df::DataFrame, array_pars_in::Vector{String})

Compute min/max values for specified input features.

# Arguments
- `df::DataFrame`: DataFrame with input features
- `array_pars_in::Vector{String}`: Column names to compute min/max for

# Returns
- `Matrix{Float64}`: Shape (n_params, 2) with [min, max] for each parameter
"""
function get_minmax_in(df::AbstractDataFrame, array_pars_in::AbstractVector)
    n_params = length(array_pars_in)
    if n_params == 0
        throw(ArgumentError("Parameter list cannot be empty"))
    end

    in_MinMax = Matrix{Float64}(undef, n_params, 2)
    for (idx, raw_key) in enumerate(array_pars_in)
        key = Symbol(raw_key)
        if !hasproperty(df, key)
            throw(ArgumentError("Column '$key' not found in DataFrame"))
        end
        col_data = df[!, key]
        in_MinMax[idx, 1] = minimum(col_data)
        in_MinMax[idx, 2] = maximum(col_data)
    end
    return in_MinMax
end

"""
    get_minmax_out(array_out::AbstractMatrix{<:Real})

Compute minimum and maximum values for each output feature.
Automatically detects the number of output features from the array dimensions.

# Arguments
- `array_out::AbstractMatrix{<:Real}`: Output array with shape (n_output_features, n_samples)

# Returns
- `out_MinMax::Matrix{Float64}`: Matrix with shape (n_output_features, 2) containing [min, max] for each feature
"""
function get_minmax_out(array_out::AbstractMatrix{<:Real})
    n_output_features, n_samples = size(array_out)
    
    if n_output_features == 0
        throw(ArgumentError("Array cannot be empty (0 output features)"))
    end
    
    if n_samples == 0
        throw(ArgumentError("Array cannot be empty (0 samples)"))
    end

    out_MinMax = Matrix{Float64}(undef, n_output_features, 2)

    # Vectorized min/max computation is more efficient
    for i in 1:n_output_features
        row_data = view(array_out, i, :)  # Use view to avoid copying
        out_MinMax[i, 1] = minimum(row_data)
        out_MinMax[i, 2] = maximum(row_data)
    end
    return out_MinMax
end

"""
    maximin_df!(df, in_MinMax, out_MinMax; input_columns,
                observable_column=:observable)

Normalize DataFrame features to [0, 1] range in-place.

# Arguments
- `df`: DataFrame to normalize
- `in_MinMax`: Min/max values for input features
- `out_MinMax`: Min/max values for output features
"""
function maximin_df!(df::AbstractDataFrame, in_MinMax::AbstractMatrix,
    out_MinMax::AbstractMatrix; input_columns, observable_column::Symbol=:observable)
    columns = Symbol.(input_columns)
    size(in_MinMax) == (length(columns), 2) ||
        throw(ArgumentError("Input min-max shape must be ($(length(columns)), 2)"))
    hasproperty(df, observable_column) || throw(ArgumentError("Observable column '$observable_column' not found"))
    input_widths = in_MinMax[:, 2] .- in_MinMax[:, 1]
    output_widths = out_MinMax[:, 2] .- out_MinMax[:, 1]
    constant_inputs = findall(iszero, input_widths)
    constant_outputs = findall(iszero, output_widths)
    isempty(constant_inputs) || throw(ArgumentError("Cannot normalize constant input features at indices $constant_inputs"))
    isempty(constant_outputs) || throw(ArgumentError("Cannot normalize constant output features at indices $constant_outputs"))

    for (i, column) in enumerate(columns)
        hasproperty(df, column) || throw(ArgumentError("Input column '$column' not found"))
        df[!, column] .-= in_MinMax[i, 1]
        df[!, column] ./= input_widths[i]
    end
    for i in 1:nrow(df)
        length(df[!, observable_column][i]) == size(out_MinMax, 1) ||
            throw(ArgumentError("Observable at row $i does not match output min-max dimensions"))
        df[!, observable_column][i] .-= out_MinMax[:, 1]
        df[!, observable_column][i] ./= output_widths
    end
    return df
end

"""
    split_indices(n_rows, pct; seed=nothing)

Construct randomized index vectors for a two-way split.

# Arguments
- `n_rows`: Number of rows to split
- `pct`: Fraction for first split (0 to 1)
- `seed`: Optional local random seed for reproducible indices

# Returns
- `(first_indices, second_indices)`: Index vectors for both partitions
"""
function split_indices(n_rows::Integer, pct::Real; seed::Union{Nothing,Integer}=nothing)
    if !(0 <= pct <= 1)
        throw(ArgumentError("Split percentage must be between 0 and 1, got $pct"))
    end
    if n_rows <= 0
        throw(ArgumentError("Cannot split empty DataFrame"))
    end
    split_idx = round(Int, n_rows * pct)
    rng = isnothing(seed) ? Random.default_rng() : Random.Xoshiro(seed)
    permutation = randperm(rng, n_rows)
    return permutation[1:split_idx], permutation[(split_idx + 1):end]
end

"""
    splitdf(df, pct; seed=nothing)

Randomly split a DataFrame into two views. Supplying `seed` uses a local RNG and
does not mutate global random state.
"""
function splitdf(df::DataFrames.DataFrame, pct::Real; seed::Union{Nothing,Integer}=nothing)
    first_indices, second_indices = split_indices(nrow(df), pct; seed)
    return view(df, first_indices, :), view(df, second_indices, :)
end

"""
    traintest_split(df, test)

Split DataFrame into training and test sets.

# Arguments
- `df`: DataFrame to split
- `test`: Fraction for test set

# Returns
- `(train_df, test_df)`: Training and test DataFrames
"""
function traintest_split(df, test; seed::Union{Nothing,Integer}=nothing)
    te, tr = splitdf(df, test; seed)
    return tr, te
end

"""
    getdata(df; test_fraction=0.2, seed=nothing, input_columns=nothing,
            observable_column=:observable, return_indices=false)

Split a DataFrame into train/test matrices. Supplying a seed makes the split
reproducible without mutating the global random generator.

# Arguments
- `df`: DataFrame with features and observables

# Returns
- `(xtrain, ytrain, xtest, ytest)`: Training and test arrays as Float64
"""
function getdata(df; test_fraction::Real=0.2, seed::Union{Nothing,Integer}=nothing,
    input_columns=nothing, observable_column::Symbol=:observable, return_indices::Bool=false)
    test_indices, train_indices = split_indices(nrow(df), test_fraction; seed)
    isempty(train_indices) && throw(ArgumentError("Training split is empty"))
    isempty(test_indices) && throw(ArgumentError("Test split is empty"))
    train_df = view(df, train_indices, :)
    test_df = view(df, test_indices, :)
    xtrain, ytrain = extract_input_output_df(train_df; input_columns, observable_column)
    xtest, ytest = extract_input_output_df(test_df; input_columns, observable_column)
    data = (Float64.(xtrain), Float64.(ytrain), Float64.(xtest), Float64.(ytest))
    return return_indices ? (data..., train_indices, test_indices) : data
end
