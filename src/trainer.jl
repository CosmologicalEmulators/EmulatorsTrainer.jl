"""
    add_observable_df!(df::DataFrame, location::String, param_file::String,
                      observable_file::String, first_idx::Int, last_idx::Int, get_tuple::Function)

Add observation slice to DataFrame with NaN checking.

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
    observable_file::String, first_idx::Int, last_idx::Int, get_tuple::Function)
    json_string = read(location * param_file, String)
    cosmo_pars = JSON3.read(json_string)

    observable = npzread(location * observable_file, "r")[first_idx:last_idx]

    if !any(isnan.(observable))
        processed_observable = get_tuple(cosmo_pars, observable)
        push!(df, processed_observable)
    else
        @warn "File with NaN at " * location
    end

    return nothing
end

"""
    add_observable_df!(df::DataFrame, location::String, param_file::String,
                      observable_file::String, get_tuple::Function)

Add complete observation to DataFrame with NaN checking.

# Arguments
- `df::DataFrame`: Target DataFrame
- `location::String`: Directory containing files
- `param_file::String`: JSON file with parameters
- `observable_file::String`: NPY file with observables
- `get_tuple::Function`: Function to process (params, observable) into tuple
"""
function add_observable_df!(df::DataFrames.DataFrame, location::String, param_file::String,
    observable_file::String, get_tuple::Function)
    json_string = read(location * param_file, String)
    cosmo_pars = JSON3.read(json_string)

    observable = npzread(location * observable_file, "r")
    
    if !any(isnan.(observable))
        processed_observable = get_tuple(cosmo_pars, observable)
        push!(df, processed_observable)
    else
        @warn "File with NaN at " * location
    end
    
    return nothing
end

"""
    load_df_directory!(df::DataFrame, Directory::String, add_observable_function::Function)

Recursively load all observations from directory into DataFrame.

# Arguments
- `df::DataFrame`: Target DataFrame
- `Directory::String`: Root directory to search
- `add_observable_function::Function`: Function to add each observation
"""
function load_df_directory!(df::DataFrames.DataFrame, Directory::String,
    add_observable_function::Function)
    if !isdir(Directory)
        throw(ArgumentError("Directory does not exist: $Directory"))
    end

    for (root, dirs, files) in walkdir(Directory)
        for file in files
            if endswith(file, ".json")
                # Call the add_observable function with the root directory
                # Note: The actual function signature depends on which add_observable_df! variant is used
                add_observable_function(df, root * "/")
            end
        end
    end
end

"""
    extract_input_output_df(df::AbstractDataFrame)

Automatically detect and extract input and output features from a DataFrame.
Assumes the last column named "observable" contains the output arrays and all other columns are input features.

# Returns
- `array_input::Matrix{Float64}`: Input features matrix (n_input_features × n_samples)
- `array_output::Matrix{Float64}`: Output features matrix (n_output_features × n_samples)
"""
function extract_input_output_df(df::AbstractDataFrame)
    # Input validation
    if nrow(df) == 0
        throw(ArgumentError("DataFrame cannot be empty"))
    end
    
    if !hasproperty(df, :observable)
        throw(ArgumentError("DataFrame must have an 'observable' column"))
    end
    
    # Auto-detect dimensions
    n_input_features = ncol(df) - 1  # All columns except "observable"
    n_samples = nrow(df)
    
    # Get n_output_features from the first observable
    first_observable = df.observable[1]
    n_output_features = length(first_observable)
    
    if n_output_features == 0
        throw(ArgumentError("Observable arrays cannot be empty"))
    end
    
    if n_input_features <= 0
        throw(ArgumentError("DataFrame must have at least one input feature column besides 'observable'"))
    end

    # Extract input features with proper typing
    array_input = Matrix{Float64}(undef, n_input_features, n_samples)
    for i in 1:n_input_features
        array_input[i, :] = df[!, i]  # More efficient column-wise access
    end

    # Extract output features (observables) with proper typing
    array_output = Matrix{Float64}(undef, n_output_features, n_samples)
    observable_col = df[!, "observable"]  # Assumes column named "observable"

    # Vectorized extraction of observables
    for i in 1:n_samples
        obs = observable_col[i]
        if length(obs) != n_output_features
            throw(ArgumentError("Observable at row $i has wrong size: expected $n_output_features, got $(length(obs))"))
        end
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
function get_minmax_in(df::DataFrames.DataFrame, array_pars_in::AbstractVector{<:AbstractString})
    n_params = length(array_pars_in)
    if n_params == 0
        throw(ArgumentError("Parameter list cannot be empty"))
    end

    in_MinMax = Matrix{Float64}(undef, n_params, 2)
    for (idx, key) in enumerate(array_pars_in)
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
    maximin_df!(df, in_MinMax, out_MinMax)

Normalize DataFrame features to [0, 1] range in-place.

# Arguments
- `df`: DataFrame to normalize
- `in_MinMax`: Min/max values for input features
- `out_MinMax`: Min/max values for output features
"""
function maximin_df!(df, in_MinMax, out_MinMax)
    n_input_features, _ = size(in_MinMax)
    for i in 1:n_input_features
        df[!, i] .-= in_MinMax[i, 1]
        df[!, i] ./= (in_MinMax[i, 2] - in_MinMax[i, 1])
    end
    for i in 1:nrow(df)
        df[!, "observable"][i] .-= out_MinMax[:, 1]
        df[!, "observable"][i] ./= (out_MinMax[:, 2] - out_MinMax[:, 1])
    end
end

"""
    splitdf(df::DataFrame, pct::Float64)

Randomly split DataFrame into two parts.

# Arguments
- `df::DataFrame`: DataFrame to split
- `pct::Float64`: Fraction for first split (0 to 1)

# Returns
- `(DataFrame, DataFrame)`: Two views of the split data
"""
function splitdf(df::DataFrames.DataFrame, pct::Float64)
    if !(0 <= pct <= 1)
        throw(ArgumentError("Split percentage must be between 0 and 1, got $pct"))
    end

    n_rows = nrow(df)
    if n_rows == 0
        throw(ArgumentError("Cannot split empty DataFrame"))
    end

    # More efficient splitting
    split_idx = round(Int, n_rows * pct)
    indices = randperm(n_rows)  # More efficient than collect + shuffle

    # Create boolean masks more efficiently
    mask1 = falses(n_rows)
    mask1[indices[1:split_idx]] .= true

    return view(df, mask1, :), view(df, .!mask1, :)
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
function traintest_split(df, test)
    te, tr = splitdf(df, test)
    return tr, te
end

"""
    getdata(df)

Split DataFrame into train/test sets with automatic dimension detection.

# Arguments
- `df`: DataFrame with features and observables

# Returns
- `(xtrain, ytrain, xtest, ytest)`: Training and test arrays as Float64
"""
function getdata(df)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

    train_df, test_df = traintest_split(df, 0.2)

    xtrain, ytrain = extract_input_output_df(train_df)
    xtest, ytest = extract_input_output_df(test_df)

    return Float64.(xtrain), Float64.(ytrain), Float64.(xtest), Float64.(ytest)
end
