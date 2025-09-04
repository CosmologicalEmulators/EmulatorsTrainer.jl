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

function add_observable_df!(df::DataFrames.DataFrame, location::String, param_file::String,
    observable_file::String, get_tuple::Function)
    json_string = read(location * param_file, String)
    cosmo_pars = JSON3.read(json_string)

    observable = npzread(location * observable_file, "r")
    processed_observable = get_tuple(cosmo_pars, observable)
    push!(df, processed_observable)
    return nothing
end

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

#TODO automatically detect n_input_features and n_output_features
function extract_input_output_df(df::AbstractDataFrame, n_input_features::Int, n_output_features::Int)
    # Input validation
    if n_input_features <= 0 || n_output_features <= 0
        throw(ArgumentError("Feature counts must be positive"))
    end
    if ncol(df) < n_input_features + 1  # +1 for observable column
        throw(ArgumentError("DataFrame has insufficient columns"))
    end
    if nrow(df) == 0
        throw(ArgumentError("DataFrame cannot be empty"))
    end

    n_samples = nrow(df)

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

function get_minmax_in(df::DataFrames.DataFrame, array_pars_in::Vector{String})
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

#TODO infer n_output_features
function get_minmax_out(array_out::AbstractMatrix{<:Real}, n_output_features::Int)
    if size(array_out, 1) != n_output_features
        throw(ArgumentError("Array first dimension ($(size(array_out, 1))) must match n_output_features ($n_output_features)"))
    end
    if n_output_features <= 0
        throw(ArgumentError("Number of output features must be positive"))
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

function traintest_split(df, test)
    te, tr = splitdf(df, test)
    return tr, te
end

function getdata(df, n_input_features::Int, n_output_features::Int)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

    train_df, test_df = traintest_split(df, 0.2)

    xtrain, ytrain = extract_input_output_df(train_df, n_input_features, n_output_features)
    xtest, ytest = extract_input_output_df(test_df, n_input_features, n_output_features)

    return Float64.(xtrain), Float64.(ytrain), Float64.(xtest), Float64.(ytest)
end
