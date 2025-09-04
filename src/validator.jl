function evaluate_residuals(Directory::String, dict_file::String, pars_array::Vector{String},
    get_ground_truth::Function, get_emu_prediction::Function; get_σ::Union{Function,Nothing}=nothing)
    # Input validation
    if !isdir(Directory)
        throw(ArgumentError("Directory does not exist: $Directory"))
    end
    if isempty(pars_array)
        throw(ArgumentError("Parameter array cannot be empty"))
    end
    if isempty(dict_file)
        throw(ArgumentError("Dictionary file name cannot be empty"))
    end
    
    # Collect all directories containing the target JSON file
    json_locations = String[]
    for (root, dirs, files) in walkdir(Directory)
        # Check if the target JSON file exists in this directory
        if dict_file in files
            push!(json_locations, root)
        end
    end
    
    if isempty(json_locations)
        throw(ArgumentError("No directories containing '$dict_file' found in: $Directory"))
    end
    
    actual_n_combs = length(json_locations)
    @info "Auto-detected $actual_n_combs directories with '$dict_file'"
    
    # Process first file to infer n_output_features
    first_location = json_locations[1]
    first_residuals = nothing
    try
        if get_σ !== nothing
            first_residuals = get_single_residuals(first_location, dict_file, pars_array,
                get_ground_truth, get_emu_prediction, get_σ)
        else
            first_residuals = get_single_residuals(first_location, dict_file, pars_array,
                get_ground_truth, get_emu_prediction)
        end
    catch e
        throw(ArgumentError("Failed to process first file to infer output dimensions: $(string(e))"))
    end
    
    n_output_features = length(first_residuals)
    @info "Auto-detected $n_output_features output features from first file"
    
    my_values = Matrix{Float64}(undef, actual_n_combs, n_output_features)
    my_values[1, :] = first_residuals
    i = 1
    
    # Process remaining files
    for location_idx in 2:actual_n_combs
        location = json_locations[location_idx]
        try
            res = nothing
            if get_σ !== nothing
                res = get_single_residuals(location, dict_file, pars_array,
                    get_ground_truth, get_emu_prediction, get_σ)
            else
                res = get_single_residuals(location, dict_file, pars_array,
                    get_ground_truth, get_emu_prediction)
            end
            
            # Validate consistent output dimensions
            if length(res) != n_output_features
                throw(ArgumentError("Inconsistent output dimensions: file in $location returned $(length(res)) features, expected $n_output_features"))
            end
            
            i += 1
            my_values[i, :] = res
        catch e
            @warn "Failed to process file in $location" exception=e
            continue
        end
    end
    
    if i < actual_n_combs
        @warn "Found fewer JSON files ($i) than expected n_combs=$actual_n_combs. Resizing output."
        return my_values[1:i, :]
    end
    
    return my_values
end

function evaluate_sorted_residuals(Directory::String, dict_file::String, pars_array::Vector{String},
    get_ground_truth::Function, get_emu_prediction::Function; 
    get_σ::Union{Function,Nothing}=nothing, percentiles::Vector{Float64}=[68.0, 95.0, 99.7])
    residuals = evaluate_residuals(Directory, dict_file, pars_array,
        get_ground_truth, get_emu_prediction; get_σ=get_σ)
    return sort_residuals(residuals; percentiles=percentiles)
end

function sort_residuals(residuals::AbstractMatrix{<:Real};
    percentiles::Vector{Float64}=[68.0, 95.0, 99.7])
    # Get dimensions from the residuals matrix
    n_elements, n_output = size(residuals)
    
    # Input validation
    if n_elements < 3
        throw(ArgumentError("Need at least 3 elements for percentile computation, got n_elements=$n_elements"))
    end
    if n_output == 0
        throw(ArgumentError("Residuals matrix has no output features"))
    end
    if isempty(percentiles)
        throw(ArgumentError("Percentiles array cannot be empty"))
    end
    if any(p -> p < 0 || p > 100, percentiles)
        throw(ArgumentError("Percentiles must be between 0 and 100"))
    end
    
    sorted_residuals = Matrix{Float64}(undef, n_elements, n_output)
    for i in 1:n_output
        sorted_residuals[:, i] = sort(residuals[:, i])
    end
    
    n_percentiles = length(percentiles)
    final_residuals = Matrix{Float64}(undef, n_percentiles, n_output)
    
    # Safe percentile computation with bounds checking
    # Using ceil for percentile indices to match standard percentile behavior
    for (p_idx, pct) in enumerate(percentiles)
        # For percentile p, we want the value at position ceil(n * p/100)
        idx = max(1, min(n_elements, ceil(Int, n_elements * pct / 100.0)))
        final_residuals[p_idx, :] = sorted_residuals[idx, :]
    end
    
    return final_residuals
end


function get_single_residuals(location::String, dict_file::String, pars_array::Vector{String},
    get_ground_truth::Function, get_emu_prediction::Function, get_σ::Function)
    # Input validation
    if isempty(location) || isempty(dict_file)
        throw(ArgumentError("Location and dict_file cannot be empty"))
    end
    if isempty(pars_array)
        throw(ArgumentError("Parameter array cannot be empty"))
    end
    
    param_file_path = joinpath(location, dict_file)
    if !isfile(param_file_path)
        throw(ArgumentError("Parameter file does not exist: $param_file_path"))
    end
    
    try
        json_string = read(param_file_path, String)
        cosmo_pars_test = JSON3.read(json_string)
        
        # Validate that all required parameters exist
        for param in pars_array
            if !haskey(cosmo_pars_test, param)
                throw(ArgumentError("Parameter '$param' not found in $param_file_path"))
            end
        end
        
        input_test = [cosmo_pars_test[param] for param in pars_array]
        
        obs_gt = get_ground_truth(location)
        obs_emu = get_emu_prediction(input_test)
        σ_obs = get_σ(location)
        
        # Check for division by zero
        if any(iszero, σ_obs)
            throw(ArgumentError("Found zero values in σ_obs, cannot divide by zero"))
        end
        
        res = abs.(obs_gt .- obs_emu) ./ σ_obs
        
        return res
    catch e
        if isa(e, ArgumentError)
            rethrow()
        else
            throw(ArgumentError("Failed to process residuals for $location: $(string(e))"))
        end
    end
end

function get_single_residuals(location::String, dict_file::String, pars_array::Vector{String},
    get_ground_truth::Function, get_emu_prediction::Function)
    # Input validation
    if isempty(location) || isempty(dict_file)
        throw(ArgumentError("Location and dict_file cannot be empty"))
    end
    if isempty(pars_array)
        throw(ArgumentError("Parameter array cannot be empty"))
    end
    
    param_file_path = joinpath(location, dict_file)
    if !isfile(param_file_path)
        throw(ArgumentError("Parameter file does not exist: $param_file_path"))
    end
    
    try
        json_string = read(param_file_path, String)
        cosmo_pars_test = JSON3.read(json_string)
        
        # Validate that all required parameters exist
        for param in pars_array
            if !haskey(cosmo_pars_test, param)
                throw(ArgumentError("Parameter '$param' not found in $param_file_path"))
            end
        end
        
        input_test = [cosmo_pars_test[param] for param in pars_array]
        
        obs_gt = get_ground_truth(location)
        obs_emu = get_emu_prediction(input_test)
        
        # Check for division by zero in relative error calculation
        if any(iszero, obs_gt)
            throw(ArgumentError("Found zero values in ground truth, cannot compute relative error"))
        end
        
        res = 100.0 .* abs.(1.0 .- obs_emu ./ obs_gt)
        
        return res
    catch e
        if isa(e, ArgumentError)
            rethrow()
        else
            throw(ArgumentError("Failed to process residuals for $location: $(string(e))"))
        end
    end
end
