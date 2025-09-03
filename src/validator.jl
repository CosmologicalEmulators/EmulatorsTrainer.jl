function evaluate_residuals(Directory::String, dict_file::String, pars_array::Vector{String},
    get_ground_truth::Function, get_emu_prediction::Function, n_combs::Int,
    n_output_features::Int; get_σ::Union{Function,Nothing}=nothing)
    # Input validation
    if !isdir(Directory)
        throw(ArgumentError("Directory does not exist: $Directory"))
    end
    if n_combs <= 0 || n_output_features <= 0
        throw(ArgumentError("n_combs and n_output_features must be positive"))
    end
    if isempty(pars_array)
        throw(ArgumentError("Parameter array cannot be empty"))
    end
    if isempty(dict_file)
        throw(ArgumentError("Dictionary file name cannot be empty"))
    end
    
    my_values = Matrix{Float64}(undef, n_combs, n_output_features)
    i = 0
    
    for (root, dirs, files) in walkdir(Directory)
        for file in files
            if endswith(file, ".json")
                if i >= n_combs
                    @warn "Found more JSON files than expected n_combs=$n_combs. Stopping early."
                    break
                end
                try
                    if get_σ !== nothing
                        res = get_single_residuals(root, dict_file, pars_array,
                            get_ground_truth, get_emu_prediction, get_σ)
                    else
                        res = get_single_residuals(root, dict_file, pars_array,
                            get_ground_truth, get_emu_prediction)
                    end
                    i += 1
                    my_values[i, :] = res
                catch e
                    @warn "Failed to process file in $root" exception=e
                    continue
                end
            end
        end
        if i >= n_combs
            break
        end
    end
    
    if i < n_combs
        @warn "Found fewer JSON files ($i) than expected n_combs=$n_combs. Resizing output."
        return my_values[1:i, :]
    end
    
    return my_values
end

function evaluate_sorted_residuals(Directory::String, dict_file::String, pars_array::Vector{String},
    get_ground_truth::Function, get_emu_prediction::Function,
    n_combs::Int, n_output_features::Int; get_σ::Union{Function,Nothing}=nothing)
    residuals = evaluate_residuals(Directory, dict_file, pars_array,
        get_ground_truth, get_emu_prediction, n_combs, n_output_features; get_σ=get_σ)
    actual_n_combs = size(residuals, 1)
    return sort_residuals(residuals, n_output_features, actual_n_combs)
end

function sort_residuals(residuals::AbstractMatrix{<:Real}, n_output::Int, n_elements::Int)
    # Input validation
    if n_output <= 0 || n_elements <= 0
        throw(ArgumentError("n_output and n_elements must be positive"))
    end
    if size(residuals, 1) != n_elements || size(residuals, 2) != n_output
        throw(ArgumentError("Residuals matrix dimensions ($(size(residuals))) don't match n_elements=$n_elements, n_output=$n_output"))
    end
    if n_elements < 3
        throw(ArgumentError("Need at least 3 elements for percentile computation, got n_elements=$n_elements"))
    end
    
    sorted_residuals = Matrix{Float64}(undef, n_elements, n_output)
    for i in 1:n_output
        sorted_residuals[:, i] = sort(residuals[:, i])
    end
    
    final_residuals = Matrix{Float64}(undef, 3, n_output)
    
    # Safe percentile computation with bounds checking
    idx_68 = max(1, min(n_elements, round(Int, n_elements * 0.68)))
    idx_95 = max(1, min(n_elements, round(Int, n_elements * 0.95)))
    idx_997 = max(1, min(n_elements, round(Int, n_elements * 0.997)))
    
    final_residuals[1, :] = sorted_residuals[idx_68, :]
    final_residuals[2, :] = sorted_residuals[idx_95, :]
    final_residuals[3, :] = sorted_residuals[idx_997, :]
    
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
