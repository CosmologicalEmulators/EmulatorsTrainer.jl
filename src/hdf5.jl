"""
    compute_dataset_hdf5(training_matrix, params, root_dir, compute_func;
                         mode=:serial, force=false, static_arrays=nothing)

Compute a dataset into one HDF5 shard per worker/thread and merge the shards
into `root_dir/dataset.h5`. `compute_func` receives one parameter dictionary
and must return a `NamedTuple` of same-shaped arrays, for example
`(TT=tt, EE=ee, TE=te, PP=pp)`.

The first dimension of every dataset is the sample dimension. Samples are
assigned deterministic, disjoint ranges, so no writer ever appends to a
shared file or modifies another writer's file.
"""
const _HDF5_IO_LOCK = ReentrantLock()

function _hdf5_shard_ranges(n_samples::Int, n_shards::Int)
    n_shards > 0 || throw(ArgumentError("n_shards must be positive"))
    q, r = divrem(n_samples, n_shards)
    ranges = UnitRange{Int}[]
    first_index = 1
    for shard in 1:n_shards
        count = q + (shard <= r)
        last_index = first_index + count - 1
        push!(ranges, first_index:last_index)
        first_index = last_index + 1
    end
    return ranges
end

function _hdf5_write_slice!(dataset, index::Int, value::AbstractArray)
    indices = (index, ntuple(_ -> Colon(), ndims(value))...)
    dataset[indices...] = value
end

function _hdf5_create_shard(path, training_matrix, parameter_names, sample_indices, first_result)
    isempty(sample_indices) && throw(ArgumentError("Cannot create an empty HDF5 shard"))
    names = propertynames(first_result)
    isempty(names) && throw(ArgumentError("compute_func returned no observables"))
    all(name -> getproperty(first_result, name) isa AbstractArray, names) ||
        throw(ArgumentError("compute_func must return arrays in a NamedTuple"))

    file = h5open(path, "w")
    try
        file["parameters"] = permutedims(training_matrix[:, sample_indices])
        file["parameter_names"] = String.(parameter_names)
        file["sample_indices"] = collect(sample_indices)
        file["valid"] = Bool[false for _ in sample_indices]
        observables = create_group(file, "observables")
        datasets = Dict{Symbol,Any}()
        for name in names
            value = getproperty(first_result, name)
            dims = (length(sample_indices), size(value)...)
            dataset = create_dataset(observables, String(name), datatype(eltype(value)), dataspace(dims))
            datasets[name] = dataset
        end
        return file, datasets
    catch
        close(file)
        rethrow()
    end
end

function _hdf5_write_result!(file, datasets, valid_dataset, local_index, result)
    Set(propertynames(result)) == Set(keys(datasets)) ||
        throw(ArgumentError("Observable names changed while writing an HDF5 shard"))
    for name in keys(datasets)
        value = getproperty(result, name)
        value isa AbstractArray || throw(ArgumentError("Observable $name is not an array"))
        size(value) == size(datasets[name])[2:end] ||
            throw(DimensionMismatch("Observable $name changed shape while writing an HDF5 shard"))
        _hdf5_write_slice!(datasets[name], local_index, value)
    end
    valid_dataset[local_index] = true
    flush(file)
end

function _hdf5_run_shard(training_matrix, parameter_names, sample_indices, shard_path,
                         compute_func, io_lock)
    isempty(sample_indices) && return shard_path
    first_result = compute_func(create_training_dict(training_matrix, first(sample_indices), parameter_names))
    file = nothing
    datasets = nothing
    valid_dataset = nothing
    try
        lock(io_lock)
        try
            file, datasets = _hdf5_create_shard(shard_path, training_matrix,
                parameter_names, sample_indices, first_result)
            valid_dataset = file["valid"]
            _hdf5_write_result!(file, datasets, valid_dataset, 1, first_result)
        finally
            unlock(io_lock)
        end

        for (offset, sample_index) in enumerate(sample_indices[2:end])
            local_index = offset + 1
            result = compute_func(create_training_dict(training_matrix, sample_index, parameter_names))
            lock(io_lock)
            try
                _hdf5_write_result!(file, datasets, valid_dataset, local_index, result)
            finally
                unlock(io_lock)
            end
        end
    finally
        if file !== nothing
            lock(io_lock)
            try
                close(file)
            finally
                unlock(io_lock)
            end
        end
    end
    return shard_path
end

function _hdf5_shard_files(shard_dir)
    files = filter(name -> endswith(name, ".h5"), readdir(shard_dir; join=true))
    isempty(files) && throw(ArgumentError("No HDF5 shards found in $shard_dir"))
    sort!(files)
    return files
end

"""Merge HDF5 shards into a single immutable dataset file."""
function merge_hdf5_shards(shard_dir::AbstractString, output_file::AbstractString;
                           n_samples::Union{Nothing,Int}=nothing, force::Bool=false)
    files = _hdf5_shard_files(shard_dir)
    isfile(output_file) && !force && error("Output HDF5 file already exists: $output_file")
    mkpath(dirname(output_file))
    tmp_file = output_file * ".tmp"
    isfile(tmp_file) && rm(tmp_file)

    parameter_names = String[]
    total_samples = 0
    n_parameters = 0
    observable_names = Symbol[]
    observable_shapes = Dict{Symbol,Tuple}()
    observable_types = Dict{Symbol,Any}()
    first_file = h5open(files[1], "r")
    try
        parameter_names = String.(first_file["parameter_names"][:])
        shard_sample_indices = map(files) do file_path
            h5open(file_path, "r") do shard
                Int.(shard["sample_indices"][:])
            end
        end
        total_samples = isnothing(n_samples) ? maximum(vcat(shard_sample_indices...)) : n_samples
        n_parameters = size(first_file["parameters"], 2)
        observable_names = Symbol.(keys(first_file["observables"]))
        observable_shapes = Dict(name => size(first_file["observables/$(name)"])[2:end] for name in observable_names)
        observable_types = Dict(name => eltype(first_file["observables/$(name)"]) for name in observable_names)
    finally
        close(first_file)
    end

    seen = falses(total_samples)
    final_file = h5open(tmp_file, "w")
    try
        final_file["parameter_names"] = parameter_names
        final_parameters = create_dataset(final_file, "parameters", datatype(Float64), dataspace(total_samples, n_parameters))
        final_indices = create_dataset(final_file, "sample_indices", datatype(Int), dataspace((total_samples,)))
        final_valid = create_dataset(final_file, "valid", datatype(Bool), dataspace((total_samples,)))
        final_valid[1:total_samples] = Bool[false for _ in 1:total_samples]
        observables = create_group(final_file, "observables")
        final_datasets = Dict{Symbol,Any}()
        for name in observable_names
            final_datasets[name] = create_dataset(observables, String(name), datatype(observable_types[name]),
                dataspace((total_samples, observable_shapes[name]...) ))
        end

        for shard_path in files
            h5open(shard_path, "r") do shard
                indices = Int.(shard["sample_indices"][:])
                valid = Bool.(shard["valid"][:])
                parameters = shard["parameters"]
                contiguous = !isempty(indices) && indices == collect(first(indices):last(indices))
                if contiguous
                    global_range = first(indices):last(indices)
                    any(seen[global_range]) && error("Duplicate sample index across HDF5 shards")
                    all((1 .<= indices) .& (indices .<= total_samples)) || error("HDF5 shard contains an invalid sample index")
                    seen[global_range] .= true
                    final_parameters[global_range, :] = parameters[:, :]
                    final_indices[global_range] = indices
                    final_valid[global_range] = valid
                    for name in observable_names
                        source = shard["observables/$(name)"]
                        source_indices = ntuple(_ -> Colon(), ndims(source))
                        destination = (global_range, ntuple(_ -> Colon(), ndims(source) - 1)...)
                        final_datasets[name][destination...] = source[source_indices...]
                    end
                else
                    for local_index in eachindex(indices)
                        global_index = indices[local_index]
                        1 <= global_index <= total_samples || error("sample index $global_index is outside 1:$total_samples")
                        !seen[global_index] || error("Duplicate sample index $global_index across HDF5 shards")
                        seen[global_index] = true
                        final_parameters[global_index, :] = parameters[local_index, :]
                        final_indices[global_index] = global_index
                        final_valid[global_index] = valid[local_index]
                        for name in observable_names
                            source = shard["observables/$(name)"]
                            source_indices = (local_index, ntuple(_ -> Colon(), ndims(source) - 1)...)
                            _hdf5_write_slice!(final_datasets[name], global_index, source[source_indices...])
                        end
                    end
                end
            end
        end
        all(seen) || error("HDF5 shards do not cover all samples: missing $(count(!, seen))")
        flush(final_file)
    finally
        close(final_file)
    end
    mv(tmp_file, output_file; force=true)
    return output_file
end

function compute_dataset_hdf5(training_matrix::AbstractMatrix, parameter_names::AbstractVector{<:AbstractString},
                              root_dir::AbstractString, compute_func::Function;
                              mode::Symbol=:serial, force::Bool=false,
                              static_arrays::Union{Nothing,NamedTuple}=nothing)
    validate_compute_inputs(training_matrix, parameter_names)
    mode in (:serial, :threads, :distributed) ||
        throw(ArgumentError("Invalid mode: $mode. Use :serial, :threads, or :distributed"))
    actual_dir = prepare_dataset_directory(String(root_dir); force=force)
    shard_dir = joinpath(actual_dir, "shards")
    mkpath(shard_dir)
    n_samples = size(training_matrix, 2)

    n_shards = mode == :serial ? 1 : mode == :threads ? Threads.nthreads() : nworkers()
    mode == :distributed && n_shards == 0 &&
        throw(ArgumentError("Distributed HDF5 mode requires workers. Call addprocs first."))
    ranges = _hdf5_shard_ranges(n_samples, n_shards)
    shard_paths = [joinpath(shard_dir, "shard_$(lpad(i, 4, '0')).h5") for i in 1:n_shards]

    if mode == :serial
        _hdf5_run_shard(training_matrix, parameter_names, ranges[1], shard_paths[1], compute_func, _HDF5_IO_LOCK)
    elseif mode == :threads
        Threads.@threads for shard_index in eachindex(ranges)
            _hdf5_run_shard(training_matrix, parameter_names, ranges[shard_index],
                            shard_paths[shard_index], compute_func, _HDF5_IO_LOCK)
        end
    else
        @sync @distributed for shard_index in eachindex(ranges)
            _hdf5_run_shard(training_matrix, parameter_names, ranges[shard_index],
                            shard_paths[shard_index], compute_func, _HDF5_IO_LOCK)
        end
    end

    output_file = joinpath(actual_dir, "dataset.h5")
    merge_hdf5_shards(shard_dir, output_file; n_samples=n_samples, force=true)
    if static_arrays !== nothing
        h5open(output_file, "r+") do file
            axes_group = create_group(file, "axes")
            for name in propertynames(static_arrays)
                value = getproperty(static_arrays, name)
                value isa AbstractArray || throw(ArgumentError("Static array $name is not an array"))
                axes_group[String(name)] = value
            end
        end
    end
    return output_file
end

"""Load the arrays stored by `compute_dataset_hdf5`."""
function load_hdf5_dataset(path::AbstractString)
    h5open(path, "r") do file
        names = String.(file["parameter_names"][:])
        parameters = Array(file["parameters"][:,:])
        sample_indices = Int.(file["sample_indices"][:])
        valid = Bool.(file["valid"][:])
        observables = Dict{Symbol,Array}()
        for name in keys(file["observables"])
            dataset = file["observables/$(name)"]
            all_indices = ntuple(_ -> Colon(), ndims(dataset))
            observables[Symbol(name)] = Array(dataset[all_indices...])
        end
        axes = Dict{Symbol,Array}()
        if haskey(file, "axes")
            for name in keys(file["axes"])
                dataset = file["axes/$(name)"]
                all_indices = ntuple(_ -> Colon(), ndims(dataset))
                axes[Symbol(name)] = Array(dataset[all_indices...])
            end
        end
        return (parameters=parameters, parameter_names=names, sample_indices=sample_indices,
                valid=valid, observables=observables, axes=axes)
    end
end
