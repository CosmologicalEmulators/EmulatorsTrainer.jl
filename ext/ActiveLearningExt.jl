module ActiveLearningExt

using EmulatorsTrainer
using Turing
using SimpleFlows
using Pathfinder
using Random
using LinearAlgebra
using QuasiMonteCarlo
using JSON3
using Distributed

# --- Internal Helpers ---

function combine_chains(chn, n_params::Int)
    if chn isa Pathfinder.MultiPathfinderResult
        # .draws is (D, N) — importance-resampled, constrained space
        return Matrix(chn.draws[1:n_params, :])
    elseif chn isa AbstractArray
        return length(size(chn)) == 3 ? Matrix(chn[1:n_params, :, 1]) : Matrix(chn[1:n_params, :])
    else
        arr = Array(chn)
        return Matrix(transpose(reshape(arr[:, 1:n_params, :], :, n_params)))
    end
end

# --- Extension Overloads ---

"""
    EmulatorsTrainer.train_active_nf(chains, n_params::Int)

Trains a Normalising Flow (RealNVP) on the provided posterior samples/chains.
"""
function EmulatorsTrainer.train_active_nf(chains, n_params::Int)
    samples = combine_chains(chains, n_params)
    
    flow = FlowDistribution(Float32; architecture=:RealNVP, 
                            n_transforms=5, dist_dims=n_params, 
                            hidden_layer_sizes=[32, 32, 32], n_layers=4)
    
    @info "Training Normalising Flow proxy on posterior samples..."
    SimpleFlows.train_flow!(flow, Float32.(samples); batch_size=256, n_epochs=2000, lr=1e-4)

    return flow
end

"""
    EmulatorsTrainer.draw_latent_samples(flow::FlowDistribution; lb::Vector{Float64}, ub::Vector{Float64}, σ::Real=5, n_samples::Int=300)

Draws samples from the latent space of the flow within a σ-hypersphere and maps them back to physical space.
Enforces hard boundaries [lb, ub].
"""
function EmulatorsTrainer.draw_latent_samples(flow::FlowDistribution; lb::Vector{Float64}, ub::Vector{Float64}, σ::Real=5, n_samples::Int=300)
    n_dims = flow.n_dims
    valid_physical_points = Matrix{Float64}(undef, n_dims, 0)
    
    @info "Starting sampling loop to reach target $n_samples points within boundaries..."
    
    while size(valid_physical_points, 2) < n_samples
        batch_size = max(n_samples * 5, 500)
        s = QuasiMonteCarlo.sample(batch_size, fill(-Float64(σ), n_dims), fill(Float64(σ), n_dims), LatinHypercubeSample())
        
        z_batch = [col for col in eachcol(s) if norm(col) <= σ]
        if isempty(z_batch)
            continue
        end
        z_proposals = hcat(z_batch...)
        
        # Latent -> Parameter Space Mapping
        x_proposals = Float32.(z_proposals)
        for i in 1:flow.model.n_transforms
            k = keys(flow.model.conditioners)[i]
            mask = flow.st.mask_list[i]
            
            cond_fn = let m = flow.model.conditioners[k], p = flow.ps.conditioners[k],
                          s = flow.st.conditioners[k]
                x_cond -> SimpleFlows.Lux.apply(m, x_cond, p, s)[1]
            end
            
            bj = SimpleFlows.MaskedCoupling(mask, cond_fn, SimpleFlows.AffineBijector)
            x_proposals, _ = SimpleFlows.forward_and_log_det(bj, x_proposals)
        end
        
        p_batch = Float64.(SimpleFlows.denormalize(flow.normalizer, x_proposals))
        
        # Enforce Boundaries
        for col in eachcol(p_batch)
            if all(lb .<= col .<= ub)
                valid_physical_points = hcat(valid_physical_points, col)
            end
            if size(valid_physical_points, 2) >= n_samples
                break
            end
        end
        @info "Yielded $(size(valid_physical_points, 2)) / $n_samples points..."
    end
    
    return valid_physical_points[:, 1:n_samples]
end

"""
    EmulatorsTrainer.run_nuts(model; nsamples=500, nadapts=200, initial_params=nothing, n_chains=1)

Standard NUTS wrapper.
"""
function EmulatorsTrainer.run_nuts(model; nsamples=500, nadapts=200, initial_params=nothing, n_chains=1)
    @info "Running NUTS inference (via Extension)..."
    if n_chains > 1
        parallel_type = nprocs() > 1 ? MCMCDistributed() : MCMCThreads()
        @info "Parallelizing with $(typeof(parallel_type)) using $(nprocs() > 1 ? nprocs() : Threads.nthreads()) workers/threads."
        return sample(model, NUTS(nadapts, 0.65), parallel_type, nsamples, n_chains, initial_params=initial_params)
    else
        return sample(model, NUTS(nadapts, 0.65), nsamples, initial_params=initial_params)
    end
end

"""
    EmulatorsTrainer.run_pathfinder(model; ndraws=5000, nruns=12)

Standard Pathfinder wrapper.
"""
function EmulatorsTrainer.run_pathfinder(model; ndraws=5000, nruns=8)
    @info "Running Pathfinder inference (via Extension) with $nruns runs..."
    return multipathfinder(model, ndraws; nruns=nruns)
end

"""
    EmulatorsTrainer.prune_dataset(chains, data_dir::String, n_sigma::Real, n_params::Int, pars::Vector{String}; archive_dir::Union{String, Nothing}=nothing)

Prunes the dataset by moving outliers (outside n_sigma) to an archive directory.
"""
function EmulatorsTrainer.prune_dataset(chains, data_dir::String, n_sigma::Real, n_params::Int, pars::Vector{String}; archive_dir::Union{String, Nothing}=nothing)
    samples = combine_chains(chains, n_params)
    n_samples_count = size(samples, 2)
    μ = sum(samples, dims=2) ./ n_samples_count
    μ2 = sum(samples.^2, dims=2) ./ n_samples_count
    σ_vec = sqrt.(max.(μ2 .- μ.^2, 0.0))
    
    archive_dir !== nothing && mkpath(archive_dir)

    subdirs = filter(isdir, [joinpath(data_dir, d) for d in readdir(data_dir)])
    pruned_count = 0
    for sd in subdirs
        dict_path = joinpath(sd, "capse_dict.json")
        !isfile(dict_path) && continue
        
        CosmoDict = JSON3.read(read(dict_path, String))
        θ = [Float64(CosmoDict[Symbol(p)]) for p in pars]
        
        is_outside = false
        for j in 1:n_params
            if abs(θ[j] - μ[j]) > n_sigma * σ_vec[j]
                is_outside = true
                break
            end
        end
        
        if is_outside
            if archive_dir !== nothing
                mv(sd, joinpath(archive_dir, basename(sd)), force=true)
            else
                rm(sd, recursive=true)
            end
            pruned_count += 1
        end
    end
    
    @info "Pruning complete. $(archive_dir === nothing ? "Removed" : "Archived") $pruned_count outliers."
end

end # module
