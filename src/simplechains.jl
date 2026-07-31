"""Configuration for the centralized SimpleChains learning-rate schedule."""
Base.@kwdef struct SimpleChainsTrainingConfig
    learning_rates::Vector{Float64} = [1.0e-4, 7.0e-5, 5.0e-5, 2.0e-5, 1.0e-5,
        7.0e-6, 5.0e-6, 2.0e-6, 1.0e-6, 7.0e-7]
    sessions_per_rate::Int = 10
    steps_per_session::Int = 1_000
    batch_size::Int = 256
    initialization_seed::Union{Nothing,Int} = nothing
end

"""Best checkpoint, complete loss history, timing, and training configuration."""
struct SimpleChainsTrainingResult{P}
    best_parameters::P
    best_validation_loss::Float64
    history::Matrix{Float64}
    total_steps::Int
    elapsed_seconds::Float64
    config::SimpleChainsTrainingConfig
end

function _validate_training_config(config::SimpleChainsTrainingConfig)
    isempty(config.learning_rates) && throw(ArgumentError("At least one learning rate is required"))
    all(rate -> isfinite(rate) && rate > 0, config.learning_rates) ||
        throw(ArgumentError("Learning rates must be finite and positive"))
    config.sessions_per_rate > 0 || throw(ArgumentError("sessions_per_rate must be positive"))
    config.steps_per_session > 0 || throw(ArgumentError("steps_per_session must be positive"))
    config.batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    return nothing
end

function _validate_training_arrays(x_train, y_train, x_validation, y_validation)
    ndims(x_train) == ndims(y_train) == ndims(x_validation) == ndims(y_validation) == 2 ||
        throw(ArgumentError("Training and validation arrays must be matrices"))
    size(x_train, 2) == size(y_train, 2) ||
        throw(ArgumentError("Training input and output sample counts differ"))
    size(x_validation, 2) == size(y_validation, 2) ||
        throw(ArgumentError("Validation input and output sample counts differ"))
    size(x_train, 1) == size(x_validation, 1) ||
        throw(ArgumentError("Training and validation input dimensions differ"))
    size(y_train, 1) == size(y_validation, 1) ||
        throw(ArgumentError("Training and validation output dimensions differ"))
    size(x_train, 2) > 0 || throw(ArgumentError("Training data cannot be empty"))
    size(x_validation, 2) > 0 || throw(ArgumentError("Validation data cannot be empty"))
    all(isfinite, x_train) && all(isfinite, y_train) &&
        all(isfinite, x_validation) && all(isfinite, y_validation) ||
        throw(ArgumentError("Training and validation arrays must contain only finite values"))
    return nothing
end

"""
    train_simplechains(network, x_train, y_train, x_validation, y_validation;
                       config=SimpleChainsTrainingConfig(), callback=nothing)

Train a SimpleChains network with a fixed learning-rate schedule, retaining the
parameters with the smallest validation loss. The callback, when provided, is
called after each session with a named tuple containing the current progress.
The checkpoint callback is called once for the initial parameters and whenever
the validation loss improves, with `(parameters, progress)` arguments.
"""
function train_simplechains(network, x_train, y_train, x_validation, y_validation;
    config::SimpleChainsTrainingConfig=SimpleChainsTrainingConfig(), callback=nothing,
    checkpoint_callback=nothing)
    _validate_training_config(config)
    _validate_training_arrays(x_train, y_train, x_validation, y_validation)

    initialization_rng = isnothing(config.initialization_seed) ?
        Random.default_rng() : Random.Xoshiro(config.initialization_seed)
    parameters = SimpleChains.init_params(network; rng=initialization_rng)
    gradient = SimpleChains.alloc_threaded_grad(network)
    training_loss = SimpleChains.add_loss(network, SimpleChains.SquaredLoss(y_train))
    validation_loss = SimpleChains.add_loss(network, SimpleChains.SquaredLoss(y_validation))

    best_loss = Float64(validation_loss(x_validation, parameters))
    isfinite(best_loss) || throw(ArgumentError("Initial validation loss is not finite"))
    best_parameters = copy(parameters)
    n_sessions = length(config.learning_rates) * config.sessions_per_rate
    history = Matrix{Float64}(undef, n_sessions, 4)
    total_steps = 0
    session = 0
    start_time = time()

    if checkpoint_callback !== nothing
        checkpoint_callback(best_parameters, (
            session=0,
            total_steps=0,
            learning_rate=NaN,
            training_loss=Float64(training_loss(x_train, parameters)),
            validation_loss=best_loss,
            best_validation_loss=best_loss,
        ))
    end

    for learning_rate in config.learning_rates
        for _ in 1:config.sessions_per_rate
            session += 1
            SimpleChains.train_batched!(
                gradient,
                parameters,
                training_loss,
                x_train,
                SimpleChains.ADAM(learning_rate),
                config.steps_per_session;
                batchsize=min(config.batch_size, size(x_train, 2)),
            )
            total_steps += config.steps_per_session
            current_training_loss = Float64(training_loss(x_train, parameters))
            current_validation_loss = Float64(validation_loss(x_validation, parameters))
            isfinite(current_training_loss) && isfinite(current_validation_loss) ||
                throw(ErrorException("Training produced a non-finite loss at step $total_steps"))
            history[session, :] .= (
                total_steps, learning_rate, current_training_loss, current_validation_loss,
            )
            if current_validation_loss < best_loss
                best_loss = current_validation_loss
                best_parameters .= parameters
                if checkpoint_callback !== nothing
                    checkpoint_callback(best_parameters, (;
                        session,
                        total_steps,
                        learning_rate,
                        training_loss=current_training_loss,
                        validation_loss=current_validation_loss,
                        best_validation_loss=best_loss,
                    ))
                end
            end
            if callback !== nothing
                callback((;
                    session,
                    total_steps,
                    learning_rate,
                    training_loss=current_training_loss,
                    validation_loss=current_validation_loss,
                    best_validation_loss=best_loss,
                ))
            end
        end
    end

    return SimpleChainsTrainingResult(
        collect(best_parameters),
        best_loss,
        history,
        total_steps,
        time() - start_time,
        config,
    )
end

function _training_metadata(result::SimpleChainsTrainingResult)
    config = result.config
    return Dict{String,Any}(
        "best_validation_loss" => result.best_validation_loss,
        "total_steps" => result.total_steps,
        "elapsed_seconds" => result.elapsed_seconds,
        "learning_rates" => config.learning_rates,
        "sessions_per_rate" => config.sessions_per_rate,
        "steps_per_session" => config.steps_per_session,
        "batch_size" => config.batch_size,
        "initialization_seed" => config.initialization_seed,
        "julia_version" => string(VERSION),
        "emulators_trainer_version" => string(pkgversion(EmulatorsTrainer)),
        "saved_at" => string(now()),
    )
end

"""
    save_training_result(output_directory, result; metadata=Dict())

Save best weights, the complete loss history, and merged JSON metadata.
"""
function save_training_result(output_directory::AbstractString,
    result::SimpleChainsTrainingResult; metadata::AbstractDict=Dict())
    mkpath(output_directory)
    npzwrite(joinpath(output_directory, "weights.npy"), result.best_parameters)
    npzwrite(joinpath(output_directory, "training_history.npy"), result.history)
    combined = _training_metadata(result)
    for (key, value) in metadata
        combined[string(key)] = value
    end
    open(joinpath(output_directory, "training_metadata.json"), "w") do stream
        JSON3.write(stream, combined)
    end
    return output_directory
end
