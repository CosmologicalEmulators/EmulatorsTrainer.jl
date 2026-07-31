"""Configuration for the Lux training schedule."""
Base.@kwdef struct LuxTrainingConfig
    learning_rates::Vector{Float64} = [1.0e-4, 7.0e-5, 5.0e-5, 2.0e-5, 1.0e-5,
        7.0e-6, 5.0e-6, 2.0e-6, 1.0e-6, 7.0e-7]
    sessions_per_rate::Int = 10
    steps_per_session::Int = 1_000
    batch_size::Int = 256
    initialization_seed::Union{Nothing,Int} = nothing
end

"""Best Lux parameters/states, complete loss history, timing, and configuration."""
struct LuxTrainingResult{M,P,S}
    model::M
    best_parameters::P
    best_states::S
    best_validation_loss::Float64
    history::Matrix{Float64}
    total_steps::Int
    elapsed_seconds::Float64
    config::LuxTrainingConfig
end

function _validate_lux_training_config(config::LuxTrainingConfig)
    isempty(config.learning_rates) && throw(ArgumentError("At least one learning rate is required"))
    all(rate -> isfinite(rate) && rate > 0, config.learning_rates) ||
        throw(ArgumentError("Learning rates must be finite and positive"))
    config.sessions_per_rate > 0 || throw(ArgumentError("sessions_per_rate must be positive"))
    config.steps_per_session > 0 || throw(ArgumentError("steps_per_session must be positive"))
    config.batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    return nothing
end

function _validate_lux_training_arrays(x_train, y_train, x_validation, y_validation)
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

function _lux_objective(model, parameters, states, data)
    x, y = data
    prediction, updated_states = Lux.apply(model, x, parameters, states)
    loss = sum(abs2, prediction .- y) / length(y)
    return loss, updated_states, (;)
end

function _lux_validation_loss(model, parameters, states, x, y)
    prediction, _ = Lux.apply(model, x, parameters, Lux.testmode(states))
    return Float64(sum(abs2, prediction .- y) / length(y))
end

function _lux_batch(x, y, rng, batch_size)
    indices = rand(rng, 1:size(x, 2), batch_size)
    return x[:, indices], y[:, indices]
end

"""
    train_lux(model, x_train, y_train, x_validation, y_validation;
              config=LuxTrainingConfig(), ad_backend=AutoZygote(), device=identity,
              callback=nothing, checkpoint_callback=nothing)

Train a Lux model using the same fixed learning-rate schedule as
`train_simplechains`. Samples are columns of the input/output matrices.

`device` can be a Lux device function such as `cpu_device()` or
`reactant_device()`. For Reactant training, use `AutoEnzyme()` as `ad_backend`
and provide the Reactant device function from the calling project. The generic
trainer does not import Reactant, so SimpleChains training remains independent
of that optional backend.

The checkpoint callback is called once with the initial parameters and whenever
the validation loss improves, using `(parameters, states, progress)` arguments.
"""
function train_lux(model, x_train, y_train, x_validation, y_validation;
    config::LuxTrainingConfig=LuxTrainingConfig(),
    ad_backend=ADTypes.AutoZygote(),
    device=identity,
    callback=nothing,
    checkpoint_callback=nothing,
)
    _validate_lux_training_config(config)
    _validate_lux_training_arrays(x_train, y_train, x_validation, y_validation)

    rng = isnothing(config.initialization_seed) ?
        Random.default_rng() : Random.Xoshiro(config.initialization_seed)
    parameters, states = Lux.setup(rng, model)
    parameters, states = device((parameters, states))
    target_eltype = eltype(first(values(parameters)).weight)
    x_train = device(target_eltype.(x_train))
    y_train = device(target_eltype.(y_train))
    x_validation = device(target_eltype.(x_validation))
    y_validation = device(target_eltype.(y_validation))
    validation_device = device isa Lux.ReactantDevice ? Lux.cpu_device() : identity

    optimizer = Optimisers.Adam(config.learning_rates[1])
    train_state = Lux.Training.TrainState(model, parameters, states, optimizer)
    best_validation_loss = _lux_validation_loss(
        model,
        validation_device(train_state.parameters),
        validation_device(train_state.states),
        validation_device(x_validation),
        validation_device(y_validation),
    )
    isfinite(best_validation_loss) ||
        throw(ArgumentError("Initial validation loss is not finite"))
    best_parameters = deepcopy(train_state.parameters)
    best_states = deepcopy(train_state.states)
    n_sessions = length(config.learning_rates) * config.sessions_per_rate
    history = Matrix{Float64}(undef, n_sessions, 4)
    total_steps = 0
    session = 0
    start_time = time()

    if checkpoint_callback !== nothing
        checkpoint_callback(best_parameters, best_states, (
            session=0,
            total_steps=0,
            learning_rate=NaN,
            training_loss=NaN,
            validation_loss=best_validation_loss,
            best_validation_loss,
        ))
    end

    for learning_rate in config.learning_rates
        train_state = Optimisers.adjust(train_state, learning_rate)
        for _ in 1:config.sessions_per_rate
            session += 1
            training_loss = 0.0
            for _ in 1:config.steps_per_session
                batch = _lux_batch(x_train, y_train, rng, config.batch_size)
                _, loss, _, train_state = Lux.Training.single_train_step!(
                    ad_backend, _lux_objective, batch, train_state,
                )
                training_loss = Float64(loss)
                total_steps += 1
            end
            validation_loss = _lux_validation_loss(
                model,
                validation_device(train_state.parameters),
                validation_device(train_state.states),
                validation_device(x_validation),
                validation_device(y_validation),
            )
            isfinite(training_loss) && isfinite(validation_loss) ||
                throw(ErrorException("Lux training produced a non-finite loss at step $total_steps"))
            history[session, :] .= (
                total_steps, learning_rate, training_loss, validation_loss,
            )
            if validation_loss < best_validation_loss
                best_validation_loss = validation_loss
                best_parameters = deepcopy(train_state.parameters)
                best_states = deepcopy(train_state.states)
                if checkpoint_callback !== nothing
                    checkpoint_callback(best_parameters, best_states, (;
                        session,
                        total_steps,
                        learning_rate,
                        training_loss,
                        validation_loss,
                        best_validation_loss,
                    ))
                end
            end
            if callback !== nothing
                callback((;
                    session,
                    total_steps,
                    learning_rate,
                    training_loss,
                    validation_loss,
                    best_validation_loss,
                ))
            end
        end
    end

    return LuxTrainingResult(
        model,
        best_parameters,
        best_states,
        best_validation_loss,
        history,
        total_steps,
        time() - start_time,
        config,
    )
end

function _lux_training_metadata(result::LuxTrainingResult)
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

function _flatten_lux_parameters(parameters)
    layers = propertynames(parameters)
    isempty(layers) && throw(ArgumentError("Lux parameters contain no layers"))
    first_layer = getproperty(parameters, first(layers))
    hasproperty(first_layer, :weight) && hasproperty(first_layer, :bias) ||
        throw(ArgumentError("Lux parameter serialization currently supports Dense layers only"))
    T = eltype(Array(getproperty(first_layer, :weight)))
    flattened = T[]
    for layer_name in layers
        layer = getproperty(parameters, layer_name)
        hasproperty(layer, :weight) && hasproperty(layer, :bias) ||
            throw(ArgumentError("Lux parameter serialization currently supports Dense layers only"))
        append!(flattened, vec(Array(getproperty(layer, :weight))))
        append!(flattened, vec(Array(getproperty(layer, :bias))))
    end
    return flattened
end

"""Save Lux weights, loss history, and metadata using the ACE-compatible files."""
function save_training_result(output_directory::AbstractString,
    result::LuxTrainingResult; metadata::AbstractDict=Dict())
    mkpath(output_directory)
    npzwrite(joinpath(output_directory, "weights.npy"), _flatten_lux_parameters(result.best_parameters))
    npzwrite(joinpath(output_directory, "training_history.npy"), result.history)
    combined = _lux_training_metadata(result)
    for (key, value) in metadata
        combined[string(key)] = value
    end
    open(joinpath(output_directory, "training_metadata.json"), "w") do stream
        JSON3.write(stream, combined)
    end
    return output_directory
end
