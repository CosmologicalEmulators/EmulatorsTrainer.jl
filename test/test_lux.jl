using Test
using EmulatorsTrainer
using JSON3
using Lux
using NPZ
using Random

@testset "Lux training" begin
    rng = Xoshiro(1234)
    x = rand(rng, Float32, 2, 40)
    y = reshape(Float32.(2 .* x[1, :] .- x[2, :]), 1, :)
    model = Chain(Dense(2 => 8, tanh), Dense(8 => 1))
    config = LuxTrainingConfig(
        learning_rates=[1.0e-2],
        sessions_per_rate=2,
        steps_per_session=10,
        batch_size=8,
        initialization_seed=42,
    )
    callback_count = Ref(0)
    checkpoint_count = Ref(0)
    result = train_lux(
        model,
        x[:, 1:32],
        y[:, 1:32],
        x[:, 33:end],
        y[:, 33:end];
        config,
        callback=progress -> begin
            callback_count[] += 1
            @test progress.total_steps > 0
        end,
        checkpoint_callback=(parameters, states, progress) -> begin
            checkpoint_count[] += 1
            @test progress.total_steps >= 0
            @test parameters !== nothing
            @test states !== nothing
        end,
    )

    @test result.total_steps == 20
    @test size(result.history) == (2, 4)
    @test callback_count[] == 2
    @test checkpoint_count[] >= 1
    @test isfinite(result.best_validation_loss)

    output = mktempdir()
    save_training_result(output, result; metadata=Dict("basis" => "synthetic"))
    @test isfile(joinpath(output, "weights.npy"))
    @test isfile(joinpath(output, "training_history.npy"))
    @test isfile(joinpath(output, "training_metadata.json"))
    @test size(npzread(joinpath(output, "training_history.npy"))) == (2, 4)
    metadata = JSON3.read(read(joinpath(output, "training_metadata.json"), String))
    @test metadata["basis"] == "synthetic"
    @test metadata["batch_size"] == 8
    rm(output; recursive=true)

    @test_throws ArgumentError train_lux(model, x, y[:, 1:10], x, y; config)
end
