using Test
using EmulatorsTrainer
using JSON3
using NPZ
using Random
using SimpleChains

@testset "SimpleChains training" begin
    x = reshape(collect(Float32, range(-1, 1; length=64)), 1, :)
    y = @. 2 * x + 0.5f0
    network = SimpleChain(
        static(1),
        TurboDense(tanh, 8),
        TurboDense(identity, 1),
    )
    config = SimpleChainsTrainingConfig(
        learning_rates=[1.0e-2],
        sessions_per_rate=3,
        steps_per_session=100,
        batch_size=64,
        initialization_seed=42,
    )
    initial_parameters = SimpleChains.init_params(network; rng=Random.Xoshiro(42))
    initial_loss = SimpleChains.add_loss(network, SquaredLoss(y))(x, initial_parameters)
    callback_count = Ref(0)
    callback = progress -> begin
        callback_count[] += 1
        @test progress.total_steps > 0
    end
    result = train_simplechains(network, x, y, x, y; config, callback)
    @test result.best_validation_loss < initial_loss
    @test result.total_steps == 300
    @test size(result.history) == (3, 4)
    @test callback_count[] == 3
    @test all(isfinite, result.best_parameters)

    output = mktempdir()
    save_training_result(output, result; metadata=Dict("spectrum" => "synthetic"))
    @test isfile(joinpath(output, "weights.npy"))
    @test isfile(joinpath(output, "training_history.npy"))
    @test isfile(joinpath(output, "training_metadata.json"))
    @test npzread(joinpath(output, "training_history.npy")) == result.history
    metadata = JSON3.read(read(joinpath(output, "training_metadata.json"), String))
    @test metadata["spectrum"] == "synthetic"
    @test metadata["best_validation_loss"] == result.best_validation_loss
    rm(output; recursive=true)

    @test_throws ArgumentError train_simplechains(network, x, y[:, 1:10], x, y; config)
end
