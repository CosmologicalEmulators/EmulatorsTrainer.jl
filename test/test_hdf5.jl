@testset "HDF5 shard datasets" begin
    training_matrix = [1.0 2.0 3.0 4.0;
                       10.0 20.0 30.0 40.0]
    parameter_names = ["a", "b"]
    compute_one(p) = (spectrum=[p["a"], p["b"]],
                      cube=reshape([p["a"], p["b"], p["a"] + p["b"], 1.0], 2, 2))

    for mode in (:serial, :threads)
        root = joinpath(mktempdir(), string(mode))
        output = EmulatorsTrainer.compute_dataset_hdf5(
            training_matrix, parameter_names, root, compute_one; mode=mode)

        @test output == joinpath(root, "dataset.h5")
        @test isfile(output)
        @test isfile(joinpath(root, "shards", "shard_0001.h5"))

        dataset = EmulatorsTrainer.load_hdf5_dataset(output)
        @test dataset.parameter_names == parameter_names
        @test dataset.parameters == permutedims(training_matrix)
        @test dataset.sample_indices == collect(1:4)
        @test all(dataset.valid)
        @test dataset.observables[:spectrum] == [1.0 10.0; 2.0 20.0; 3.0 30.0; 4.0 40.0]
        @test size(dataset.observables[:cube]) == (4, 2, 2)
    end
end

@testset "HDF5 per-sample failure handling" begin
    training_matrix = reshape(collect(1.0:8.0), 1, :)
    parameter_names = ["x"]
    function compute_with_failures(parameters)
        x = Int(parameters["x"])
        x in (1, 4, 7) && error("rejected sample $x")
        return (value=[x, x^2],)
    end

    strict_root = joinpath(mktempdir(), "strict")
    @test_throws ErrorException EmulatorsTrainer.compute_dataset_hdf5(
        training_matrix, parameter_names, strict_root, compute_with_failures,
    )

    tolerant_root = joinpath(mktempdir(), "tolerant")
    output = EmulatorsTrainer.compute_dataset_hdf5(
        training_matrix, parameter_names, tolerant_root, compute_with_failures;
        skip_errors=true,
    )
    dataset = EmulatorsTrainer.load_hdf5_dataset(output)
    @test dataset.sample_indices == [2, 3, 5, 6, 8]
    @test dataset.parameters == reshape([2.0, 3.0, 5.0, 6.0, 8.0], :, 1)
    @test all(dataset.valid)
    @test dataset.observables[:value] == [2 4; 3 9; 5 25; 6 36; 8 64]

    failure_path = joinpath(tolerant_root, "generation_failures.json")
    @test isfile(failure_path)
    failures = JSON3.read(read(failure_path, String))
    @test [failure["sample_index"] for failure in failures] == [1, 4, 7]
    @test all(occursin("rejected sample", failure["error"]) for failure in failures)
end
