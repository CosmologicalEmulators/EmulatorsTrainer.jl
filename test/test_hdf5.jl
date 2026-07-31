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
