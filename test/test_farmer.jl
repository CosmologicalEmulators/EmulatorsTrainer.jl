using Test
using EmulatorsTrainer
using QuasiMonteCarlo
using JSON3
using Distributed
using Dates
using Random

@testset "farmer.jl tests" begin
    
    @testset "create_training_dataset" begin
        @testset "Valid inputs" begin
            # Test basic functionality
            n = 100
            lb = [0.0, 1.0, 2.0]
            ub = [1.0, 2.0, 3.0]
            samples = EmulatorsTrainer.create_training_dataset(n, lb, ub)
            
            @test size(samples) == (3, n)
            @test all(samples[1, :] .>= 0.0) && all(samples[1, :] .<= 1.0)
            @test all(samples[2, :] .>= 1.0) && all(samples[2, :] .<= 2.0)
            @test all(samples[3, :] .>= 2.0) && all(samples[3, :] .<= 3.0)
            
            # Test with single parameter
            samples_1d = EmulatorsTrainer.create_training_dataset(50, [0.0], [10.0])
            @test size(samples_1d) == (1, 50)
            @test all(samples_1d .>= 0.0) && all(samples_1d .<= 10.0)
            
            # Test reproducibility with seed
            Random.seed!(42)
            samples1 = EmulatorsTrainer.create_training_dataset(10, [0.0, 0.0], [1.0, 1.0])
            Random.seed!(42)
            samples2 = EmulatorsTrainer.create_training_dataset(10, [0.0, 0.0], [1.0, 1.0])
            @test samples1 ≈ samples2
        end
        
        @testset "Input validation" begin
            # Test n <= 0
            @test_throws ArgumentError EmulatorsTrainer.create_training_dataset(0, [0.0], [1.0])
            @test_throws ArgumentError EmulatorsTrainer.create_training_dataset(-5, [0.0], [1.0])
            
            # Test mismatched lengths
            @test_throws ArgumentError EmulatorsTrainer.create_training_dataset(10, [0.0, 1.0], [1.0])
            @test_throws ArgumentError EmulatorsTrainer.create_training_dataset(10, [0.0], [1.0, 2.0])
            
            # Test empty bounds
            @test_throws ArgumentError EmulatorsTrainer.create_training_dataset(10, [], [])
            
            # Test invalid bounds (lb >= ub)
            @test_throws ArgumentError EmulatorsTrainer.create_training_dataset(10, [1.0], [0.0])
            @test_throws ArgumentError EmulatorsTrainer.create_training_dataset(10, [1.0], [1.0])
            @test_throws ArgumentError EmulatorsTrainer.create_training_dataset(10, [0.0, 2.0], [1.0, 1.5])
            
            # Test non-finite bounds
            @test_throws ArgumentError EmulatorsTrainer.create_training_dataset(10, [NaN], [1.0])
            @test_throws ArgumentError EmulatorsTrainer.create_training_dataset(10, [0.0], [Inf])
            @test_throws ArgumentError EmulatorsTrainer.create_training_dataset(10, [-Inf], [0.0])
        end
    end
    
    @testset "create_training_dict" begin
        training_matrix = [1.0 2.0 3.0;
                          4.0 5.0 6.0;
                          7.0 8.0 9.0]
        params = ["a", "b", "c"]
        
        # Test first combination
        dict1 = EmulatorsTrainer.create_training_dict(training_matrix, 1, params)
        @test dict1["a"] == 1.0
        @test dict1["b"] == 4.0
        @test dict1["c"] == 7.0
        
        # Test second combination
        dict2 = EmulatorsTrainer.create_training_dict(training_matrix, 2, params)
        @test dict2["a"] == 2.0
        @test dict2["b"] == 5.0
        @test dict2["c"] == 8.0
        
        # Test with single parameter
        single_matrix = reshape([1.0, 2.0, 3.0], 1, 3)
        single_param = ["only_param"]
        dict_single = EmulatorsTrainer.create_training_dict(single_matrix, 2, single_param)
        @test dict_single["only_param"] == 2.0
    end
    
    @testset "prepare_dataset_directory" begin
        # Use temporary directory for tests
        test_dir = mktempdir()
        
        @testset "New directory creation" begin
            new_dir = joinpath(test_dir, "new_dataset")
            result = EmulatorsTrainer.prepare_dataset_directory(new_dir; force=false)
            
            @test isdir(new_dir)
            @test result == new_dir
            
            # Check metadata file exists
            metadata_file = joinpath(new_dir, ".dataset_metadata.json")
            @test isfile(metadata_file)
            
            # Check metadata content
            metadata = JSON3.read(read(metadata_file, String))
            @test haskey(metadata, "created_at")
            @test haskey(metadata, "julia_version")
            @test metadata["julia_version"] == string(VERSION)
        end
        
        @testset "Existing directory - error mode" begin
            existing_dir = joinpath(test_dir, "existing_dataset")
            mkdir(existing_dir)
            
            # Should throw error when directory exists and force=false
            @test_throws ErrorException EmulatorsTrainer.prepare_dataset_directory(existing_dir; force=false)
            
            # Directory should still exist
            @test isdir(existing_dir)
        end
        
        @testset "Existing directory - force mode" begin
            force_dir = joinpath(test_dir, "force_dataset")
            mkdir(force_dir)
            
            # Create a file in the directory to verify backup
            test_file = joinpath(force_dir, "test.txt")
            write(test_file, "test content")
            
            # Use force=true to backup and recreate
            result = EmulatorsTrainer.prepare_dataset_directory(force_dir; force=true)
            
            @test isdir(force_dir)
            @test result == force_dir
            
            # Original file should not exist in new directory
            @test !isfile(test_file)
            
            # Check that a backup was created
            backup_dirs = filter(x -> startswith(x, "force_dataset_backup_"), readdir(test_dir))
            @test length(backup_dirs) == 1
            
            # Verify backup contains the original file
            backup_file = joinpath(test_dir, backup_dirs[1], "test.txt")
            @test isfile(backup_file)
            @test read(backup_file, String) == "test content"
        end
        
        # Cleanup
        rm(test_dir, recursive=true)
    end
    
    @testset "compute_dataset" begin
        # Setup
        test_dir = mktempdir()
        
        @testset "Input validation" begin
            training_matrix = [1.0 2.0; 3.0 4.0]
            params = ["a", "b"]
            
            # Dummy function that does nothing
            dummy_func = (dict, dir) -> nothing
            
            # Test parameter count mismatch
            wrong_params = ["a", "b", "c"]
            @test_throws ArgumentError EmulatorsTrainer.compute_dataset(
                training_matrix, wrong_params, joinpath(test_dir, "test1"), dummy_func, :serial
            )
            
            # Test empty combinations
            empty_matrix = zeros(2, 0)
            @test_throws ArgumentError EmulatorsTrainer.compute_dataset(
                empty_matrix, params, joinpath(test_dir, "test2"), dummy_func, :serial
            )
            
            # Test empty parameter names
            @test_throws ArgumentError EmulatorsTrainer.compute_dataset(
                training_matrix, String[], joinpath(test_dir, "test3"), dummy_func, :serial
            )
            
            # Test empty string in parameter names
            params_with_empty = ["a", ""]
            @test_throws ArgumentError EmulatorsTrainer.compute_dataset(
                training_matrix, params_with_empty, joinpath(test_dir, "test4"), dummy_func, :serial
            )
            
            # Test duplicate parameter names
            duplicate_params = ["a", "a"]
            @test_throws ArgumentError EmulatorsTrainer.compute_dataset(
                training_matrix, duplicate_params, joinpath(test_dir, "test5"), dummy_func, :serial
            )
            
            # Test non-finite values in matrix
            matrix_with_nan = [1.0 NaN; 3.0 4.0]
            @test_throws ArgumentError EmulatorsTrainer.compute_dataset(
                matrix_with_nan, params, joinpath(test_dir, "test6"), dummy_func, :serial
            )
            
            matrix_with_inf = [1.0 2.0; Inf 4.0]
            @test_throws ArgumentError EmulatorsTrainer.compute_dataset(
                matrix_with_inf, params, joinpath(test_dir, "test7"), dummy_func, :serial
            )
        end
        
        @testset "Basic functionality" begin
            # Create simple training matrix
            training_matrix = [1.0 2.0 3.0;
                              4.0 5.0 6.0]
            params = ["param1", "param2"]
            output_dir = joinpath(test_dir, "basic_test")
            
            # Counter to track function calls
            call_count = Ref(0)
            results = Dict{Int, Dict{String, Float64}}()
            
            function test_script(dict, dir)
                call_count[] += 1
                # Store the dictionary to verify correct values
                for idx in 1:3
                    if dict["param1"] == training_matrix[1, idx] && 
                       dict["param2"] == training_matrix[2, idx]
                        results[idx] = dict
                        break
                    end
                end
            end
            
            # Run compute_dataset
            result_dir = EmulatorsTrainer.compute_dataset(
                training_matrix, params, output_dir, test_script, :serial
            )
            
            # Verify directory was created
            @test isdir(output_dir)
            @test result_dir == output_dir
            
            # Verify metadata was created
            @test isfile(joinpath(output_dir, ".dataset_metadata.json"))
            
            # Note: In actual distributed execution, we can't easily test the exact
            # number of calls or order, but we can verify the structure is correct
        end
        
        @testset "Backward compatibility" begin
            # Test that old signature still works
            training_matrix = [1.0 2.0; 3.0 4.0]
            params = ["a", "b"]
            output_dir = joinpath(test_dir, "backward_compat")
            
            dummy_func = (dict, dir) -> nothing
            
            # This should work without the force parameter (defaults to false)
            result = EmulatorsTrainer.compute_dataset(
                training_matrix, params, output_dir, dummy_func, :serial
            )
            @test isdir(output_dir)
            
            # Should error when called again without force
            @test_throws ErrorException EmulatorsTrainer.compute_dataset(
                training_matrix, params, output_dir, dummy_func, :serial
            )
        end
        
        @testset "Computation modes" begin
            training_matrix = [1.0 2.0 3.0;
                              4.0 5.0 6.0]
            params = ["param1", "param2"]
            
            # Test serial mode
            @testset "Serial mode" begin
                output_dir = joinpath(test_dir, "serial_test")
                call_order = Int[]
                
                function track_serial(dict, dir)
                    # Track which combination was processed
                    for idx in 1:3
                        if dict["param1"] == training_matrix[1, idx]
                            push!(call_order, idx)
                            break
                        end
                    end
                end
                
                result_dir = EmulatorsTrainer.compute_dataset(
                    training_matrix, params, output_dir, track_serial, :serial
                )
                
                @test isdir(output_dir)
                @test result_dir == output_dir
                @test length(call_order) == 3
                # Serial execution should be in order
                @test call_order == [1, 2, 3]
            end
            
            # Test threads mode
            @testset "Threads mode" begin
                output_dir = joinpath(test_dir, "threads_test")
                calls_made = Threads.Atomic{Int}(0)
                
                function track_threads(dict, dir)
                    Threads.atomic_add!(calls_made, 1)
                    # Simulate some work
                    sleep(0.01)
                end
                
                result_dir = EmulatorsTrainer.compute_dataset(
                    training_matrix, params, output_dir, track_threads, :threads
                )
                
                @test isdir(output_dir)
                @test result_dir == output_dir
                @test calls_made[] == 3
            end
            
            # Test distributed mode explicitly
            @testset "Distributed mode explicit" begin
                output_dir = joinpath(test_dir, "distributed_test")
                
                dummy_func = (dict, dir) -> nothing
                
                result_dir = EmulatorsTrainer.compute_dataset(
                    training_matrix, params, output_dir, dummy_func, :distributed
                )
                
                @test isdir(output_dir)
                @test result_dir == output_dir
            end
            
            # Test invalid mode
            @testset "Invalid mode" begin
                output_dir = joinpath(test_dir, "invalid_mode")
                dummy_func = (dict, dir) -> nothing
                
                @test_throws ArgumentError EmulatorsTrainer.compute_dataset(
                    training_matrix, params, output_dir, dummy_func, :invalid_mode
                )
            end
        end
        
        # Cleanup
        rm(test_dir, recursive=true)
    end
    
    @testset "Integration test" begin
        if nworkers() > 1
            # Only run if we have multiple workers
            test_dir = mktempdir()
            
            # Create a small dataset
            n_samples = 10
            lb = [0.0, 0.0]
            ub = [1.0, 1.0]
            params = ["x", "y"]
            
            training_matrix = EmulatorsTrainer.create_training_dataset(n_samples, lb, ub)
            output_dir = joinpath(test_dir, "integration_test")
            
            # Create files for each combination
            function create_file(dict, dir)
                idx = round(Int, dict["x"] * 1000 + dict["y"] * 1000000)  # Create unique ID
                filename = joinpath(dir, "result_$(idx).json")
                open(filename, "w") do io
                    JSON3.write(io, dict)
                end
            end
            
            # Run the full pipeline
            result_dir = EmulatorsTrainer.compute_dataset(
                training_matrix, params, output_dir, create_file; force=false
            )
            
            # Verify files were created
            files = filter(x -> endswith(x, ".json") && startswith(x, "result_"), 
                          readdir(output_dir))
            @test length(files) == n_samples
            
            # Cleanup
            rm(test_dir, recursive=true)
        else
            @info "Skipping integration test - requires multiple workers"
        end
    end
end