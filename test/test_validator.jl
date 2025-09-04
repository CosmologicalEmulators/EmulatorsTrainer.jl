using Test
using EmulatorsTrainer
using JSON3
using NPZ
using DataFrames
using Random
using Dates

@testset "Validator Tests" begin
    
    @testset "sort_residuals" begin
        @testset "Basic functionality" begin
            residuals = [1.0 2.0; 3.0 4.0; 2.0 3.0]  # 3x2 matrix
            result = EmulatorsTrainer.sort_residuals(residuals)
            
            @test size(result) == (3, 2)
            # For 3 elements: 68% ≈ ceil(3*0.68) = 3, 95% ≈ ceil(3*0.95) = 3, 99.7% ≈ ceil(3*0.997) = 3
            # Column 1: [1.0, 3.0, 2.0] → sorted [1.0, 2.0, 3.0]
            @test result[1, 1] ≈ 3.0  # index 3 → 3.0 (68th percentile)
            @test result[2, 1] ≈ 3.0  # index 3 → 3.0 (95th percentile)
            @test result[3, 1] ≈ 3.0  # index 3 → 3.0 (99.7th percentile)
        end
        
        @testset "Custom percentiles" begin
            residuals = [1.0; 2.0; 3.0; 4.0; 5.0]  # 5 element vector
            residuals_matrix = reshape(residuals, 5, 1)
            
            # Test with custom percentiles [25, 50, 75]
            result = EmulatorsTrainer.sort_residuals(residuals_matrix; percentiles=[25.0, 50.0, 75.0])
            @test size(result) == (3, 1)
            # For 5 elements: 25% ≈ ceil(5*0.25) = 2, 50% ≈ ceil(5*0.5) = 3, 75% ≈ ceil(5*0.75) = 4
            @test result[1, 1] ≈ 2.0  # 25th percentile
            @test result[2, 1] ≈ 3.0  # 50th percentile
            @test result[3, 1] ≈ 4.0  # 75th percentile
            
            # Test with different number of percentiles
            result2 = EmulatorsTrainer.sort_residuals(residuals_matrix; percentiles=[10.0, 90.0])
            @test size(result2) == (2, 1)
            
            # Test with single percentile
            result3 = EmulatorsTrainer.sort_residuals(residuals_matrix; percentiles=[50.0])
            @test size(result3) == (1, 1)
            @test result3[1, 1] ≈ 3.0  # median
        end
        
        @testset "Percentile validation" begin
            residuals = reshape([1.0; 2.0; 3.0], 3, 1)
            
            # Test invalid percentiles
            @test_throws ArgumentError EmulatorsTrainer.sort_residuals(residuals; percentiles=Float64[])
            @test_throws ArgumentError EmulatorsTrainer.sort_residuals(residuals; percentiles=[-10.0])
            @test_throws ArgumentError EmulatorsTrainer.sort_residuals(residuals; percentiles=[101.0])
            @test_throws ArgumentError EmulatorsTrainer.sort_residuals(residuals; percentiles=[50.0, 110.0])
        end
        
        @testset "Input validation" begin
            # Test with too few elements
            small_residuals = [1.0; 2.0]  # Only 2 elements
            small_matrix = reshape(small_residuals, 2, 1)
            @test_throws ArgumentError EmulatorsTrainer.sort_residuals(small_matrix)  # Need at least 3 elements
            
            # Test with empty matrix
            empty_residuals = Matrix{Float64}(undef, 0, 5)
            @test_throws ArgumentError EmulatorsTrainer.sort_residuals(empty_residuals)
            
            # Test with no output features
            no_output = Matrix{Float64}(undef, 5, 0)
            @test_throws ArgumentError EmulatorsTrainer.sort_residuals(no_output)
        end
        
        @testset "Edge cases" begin
            # Minimum valid case: 3 elements
            residuals = [1.0; 2.0; 3.0]  # Vector needs to be reshaped to matrix
            residuals_matrix = reshape(residuals, 3, 1)
            result = EmulatorsTrainer.sort_residuals(residuals_matrix)
            @test size(result) == (3, 1)
            
            # Large array
            Random.seed!(42)
            large_residuals = rand(1000, 5)
            result = EmulatorsTrainer.sort_residuals(large_residuals)
            @test size(result) == (3, 5)
            # Check that percentiles are in ascending order
            for col in 1:5
                @test result[1, col] <= result[2, col] <= result[3, col]
            end
            
            # Test bounds safety
            for n in [3, 10, 100, 1000]
                residuals = rand(n, 2)
                result = EmulatorsTrainer.sort_residuals(residuals)
                @test all(isfinite.(result))
                @test size(result) == (3, 2)
            end
        end
    end
    
    @testset "get_single_residuals" begin
        # Create temporary test directory
        test_dir = mktempdir()
        
        try
            # Create test parameter file
            param_file = "params.json"
            param_path = joinpath(test_dir, param_file)
            test_params = Dict("omega_m" => 0.3, "sigma_8" => 0.8, "h" => 0.7)
            open(param_path, "w") do f
                JSON3.write(f, test_params)
            end
            
            # Mock functions for testing
            mock_ground_truth(loc) = [1.0, 2.0, 3.0]
            mock_emu_prediction(input) = [1.1, 2.1, 2.9]
            mock_sigma(loc) = [0.1, 0.2, 0.3]
            
            @testset "With sigma function" begin
                pars_array = ["omega_m", "sigma_8", "h"]
                result = EmulatorsTrainer.get_single_residuals(test_dir, param_file, pars_array,
                    mock_ground_truth, mock_emu_prediction, mock_sigma)
                
                expected = abs.([1.0, 2.0, 3.0] .- [1.1, 2.1, 2.9]) ./ [0.1, 0.2, 0.3]
                @test result ≈ expected
            end
            
            @testset "Without sigma function (relative error)" begin
                pars_array = ["omega_m", "sigma_8", "h"]
                result = EmulatorsTrainer.get_single_residuals(test_dir, param_file, pars_array,
                    mock_ground_truth, mock_emu_prediction)
                
                expected = 100.0 .* abs.(1.0 .- [1.1, 2.1, 2.9] ./ [1.0, 2.0, 3.0])
                @test result ≈ expected
            end
            
            @testset "Input validation" begin
                pars_array = ["omega_m", "sigma_8", "h"]
                
                # Empty location
                @test_throws ArgumentError EmulatorsTrainer.get_single_residuals("", param_file, pars_array,
                    mock_ground_truth, mock_emu_prediction)
                
                # Empty dict_file
                @test_throws ArgumentError EmulatorsTrainer.get_single_residuals(test_dir, "", pars_array,
                    mock_ground_truth, mock_emu_prediction)
                
                # Empty pars_array
                @test_throws ArgumentError EmulatorsTrainer.get_single_residuals(test_dir, param_file, String[],
                    mock_ground_truth, mock_emu_prediction)
                
                # Non-existent file
                @test_throws ArgumentError EmulatorsTrainer.get_single_residuals(test_dir, "nonexistent.json", pars_array,
                    mock_ground_truth, mock_emu_prediction)
                
                # Missing parameter in JSON
                @test_throws ArgumentError EmulatorsTrainer.get_single_residuals(test_dir, param_file, ["missing_param"],
                    mock_ground_truth, mock_emu_prediction)
            end
            
            @testset "Division by zero protection" begin
                pars_array = ["omega_m", "sigma_8", "h"]
                mock_zero_sigma(loc) = [0.0, 0.2, 0.3]
                mock_zero_gt(loc) = [0.0, 2.0, 3.0]
                
                # Zero sigma should throw error
                @test_throws ArgumentError EmulatorsTrainer.get_single_residuals(test_dir, param_file, pars_array,
                    mock_ground_truth, mock_emu_prediction, mock_zero_sigma)
                
                # Zero ground truth should throw error for relative error
                @test_throws ArgumentError EmulatorsTrainer.get_single_residuals(test_dir, param_file, pars_array,
                    mock_zero_gt, mock_emu_prediction)
            end
            
        finally
            rm(test_dir, recursive=true)
        end
    end
    
    @testset "evaluate_residuals" begin
        # Create temporary test directory structure
        test_dir = mktempdir()
        
        try
            # Create subdirectories with JSON files
            for i in 1:3
                subdir = joinpath(test_dir, "run_$i")
                mkdir(subdir)
                
                # Create parameter file
                param_file = "params.json"
                param_path = joinpath(subdir, param_file)
                test_params = Dict("omega_m" => 0.3 + i*0.1, "sigma_8" => 0.8 + i*0.05)
                open(param_path, "w") do f
                    JSON3.write(f, test_params)
                end
            end
            
            # Mock functions
            mock_ground_truth(loc) = [1.0, 2.0]
            mock_emu_prediction(input) = [input[1], input[2]]  # Simple echo
            mock_sigma(loc) = [0.1, 0.1]
            
            @testset "Basic functionality with sigma" begin
                pars_array = ["omega_m", "sigma_8"]
                result = EmulatorsTrainer.evaluate_residuals(test_dir, "params.json", pars_array,
                    mock_ground_truth, mock_emu_prediction; get_σ=mock_sigma)
                
                @test size(result) == (3, 2)
                @test all(isfinite.(result))
            end
            
            @testset "Basic functionality without sigma" begin
                pars_array = ["omega_m", "sigma_8"]
                result = EmulatorsTrainer.evaluate_residuals(test_dir, "params.json", pars_array,
                    mock_ground_truth, mock_emu_prediction)
                
                @test size(result) == (3, 2)
                @test all(isfinite.(result))
            end
            
            @testset "Auto-infer dimensions" begin
                pars_array = ["omega_m", "sigma_8"]
                
                # Test auto-detection
                result = EmulatorsTrainer.evaluate_residuals(test_dir, "params.json", pars_array,
                    mock_ground_truth, mock_emu_prediction)
                
                @test size(result) == (3, 2)  # Should find 3 JSON files, 2 output features
                @test all(isfinite.(result))
                
                # Test with get_σ
                result_with_sigma = EmulatorsTrainer.evaluate_residuals(test_dir, "params.json", pars_array,
                    mock_ground_truth, mock_emu_prediction; get_σ=mock_sigma)
                
                @test size(result_with_sigma) == (3, 2)
                @test all(isfinite.(result_with_sigma))
            end
            
            @testset "Input validation" begin
                pars_array = ["omega_m", "sigma_8"]
                
                # Non-existent directory
                @test_throws ArgumentError EmulatorsTrainer.evaluate_residuals("/nonexistent", "params.json", pars_array,
                    mock_ground_truth, mock_emu_prediction)
                
                
                # Empty pars_array
                @test_throws ArgumentError EmulatorsTrainer.evaluate_residuals(test_dir, "params.json", String[],
                    mock_ground_truth, mock_emu_prediction)
                
                # Empty dict_file
                @test_throws ArgumentError EmulatorsTrainer.evaluate_residuals(test_dir, "", pars_array,
                    mock_ground_truth, mock_emu_prediction)
            end
            
            @testset "Auto-detect from available files" begin
                pars_array = ["omega_m", "sigma_8"]
                
                # Should always return what's available (3 combinations, 2 features)
                result = EmulatorsTrainer.evaluate_residuals(test_dir, "params.json", pars_array,
                    mock_ground_truth, mock_emu_prediction)
                
                @test size(result, 1) == 3
                @test size(result, 2) == 2
            end
            
        finally
            rm(test_dir, recursive=true)
        end
    end
    
    @testset "evaluate_sorted_residuals" begin
        # Create temporary test directory
        test_dir = mktempdir()
        
        try
            # Create test data
            for i in 1:5
                subdir = joinpath(test_dir, "run_$i")
                mkdir(subdir)
                
                param_file = "params.json"
                param_path = joinpath(subdir, param_file)
                test_params = Dict("omega_m" => 0.2 + i*0.1)
                open(param_path, "w") do f
                    JSON3.write(f, test_params)
                end
            end
            
            # Mock functions
            mock_ground_truth(loc) = [1.0]
            mock_emu_prediction(input) = [input[1] + 0.1]  # Constant offset
            
            @testset "Basic functionality" begin
                pars_array = ["omega_m"]
                result = EmulatorsTrainer.evaluate_sorted_residuals(test_dir, "params.json", pars_array,
                    mock_ground_truth, mock_emu_prediction)
                
                @test size(result) == (3, 1)  # 3 percentiles, 1 output feature
                @test all(isfinite.(result))
                # Should be sorted (ascending percentiles)
                @test result[1, 1] <= result[2, 1] <= result[3, 1]
            end
            
            @testset "With sigma function" begin
                pars_array = ["omega_m"]
                mock_sigma(loc) = [0.05]
                
                result = EmulatorsTrainer.evaluate_sorted_residuals(test_dir, "params.json", pars_array,
                    mock_ground_truth, mock_emu_prediction; get_σ=mock_sigma)
                
                @test size(result) == (3, 1)
                @test all(isfinite.(result))
            end
            
            @testset "Handles dynamic sizing" begin
                pars_array = ["omega_m"]
                
                # Should handle available files automatically
                result = EmulatorsTrainer.evaluate_sorted_residuals(test_dir, "params.json", pars_array,
                    mock_ground_truth, mock_emu_prediction)
                
                @test size(result, 2) == 1  # Should still return 1 output feature
                @test size(result, 1) == 3  # Should still return 3 percentiles
            end
            
            @testset "Custom percentiles" begin
                pars_array = ["omega_m"]
                
                # Test with custom percentiles
                result = EmulatorsTrainer.evaluate_sorted_residuals(test_dir, "params.json", pars_array,
                    mock_ground_truth, mock_emu_prediction; percentiles=[25.0, 50.0, 75.0])
                
                @test size(result) == (3, 1)  # 3 percentiles requested
                @test all(isfinite.(result))
            end
            
            @testset "Auto-inference with custom percentiles" begin
                pars_array = ["omega_m"]
                
                # Test with custom percentiles
                result = EmulatorsTrainer.evaluate_sorted_residuals(test_dir, "params.json", pars_array,
                    mock_ground_truth, mock_emu_prediction; percentiles=[10.0, 50.0, 90.0])
                
                @test size(result) == (3, 1)  # 3 percentiles requested
                @test all(isfinite.(result))
            end
            
            @testset "Default percentiles" begin
                pars_array = ["omega_m"]
                
                # Test with default percentiles
                result = EmulatorsTrainer.evaluate_sorted_residuals(test_dir, "params.json", pars_array,
                    mock_ground_truth, mock_emu_prediction)
                
                @test size(result) == (3, 1)  # Should return default 3 percentiles
                @test all(isfinite.(result))
            end
            
        finally
            rm(test_dir, recursive=true)
        end
    end
    
    @testset "Integration test" begin
        # Test the complete workflow
        test_dir = mktempdir()
        
        try
            # Create realistic test data
            Random.seed!(42)
            n_samples = 20
            
            for i in 1:n_samples
                subdir = joinpath(test_dir, "sample_$i")
                mkdir(subdir)
                
                # Create parameter file with realistic cosmological parameters
                param_file = "cosmology.json"
                param_path = joinpath(subdir, param_file)
                test_params = Dict(
                    "omega_m" => 0.15 + rand() * 0.5,  # 0.15-0.65
                    "sigma_8" => 0.6 + rand() * 0.4,   # 0.6-1.0
                    "h" => 0.6 + rand() * 0.2          # 0.6-0.8
                )
                open(param_path, "w") do f
                    JSON3.write(f, test_params)
                end
            end
            
            # Realistic mock functions
            function realistic_ground_truth(loc)
                # Simulate power spectrum values
                return rand(10) * 1000  # Mock power spectrum
            end
            
            function realistic_emu_prediction(input)
                # Simple polynomial emulator
                om, s8, h = input
                return rand(10) * 1000 * (1 + 0.1 * (om - 0.3) + 0.05 * (s8 - 0.8))
            end
            
            function realistic_sigma(loc)
                return ones(10) * 10  # 1% errors
            end
            
            @testset "Complete workflow" begin
                pars_array = ["omega_m", "sigma_8", "h"]
                
                # Test evaluate_residuals
                residuals = EmulatorsTrainer.evaluate_residuals(test_dir, "cosmology.json", pars_array,
                    realistic_ground_truth, realistic_emu_prediction; get_σ=realistic_sigma)
                
                @test size(residuals) == (n_samples, 10)
                @test all(isfinite.(residuals))
                @test all(residuals .>= 0)  # Residuals should be non-negative
                
                # Test sort_residuals
                sorted_res = EmulatorsTrainer.sort_residuals(residuals)
                @test size(sorted_res) == (3, 10)
                @test all(isfinite.(sorted_res))
                
                # Test evaluate_sorted_residuals (integrated)
                integrated_result = EmulatorsTrainer.evaluate_sorted_residuals(test_dir, "cosmology.json", pars_array,
                    realistic_ground_truth, realistic_emu_prediction; get_σ=realistic_sigma)
                
                @test size(integrated_result) == (3, 10)
                @test all(isfinite.(integrated_result))
                # Note: Results may not be exactly equal due to dynamic file processing order
            end
            
        finally
            rm(test_dir, recursive=true)
        end
    end
end