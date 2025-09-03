using Test
using EmulatorsTrainer

@testset "EmulatorsTrainer.jl" begin
    include("test_farmer.jl")
    include("test_trainer.jl")
    # Future test files can be added here:
    # include("test_validator.jl")
end