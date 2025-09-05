using Test
using EmulatorsTrainer

@testset "EmulatorsTrainer.jl" begin
    include("test_farmer.jl")
    include("test_trainer.jl")
    include("test_validator.jl")
end