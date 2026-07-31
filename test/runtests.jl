using Test
using EmulatorsTrainer

@testset "EmulatorsTrainer.jl" begin
    include("test_farmer.jl")
    include("test_hdf5.jl")
    include("test_trainer.jl")
    include("test_validator.jl")
    include("test_simplechains.jl")
end
