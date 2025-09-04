using Documenter
using EmulatorsTrainer

ENV["GKSwstype"] = "100"

push!(LOAD_PATH,"../src/")

makedocs(
    modules = [EmulatorsTrainer],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true",
    sidebar_sitename=false),
    sitename = "EmulatorsTrainer.jl",
    authors  = "Marco Bonici",
    pages = [
        "Home" => "index.md",
        "API Reference" => "api.md",
    ],
    checkdocs = :exports  # Only check exported functions
)

deploydocs(
    repo = "github.com/CosmologicalEmulators/EmulatorsTrainer.jl.git",
    devbranch = "develop"
)
