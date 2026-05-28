module FiVoHydro

# Thin package wrapper around the existing script entrypoint.
# This makes the repo a valid Julia project/package (for precompilation, CI, tests)
# while keeping the current `main.jl` + `module hydro` structure intact.

include(joinpath(@__DIR__, "..", "main.jl"))

# Re-export the solver module for convenience.
export hydro

end # module FiVoHydro
