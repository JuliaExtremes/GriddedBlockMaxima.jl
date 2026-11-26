module GriddedBlockMaxima

    using Distributions, Extremes, MambaLite
    using GMRF, LinearAlgebra, Random, SparseArrays
    using ProgressMeter

    import Random.default_rng

    include("types.jl")
    include("data_generation.jl")
    include("likelihood.jl")
    include("mcmc.jl")
    include("utils.jl")

end
