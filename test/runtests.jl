using GriddedBlockMaxima, Test
using Distributions, Extremes, GMRF, LinearAlgebra, Random

@testset "GriddedBlockMaxima.jl" begin
    include("types_test.jl")
    include("data_generation_test.jl")
    include("likelihood_test.jl")
    include("mcmc_test.jl")
    include("utils_test.jl")    
end
