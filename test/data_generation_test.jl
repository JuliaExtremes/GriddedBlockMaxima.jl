
@testset "generate_gridded_parameters" begin
    import GriddedBlockMaxima.generate_gridded_parameters
    import Random.Xoshiro
    
    F = GMRF.iGMRF(3, 4; order=1, precision=1.0)
    opts = (ξ=0.1, location_precision=2.0, logscale_precision=8.0)
    p = generate_gridded_parameters(F; opts..., rng=Xoshiro(42))

    @test p[:F] === F
    @test (p[:ξ], p[:κ₁], p[:κ₂]) == (0.1, 2.0, 8.0)
    @test size(p[:μ]) == size(p[:ϕ]) == (12,)

    rng = Xoshiro(42)
    @test p[:μ] ≈ rand(rng, GMRF.set_precision(F, 2.0))
    @test p[:ϕ] ≈ rand(rng, GMRF.set_precision(F, 8.0))

    for ξ in (-0.5, 0.5)
        @test_throws ArgumentError generate_gridded_parameters(F; opts..., ξ=ξ)
    end
    @test_throws ArgumentError generate_gridded_parameters(F; opts..., location_precision=Inf)
    @test_throws ArgumentError generate_gridded_parameters(F; opts..., logscale_precision=0.0)
end

@testset "generate_gridded_blockmaxima" begin
    import GriddedBlockMaxima: generate_gridded_parameters, generate_gridded_blockmaxima
    import Random.Xoshiro

    p = Dict(:μ => [10.0, 20.0], :ϕ => log.([2.0, 4.0]), :ξ => -0.2)
    y = generate_gridded_blockmaxima(p, 3; rng=Xoshiro(42))

    rng = Xoshiro(42)
    @test length.(y) == [3, 3]
    @test y[1] ≈ rand(rng, GeneralizedExtremeValue(10.0, 2.0, -0.2), 3)
    @test y[2] ≈ rand(rng, GeneralizedExtremeValue(20.0, 4.0, -0.2), 3)

    @test generate_gridded_blockmaxima(p, 0) == [Float64[], Float64[]]
    @test_throws ArgumentError generate_gridded_blockmaxima(p, -1)

    p[:ϕ] = [0.0]
    @test_throws DimensionMismatch generate_gridded_blockmaxima(p, 3)
end