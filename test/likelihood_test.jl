
@testset "gev_loglikelihood" begin
    import GriddedBlockMaxima.gev_loglikelihood 

    y = [8.0, 10.0, 14.0]
    μ, σ = 10.0, 2.0
    ϕ = log(σ)

    # GEV de forme négative ou positive
    for ξ in (-0.25, 0.25)
        d = GeneralizedExtremeValue(μ, σ, ξ)
        expected = sum(logpdf(d, x) for x in y)
        @test gev_loglikelihood(y, μ, ϕ, ξ) ≈ expected
    end

    # Pour ξ = 0
    d = Gumbel(μ, σ)
    @test gev_loglikelihood(y, μ, ϕ, 0.0) ≈ sum(logpdf(d, x) for x in y)

    # Une seule observation en dehors du support
    @test gev_loglikelihood([10.0, 19.0], μ, ϕ, -0.25) == -Inf
end