
@testset "MCMC initial values" begin
    import GriddedBlockMaxima: compute_initial_values, set_default_initial_values

    y = [[1.0, 3.0, 5.0 + j^2] for j in 1:16]
    F₁ = GMRF.iGMRF(4, 4; order=1, precision=1.0)
    F₂ = GMRF.iGMRF(4, 4; order=2, precision=1.0)

    # Default initialization
    d = set_default_initial_values(y, F₁)
    @test d.μ == d.ϕ == zeros(16)
    @test d.μ !== d.ϕ
    @test (d.ξ, d.κ₁, d.κ₂) == (0.0, 1.0, 1.0)
    @test d.loglike ≈ [loglikelihood(Gumbel(), obs) for obs in y]

    # Cellwise PWM estimates and corresponding log-likelihoods
    fits = [Extremes.gumbelfitpwm(obs).θ̂ for obs in y]
    μ = first.(fits)
    ϕ = last.(fits)
    loglike = [loglikelihood(Gumbel(μ[j], exp(ϕ[j])), y[j]) for j in eachindex(y)]

    for F in (F₁, F₂)
        s = compute_initial_values(y, F)
        @test s.ξ == 0.0
        @test s.μ ≈ μ
        @test s.ϕ ≈ ϕ
        @test s.loglike ≈ loglike

        rank_deficiency = F.order == 1 ? 1 : 3
        rank_W = F.G.m₁ * F.G.m₂ - rank_deficiency
        shape = 1 + rank_W / 2
    
        @test s.κ₁ ≈ shape / (1 + dot(μ, F.W * μ) / 2)
        @test s.κ₂ ≈ shape / (1 + dot(ϕ, F.W * ϕ) / 2)

        s₂ = compute_initial_values(y, F)
        @test (s₂.κ₁, s₂.κ₂) == (s.κ₁, s.κ₂)
    end

    @test_throws DimensionMismatch compute_initial_values(y[2:end], F₁)
    @test_throws DimensionMismatch set_default_initial_values(y[2:end], F₁)

    # A constant cell admits the default initialization but has zero PWM scale.
    y[1] = fill(3.0, 4)
    @test_throws ArgumentError compute_initial_values(y, F₁)
    @test isfinite(set_default_initial_values(y, F₁).loglike[1])
end

@testset "MCMC proposals" begin
    import GriddedBlockMaxima: CurrentStates, MHSampler, propose_location_candidates, propose_logscale_candidates, propose_shape_candidates

    θ = CurrentStates([10.0, 20.0], [-0.4, 0.2], 0.1, 2.0, 3.0, [-4.0, -5.0])
    s = MHSampler([0.2, 0.5], [0.03, 0.07], 0.01, zeros(2), zeros(2))
    μ₀, ϕ₀, ξ₀ = copy(θ.μ), copy(θ.ϕ), θ.ξ

    ref = Random.Xoshiro(42)
    expected_μ = μ₀ .+ s.stepsize_mu .* randn!(ref, zeros(2))
    expected_ϕ = ϕ₀ .+ s.stepsize_phi .* randn!(ref, zeros(2))
    expected_ξ = ξ₀ + s.stepsize_xi * randn(ref)

    rng = Random.Xoshiro(42)
    μ̃ = propose_location_candidates(θ, s; rng=rng)
    ϕ̃ = propose_logscale_candidates(θ, s; rng=rng)
    ξ̃ = propose_shape_candidates(θ, s; rng=rng)

    # Reusing the work vector must preserve both proposals.
    fill!(s.δ, NaN)
    @test μ̃ ≈ expected_μ
    @test ϕ̃ ≈ expected_ϕ
    @test ξ̃ ≈ expected_ξ
    @test (θ.μ, θ.ϕ, θ.ξ) == (μ₀, ϕ₀, ξ₀)
    @test randn(rng) == randn(ref)

    one_cell = MHSampler([0.2], [0.03], 0.01, zeros(1), zeros(1))
    @test_throws DimensionMismatch propose_location_candidates(θ, one_cell)
    @test_throws DimensionMismatch propose_logscale_candidates(θ, one_cell)
end

@testset "sample_gev_parameters" begin

    import GriddedBlockMaxima: CurrentStates, gev_loglikelihood, initialize_sampler, sample_location!, sample_logscale!, sample_shape!

    # Fix both draws so the acceptance decision at each cell is known.
    struct TestRNG <: Random.AbstractRNG end
    Random.randn!(::TestRNG, z::Vector{Float64}) = fill!(z, 1.0)
    Random.rand!(::TestRNG, u::Vector{Float64}) = fill!(u, 0.5)
    Random.randn(::TestRNG) = 1.0
    Random.rand(::TestRNG) = .5


    F = GMRF.iGMRF(2, 2; order=1, precision=1.0)
    m = size(F.W, 1)

    sampler = initialize_sampler(F; stepsize_location=ones(m),stepsize_logscale = ones(m), stepsize_shape = 1.)

    y = [fill(1., 5) for _ in 1:m]
    θ = CurrentStates(zeros(m), zeros(m), 0., 1.0, 1.0, gev_loglikelihood.(y, 0., 0., 0.))

    # Every candidates should be accepted
    sample_location!(θ, sampler, y, F; rng = TestRNG())
    @test θ.μ ≈ ones(m)

    # Every candidates should be refused
    sample_logscale!(θ, sampler, y, F; rng = TestRNG())
    @test θ.ϕ ≈ zeros(m)

    # The candidate should be refused
    sample_shape!(θ, sampler, y, F; rng = TestRNG())
    θ.ξ ≈ 0.

end

@testset "sample_precision!" begin
    import GriddedBlockMaxima.sample_precision!
    import LinearAlgebra.rank

    for order in (1, 2)
        F = GMRF.iGMRF(5, 5; order=order, precision=1.0)
        m = size(F.W, 1)
        j = 13  # Center of the 5 × 5 grid

        μ, ϕ = zeros(m), zeros(m)
        μ[j], ϕ[j] = 2.0, 1.0
        loglike = fill(-3.0, m)
        θ = CurrentStates(copy(μ), copy(ϕ), 0.1, 1.0, 1.0, copy(loglike))

        shape = 1 + rank(Matrix(F.W)) / 2
        w = F.W[j, j]
        ref_rng = Random.MersenneTwister(42)
        expected_κ₁ = rand(ref_rng, Gamma(shape, 1 / (1 + 2 * w)))
        expected_κ₂ = rand(ref_rng, Gamma(shape, 1 / (1 + w / 2)))

        @test sample_precision!(θ, F; rng=Random.MersenneTwister(42)) === nothing
        @test θ.κ₁ ≈ expected_κ₁
        @test θ.κ₂ ≈ expected_κ₂
        @test (θ.μ, θ.ϕ, θ.ξ, θ.loglike) == (μ, ϕ, 0.1, loglike)
    end
end

using GriddedBlockMaxima: run_mcmc!, set_default_initial_values, initialize_sampler,
                          sample_location!, sample_logscale!, sample_shape!, sample_precision!,
                          changerate, update_stepsize

@testset "run_mcmc!" begin

    import GriddedBlockMaxima: run_mcmc!, set_default_initial_values, initialize_sampler,
                          sample_location!, sample_logscale!, sample_shape!, sample_precision!,
                          changerate, update_stepsize

    F = GMRF.iGMRF(3, 3; order=1, precision=1.0)
    m = size(F.W, 1)
    y = [[0.2, 0.7, 1.5] for _ in 1:m]
    θ_ref = set_default_initial_values(y, F)
    sampler_ref = initialize_sampler(F; stepsize_logscale=fill(1.5, m))
    θ, sampler = deepcopy(θ_ref), deepcopy(sampler_ref)

    chain = run_mcmc!(θ, sampler, y, F; niter=5, warmup=2, batchsize=2,
                      rng=Random.MersenneTwister(42))

    state_parameters(s) = (s.μ, s.ϕ, s.ξ, s.κ₁, s.κ₂)
    stored(i) = (chain.μ[:, i], chain.ϕ[:, i], chain.ξ[i], chain.κ₁[i], chain.κ₂[i])
    @test size(chain.μ) == size(chain.ϕ) == (m, 5)
    @test length(chain.ξ) == length(chain.κ₁) == length(chain.κ₂) == 5
    @test stored(1) == state_parameters(θ_ref)
    @test stored(5) == state_parameters(θ)

    # Replay the first sweep with the same RNG and the tested kernels.
    rng = Random.MersenneTwister(42)
    sample_location!(θ_ref, sampler_ref, y, F; rng=rng)
    sample_logscale!(θ_ref, sampler_ref, y, F; rng=rng)
    sample_shape!(θ_ref, sampler_ref, y, F; rng=rng)
    sample_precision!(θ_ref, F; rng=rng)
    @test stored(2) == state_parameters(θ_ref)

    # Only the batch ending at sweep 2 adapts the proposal step sizes.
    rate_mu = changerate(chain.μ[:, 1:3])
    rate_phi = changerate(chain.ϕ[:, 1:3])
    rate_xi = changerate(chain.ξ[1:3])
    @test sampler.stepsize_mu ≈ update_stepsize.(sampler_ref.stepsize_mu, rate_mu)
    @test sampler.stepsize_phi ≈ update_stepsize.(sampler_ref.stepsize_phi, rate_phi)
    @test sampler.stepsize_xi ≈ update_stepsize(sampler_ref.stepsize_xi, rate_xi)
end


@testset "changerate" begin
    import GriddedBlockMaxima.changerate

    @test changerate([1, 1, 2, 2]) == 1 / 3
    @test changerate([1.0, nextfloat(1.0)]) == 1.0

    X = [2 2 2
         1 2 3]
    @test changerate(X) == [0.0, 1.0]

    @test_throws ArgumentError changerate(Float64[])
    @test_throws ArgumentError changerate([1.0])
    @test_throws ArgumentError changerate(zeros(2, 1))
end

@testset "update_stepsize" begin
    import GriddedBlockMaxima.update_stepsize

    δ = 2.0

    @test update_stepsize(δ, 1.0) ≈ δ * exp(0.01)
    @test update_stepsize(δ, 0.0; logstep=log(2)) ≈ δ / 2
    @test update_stepsize(δ, 0.44) == δ
    @test update_stepsize(δ, 0.44f0) == δ
    @test update_stepsize(δ, 1.0; logstep=0) == δ

    @test_throws DomainError update_stepsize(0.0, 0.5)
    @test_throws DomainError update_stepsize(Inf, 0.5)
    for rate in (-0.1, 1.1, NaN)
        @test_throws DomainError update_stepsize(δ, rate)
    end
    @test_throws DomainError update_stepsize(δ, 0.5; logstep=-0.01)
    @test_throws DomainError update_stepsize(δ, 0.5; logstep=Inf)
end