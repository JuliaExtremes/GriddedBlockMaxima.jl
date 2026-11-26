
@testset "CurrentStates" begin
    import GriddedBlockMaxima.CurrentStates

    μ = [10.0, 12.0]
    ϕ = [0.2, 0.3]
    ll = [-2.0, -3.0]
    state = CurrentStates(μ, ϕ, 0.1, 2.0, 3.0, ll)

    @test_throws DimensionMismatch CurrentStates(μ, ϕ[1:1], 0.1, 2.0, 3.0, ll)
    @test_throws DimensionMismatch CurrentStates(μ, ϕ, 0.1, 2.0, 3.0, ll[1:1])

    compact = repr(state)
    @test occursin("2 grid cells", compact) && !occursin('\n', compact)

    state.κ₁ = 4.0
    detail = repr("text/plain", state)
    @test occursin("κ₁ = 4.0", detail)
    @test occursin("loglike", detail)

    indented = sprint(io -> show(io, state; prefix="> "))
    @test split(indented, '\n') == ["> " * line for line in split(detail, '\n')]
end

@testset "MHSampler" begin
    import GriddedBlockMaxima.MHSampler

    μsteps = [0.1, 0.2]
    ϕsteps = [0.3, 0.4]
    δ, u = zeros(2), zeros(2)
    s = MHSampler(μsteps, ϕsteps, 0.05, δ, u)

    @test (s.stepsize_mu, s.stepsize_phi, s.stepsize_xi) == (μsteps, ϕsteps, 0.05)

    s.stepsize_xi = 0.08
    @test repr(s) == "MHSampler (2 grid cells)"
    detail = repr("text/plain", s)
    @test occursin("stepsize_xi = 0.08", detail)

    indented = sprint(io -> show(io, s; prefix="> "))
    @test split(indented, '\n') == ["> " * line for line in split(detail, '\n')]

    @test_throws DimensionMismatch MHSampler(μsteps, ϕsteps, 0.05, δ, zeros(1))
    @test_throws ArgumentError MHSampler(Float64[], Float64[], 0.05, Float64[], Float64[])
    @test_throws ArgumentError MHSampler([0.0, 0.2], ϕsteps, 0.05, δ, u)
    @test_throws ArgumentError MHSampler(μsteps, [0.3, Inf], 0.05, δ, u)
    @test_throws ArgumentError MHSampler(μsteps, ϕsteps, NaN, δ, u)
end