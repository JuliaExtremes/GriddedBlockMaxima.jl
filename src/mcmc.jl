
## Initial values

function _initial_loglikelihood(y, μ, ϕ, ξ)
    loglike = gev_loglikelihood.(y, μ, ϕ, ξ)
    j = findfirst(x -> !isfinite(x), loglike)
    isnothing(j) || throw(ArgumentError("non-finite initial log-likelihood at cell $j"))
    return loglike
end

"""
    compute_initial_values(y, F)

Initialize the MCMC state with a Gumbel PWM fit at each grid cell.

Set the common shape to zero and initialize both field precisions at their
conditional means under independent Gamma(1, 1) priors.
"""
function compute_initial_values(y::Vector{<:Vector{<:Real}}, F::GMRF.iGMRF)
    m = size(F.W, 1)
    length(y) == m || throw(DimensionMismatch("expected $m grid cells, got $(length(y))"))
    F.order in (1, 2) || throw(ArgumentError("unsupported iGMRF order $(F.order)"))

    μ = Vector{Float64}(undef, m)
    ϕ = similar(μ)
    ξ = 0.0

    for (j, obs) in enumerate(y)
        valid = length(obs) >= 2 && all(isfinite, obs) && minimum(obs) < maximum(obs)
        valid || throw(ArgumentError("cell $j needs at least two distinct, finite observations"))

        μ[j], ϕ[j] = Extremes.gumbelfitpwm(obs).θ̂
    end

    loglike = _initial_loglikelihood(y, μ, ϕ, ξ)

    rank_deficiency = F.order == 1 ? 1 : 3
    rank_W = m - rank_deficiency
    shape = 1 + rank_W / 2

    location_rate = 1 + dot(μ, F.W, μ) / 2
    logscale_rate = 1 + dot(ϕ, F.W, ϕ) / 2

    κ₁ = shape / location_rate
    κ₂ = shape / logscale_rate

    return CurrentStates(μ, ϕ, ξ, κ₁, κ₂, loglike)
end

"""
    set_default_initial_values(y, F)

Return an MCMC state with a standard Gumbel distribution at each grid cell,
unit field precisions, and the corresponding cellwise log-likelihoods.
"""
function set_default_initial_values(y::Vector{<:Vector{<:Real}}, F::GMRF.iGMRF)
    m = size(F.W, 1)
    length(y) == m || throw(DimensionMismatch("expected $m grid cells, got $(length(y))"))

    μ = zeros(m)
    ϕ = zeros(m)
    ξ = 0.0
    loglike = _initial_loglikelihood(y, μ, ϕ, ξ)

    return CurrentStates(μ, ϕ, ξ, 1.0, 1.0, loglike)
end

## MCMC initialization

"""
    initialize_chain(initial_values::CurrentStates; niter)

Allocate storage for `niter` MCMC iterations, including the initial state.

Rows of `μ` and `ϕ` correspond to grid cells and columns to iterations.
Only the first iteration is initialized.
"""
function initialize_chain(initial_values::CurrentStates; niter::Int)
    niter >= 1 || throw(ArgumentError("niter must be at least 1, got $niter"))

    m = length(initial_values.μ)
    μ = Matrix{Float64}(undef, m, niter)
    ϕ = similar(μ)
    ξ = Vector{Float64}(undef, niter)
    κ₁ = similar(ξ)
    κ₂ = similar(ξ)

    μ[:, 1] = initial_values.μ
    ϕ[:, 1] = initial_values.ϕ
    ξ[1] = initial_values.ξ
    κ₁[1] = initial_values.κ₁
    κ₂[1] = initial_values.κ₂

    return (; μ, ϕ, ξ, κ₁, κ₂)
end

"""
    initialize_sampler(F; stepsize_location=fill(0.05, m),
                       stepsize_logscale=fill(0.005, m), stepsize_shape=0.005)

Create a Metropolis–Hastings sampler for the `m` grid cells of `F`.

Copy the cellwise location and log-scale step sizes for adaptation.
Use a single step size for the common shape parameter.
"""
function initialize_sampler(
    F::GMRF.iGMRF;
    stepsize_location::AbstractVector{<:Real}=fill(0.05, size(F.W, 1)),
    stepsize_logscale::AbstractVector{<:Real}=fill(0.005, size(F.W, 1)),
    stepsize_shape::Real=0.005,
)
    m = size(F.W, 1)
    stepsize_mu = collect(Float64, stepsize_location)
    stepsize_phi = collect(Float64, stepsize_logscale)
    δ = Vector{Float64}(undef, m)
    u = similar(δ)

    return MHSampler(stepsize_mu, stepsize_phi, stepsize_shape, δ, u)
end


## Candidate proposals

"""
    propose_location_candidates(θ, sampler; rng=default_rng())

Draw Gaussian random-walk candidates for `θ.μ` using cellwise standard deviations
`sampler.stepsize_mu`. Store the standard normal draws in `sampler.δ`.
"""
function propose_location_candidates(θ::CurrentStates, sampler::MHSampler; rng::AbstractRNG=Random.default_rng())

    length(θ.μ) == length(sampler.stepsize_mu) == length(sampler.δ) ||
        throw(DimensionMismatch("location field and sampler must have the same number of cells"))

    randn!(rng, sampler.δ)
    return θ.μ .+ sampler.stepsize_mu .* sampler.δ
end

"""
    propose_logscale_candidates(θ, sampler; rng=default_rng())

Draw Gaussian random-walk candidates for `θ.ϕ` using cellwise standard deviations
`sampler.stepsize_phi`. Store the standard normal draws in `sampler.δ`.
"""
function propose_logscale_candidates(θ::CurrentStates, sampler::MHSampler; rng::AbstractRNG=Random.default_rng())

    length(θ.ϕ) == length(sampler.stepsize_phi) == length(sampler.δ) ||
        throw(DimensionMismatch("log-scale field and sampler must have the same number of cells"))

    randn!(rng, sampler.δ)
    return θ.ϕ .+ sampler.stepsize_phi .* sampler.δ
end

"""
    propose_shape_candidates(θ, sampler; rng=default_rng())

Draw one Gaussian candidate for the common shape parameter, with mean `θ.ξ`
and standard deviation `sampler.stepsize_xi`.
"""
function propose_shape_candidates(θ::CurrentStates, sampler::MHSampler; rng::AbstractRNG=Random.default_rng())
    return θ.ξ + sampler.stepsize_xi * randn(rng)
end


## MCMC

"""
    sample_location!(θ, sampler, y, F_struct; rng=default_rng())

Perform one Metropolis sweep of the GEV location field `θ.μ`.
Update `θ.loglike` for accepted proposals and process conditionally
independent cells in parallel.
"""
function sample_location!(
    θ::CurrentStates, sampler::MHSampler,
    y::AbstractVector{<:AbstractVector{<:Real}}, F_struct::GMRF.iGMRF;
    rng::AbstractRNG=Random.default_rng(),
)
    cells = axes(F_struct.W, 1)
    for v in (y, θ.μ, θ.ϕ, θ.loglike, sampler.stepsize_mu, sampler.δ, sampler.u)
        axes(v, 1) == cells || throw(DimensionMismatch("expected one entry per grid cell"))
    end

    F = GMRF.set_precision(F_struct, θ.κ₁)
    μ̃ = propose_location_candidates(θ, sampler; rng=rng)
    rand!(rng, sampler.u)

    for subset in F.cond_ind_subset
        # Condition on the field after the preceding subsets have been updated.
        conditionals = GMRF.full_conditionals(F, θ.μ)

        Threads.@threads for j in subset
            candidate_loglike = gev_loglikelihood(y[j], μ̃[j], θ.ϕ[j], θ.ξ)
            prior = conditionals[j]
            logprior_ratio = logpdf(prior, μ̃[j]) - logpdf(prior, θ.μ[j])
            logratio = (candidate_loglike - θ.loglike[j]) + logprior_ratio

            if log(sampler.u[j]) < logratio
                θ.μ[j] = μ̃[j]
                θ.loglike[j] = candidate_loglike
            end
        end
    end
    return nothing
end

"""
    sample_logscale!(θ, sampler, y, F_struct; rng=default_rng())

Perform one Metropolis sweep of the GEV log-scale field `θ.ϕ`.
Update each accepted cell and its entry in `θ.loglike`, processing
conditionally independent cells in parallel.
"""
function sample_logscale!(
    θ::CurrentStates, sampler::MHSampler,
    y::AbstractVector{<:AbstractVector{<:Real}}, F_struct::GMRF.iGMRF;
    rng::AbstractRNG=Random.default_rng(),
)
    cells = axes(F_struct.W, 1)
    for v in (y, θ.μ, θ.ϕ, θ.loglike, sampler.stepsize_phi, sampler.δ, sampler.u)
        axes(v, 1) == cells || throw(DimensionMismatch("expected one entry per grid cell"))
    end

    F = GMRF.set_precision(F_struct, θ.κ₂)
    ϕ̃ = propose_logscale_candidates(θ, sampler; rng=rng)
    rand!(rng, sampler.u)

    for subset in F.cond_ind_subset
        # Condition on the field after the preceding subsets have been updated.
        conditionals = GMRF.full_conditionals(F, θ.ϕ)

        Threads.@threads for j in subset
            candidate_loglike = gev_loglikelihood(y[j], θ.μ[j], ϕ̃[j], θ.ξ)
            prior = conditionals[j]
            logprior_ratio = logpdf(prior, ϕ̃[j]) - logpdf(prior, θ.ϕ[j])
            logratio = (candidate_loglike - θ.loglike[j]) + logprior_ratio

            if log(sampler.u[j]) < logratio
                θ.ϕ[j] = ϕ̃[j]
                θ.loglike[j] = candidate_loglike
            end
        end
    end
    return nothing
end

"""
    sample_shape!(θ, sampler, y, F_struct; rng=default_rng())

Perform one Metropolis update of the common GEV shape under a uniform prior
on `(-0.5, 0.5)`. On acceptance, update `θ.ξ` and all entries of `θ.loglike`.
"""
function sample_shape!(
    θ::CurrentStates, sampler::MHSampler,
    y::AbstractVector{<:AbstractVector{<:Real}}, F_struct::GMRF.iGMRF;
    rng::AbstractRNG=Random.default_rng(),
)
    cells = axes(F_struct.W, 1)
    for v in (y, θ.μ, θ.ϕ, θ.loglike)
        axes(v, 1) == cells || throw(DimensionMismatch("expected one entry per grid cell"))
    end
    -0.5 < θ.ξ < 0.5 || throw(DomainError(θ.ξ, "current shape must lie in (-0.5, 0.5)"))

    ξ̃ = propose_shape_candidates(θ, sampler; rng=rng)
    -0.5 < ξ̃ < 0.5 || return nothing

    candidate_loglike = similar(θ.loglike)
    Threads.@threads for j in cells
        candidate_loglike[j] = gev_loglikelihood(y[j], θ.μ[j], θ.ϕ[j], ξ̃)
    end

    logratio = sum(candidate_loglike[j] - θ.loglike[j] for j in cells)
    if log(rand(rng)) < logratio
        θ.ξ = ξ̃
        copyto!(θ.loglike, candidate_loglike)
    end
    return nothing
end

"""
    sample_precision!(θ, F_struct; rng=default_rng())

Draw the location and log-scale precisions `θ.κ₁` and `θ.κ₂` from their full
conditionals under independent `Gamma(1, 1)` priors, using the unit-precision
structure matrix `F_struct.W`.
"""
function sample_precision!(
    θ::CurrentStates, F_struct::GMRF.iGMRF; rng::AbstractRNG=default_rng(),
)
    W = F_struct.W
    m = size(W, 1)
    length(θ.μ) == m == length(θ.ϕ) ||
        throw(DimensionMismatch("expected one location and one log-scale per grid cell"))

    order = F_struct.order
    order in (1, 2) || throw(ArgumentError("unsupported iGMRF order $order"))
    rank_deficiency = order == 1 ? 1 : 3
    rank_W = m - rank_deficiency

    shape = 1 + rank_W / 2
    location_rate = 1 + dot(θ.μ, W, θ.μ) / 2
    logscale_rate = 1 + dot(θ.ϕ, W, θ.ϕ) / 2

    κ₁ = rand(rng, Gamma(shape, 1 / location_rate))
    κ₂ = rand(rng, Gamma(shape, 1 / logscale_rate))
    θ.κ₁, θ.κ₂ = κ₁, κ₂
    return nothing
end

"""
    run_mcmc!(θ, sampler, y, F_unit; niter, warmup=0, batchsize=100, rng=default_rng())

Run `niter - 1` sweeps from a state `θ` initialized for `y`, updating `θ` and
`sampler` in place. Return the parameter chains in a named tuple, including
the initial and warmup states. Adapt proposal step sizes after each complete
batch of `batchsize` sweeps within the first `warmup` sweeps.
"""
function run_mcmc!(
    θ::CurrentStates, sampler::MHSampler,
    y::AbstractVector{<:AbstractVector{<:Real}}, F_unit::GMRF.iGMRF;
    niter::Int, warmup::Int=0, batchsize::Int=100, rng::AbstractRNG=default_rng(),
)
    niter >= 1 || throw(ArgumentError("niter must be at least 1"))
    0 <= warmup < niter || throw(ArgumentError("warmup must lie between 0 and niter - 1"))
    batchsize >= 1 || throw(ArgumentError("batchsize must be at least 1"))
    axes(y, 1) == axes(θ.μ, 1) == axes(F_unit.W, 1) ||
        throw(DimensionMismatch("y, θ and F_unit must use the same grid"))

    chain = initialize_chain(θ; niter=niter)
    @showprogress for iter in 2:niter
        sample_location!(θ, sampler, y, F_unit; rng=rng)
        sample_logscale!(θ, sampler, y, F_unit; rng=rng)
        sample_shape!(θ, sampler, y, F_unit; rng=rng)
        sample_precision!(θ, F_unit; rng=rng)

        chain.μ[:, iter] = θ.μ
        chain.ϕ[:, iter] = θ.ϕ
        chain.ξ[iter] = θ.ξ
        chain.κ₁[iter] = θ.κ₁
        chain.κ₂[iter] = θ.κ₂

        sweep = iter - 1
        if sweep <= warmup && sweep % batchsize == 0
            cols = (iter-batchsize):iter

            rate_mu = changerate(view(chain.μ, :, cols))
            rate_phi = changerate(view(chain.ϕ, :, cols))
            rate_xi = changerate(view(chain.ξ, cols))

            sampler.stepsize_mu .= update_stepsize.(sampler.stepsize_mu, rate_mu)
            sampler.stepsize_phi .= update_stepsize.(sampler.stepsize_phi, rate_phi)
            sampler.stepsize_xi = update_stepsize(sampler.stepsize_xi, rate_xi)
        end
    end
    return chain
end

## Metropolis-Hastings sampler adjustment

"""
    changerate(x::AbstractVector)
    changerate(X::AbstractMatrix)

Return the fraction of consecutive state pairs that differ according to `!=`.

For a matrix, return one rate per row, with iterations in columns.
Each series must contain at least two states.
"""
function changerate(x::AbstractVector{<:Real})
    length(x) >= 2 || throw(ArgumentError("at least two states are required"))
    indices = (firstindex(x)+1):lastindex(x)
    return count(i -> x[i] != x[i-1], indices) / (length(x) - 1)
end

changerate(X::AbstractMatrix{<:Real}) = map(changerate, eachrow(X))

"""
    update_stepsize(δ, accrate; logstep=0.01)

Adjust the standard deviation `δ` of a scalar random-walk proposal.

Increase `log(δ)` by `logstep` when `accrate` is above 0.44 and decrease it
by `logstep` when it is below. Return `δ` at the target.
"""
function update_stepsize(δ::Real, accrate::Real; logstep::Real=0.01)
    isfinite(δ) && δ > 0 || throw(DomainError(δ, "expected a finite positive step size"))
    0 <= accrate <= 1 || throw(DomainError(accrate, "expected a rate in [0, 1]"))
    isfinite(logstep) && logstep >= 0 ||
        throw(DomainError(logstep, "expected a finite nonnegative log step"))

    target = oftype(float(accrate), 44 // 100)
    return δ * exp(logstep * sign(accrate - target))
end



## Manipulating chains

"""
    chain2tuple(chain::MambaLite.Chains; run=1)

Extract `(μ, ϕ, ξ, κ₁, κ₂)` from the run at position `run` in `chain`.

Rows of `μ` and `ϕ` follow grid-cell indices `1:ncells`, and columns index iterations.
"""
function chain2tuple(chain::MambaLite.Chains; run::Integer=1)
    draws = view(chain.value, :, :, run)
    column = Dict(name => j for (j, name) in enumerate(chain.names))

    nμ = count(startswith("μ["), chain.names)
    nϕ = count(startswith("ϕ["), chain.names)
    nμ == nϕ || throw(DimensionMismatch("μ and ϕ must have the same number of cells"))

    μcols = [column["μ[$j]"] for j in 1:nμ]
    ϕcols = [column["ϕ[$j]"] for j in 1:nϕ]

    return (
        μ=permutedims(view(draws, :, μcols)),
        ϕ=permutedims(view(draws, :, ϕcols)),
        ξ=draws[:, column["ξ"]],
        κ₁=draws[:, column["κ₁"]],
        κ₂=draws[:, column["κ₂"]],
    )
end

"""
    tuple2chain(res::NamedTuple)

Convert the MCMC draws in `res` to a single `MambaLite.Chains` run.

`μ` and `ϕ` are cell-by-iteration matrices of the same size.
`ξ`, `κ₁`, and `κ₂` are vectors with one value per iteration.
"""
function tuple2chain(res::NamedTuple)
    size(res.μ) == size(res.ϕ) ||
        throw(DimensionMismatch("μ and ϕ must have the same dimensions"))

    ncells = size(res.μ, 1)
    μnames = ["μ[$j]" for j in 1:ncells]
    ϕnames = ["ϕ[$j]" for j in 1:ncells]
    names = vcat(μnames, ϕnames, ["ξ", "κ₁", "κ₂"])

    draws = hcat(transpose(res.μ), transpose(res.ϕ), res.ξ, res.κ₁, res.κ₂)
    return MambaLite.Chains(draws; names=names)
end