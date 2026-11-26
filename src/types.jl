
"""
    CurrentStates(μ, ϕ, ξ, κ₁, κ₂, loglike)

Store the current MCMC state of the gridded GEV model.

`μ`, `ϕ`, and `loglike` hold the location, log-scale, and log-likelihood
contribution for each cell. `ξ` is the common shape parameter. `κ₁` and `κ₂`
are the precisions of the location and log-scale fields, respectively.
"""
mutable struct CurrentStates
    μ::Vector{Float64}
    ϕ::Vector{Float64}
    ξ::Float64
    κ₁::Float64
    κ₂::Float64
    loglike::Vector{Float64}

    function CurrentStates(μ, ϕ, ξ, κ₁, κ₂, loglike)
        length(μ) == length(ϕ) == length(loglike) ||
            throw(DimensionMismatch("μ, ϕ, and loglike must have the same length"))
        return new(μ, ϕ, ξ, κ₁, κ₂, loglike)
    end
end

"""
    show(io, state::CurrentStates; prefix=nothing)
    show(io, ::MIME"text/plain", state::CurrentStates; prefix="")

Write a summary of `state` to `io`. Use `text/plain` or pass `prefix`
for a detailed summary with one line per field.
"""
function Base.show(io::IO, state::CurrentStates; prefix=nothing)
    if isnothing(prefix)
        return show(IOContext(io, :compact => true), MIME"text/plain"(), state)
    end
    return show(io, MIME"text/plain"(), state; prefix=prefix)
end

function Base.show(io::IO, ::MIME"text/plain", state::CurrentStates;
                   prefix::AbstractString="")
    print(io, prefix, "CurrentStates (", length(state.μ), " grid cells)")
    get(io, :compact, false) && return nothing

    println(io)
    println(io, prefix, "  location μ = ", state.μ)
    println(io, prefix, "  log-scale ϕ = ", state.ϕ)
    println(io, prefix, "  shape ξ = ", state.ξ)
    println(io, prefix, "  location precision κ₁ = ", state.κ₁)
    println(io, prefix, "  log-scale precision κ₂ = ", state.κ₂)
    print(io, prefix, "  loglike = ", state.loglike)
end

"""
    MHSampler(stepsize_mu, stepsize_phi, stepsize_xi, δ, u)

Store proposal standard deviations and work vectors for the gridded GEV sampler.

`stepsize_mu` and `stepsize_phi` contain one step size per cell, and `stepsize_xi`
controls the proposal for the common shape parameter. All step sizes must be
finite and positive. `δ` and `u` hold standard normal and uniform draws per cell.
"""
mutable struct MHSampler
    stepsize_mu::Vector{Float64}
    stepsize_phi::Vector{Float64}
    stepsize_xi::Float64
    δ::Vector{Float64}
    u::Vector{Float64}

    function MHSampler(stepsize_mu, stepsize_phi, stepsize_xi, δ, u)
        s = new(stepsize_mu, stepsize_phi, stepsize_xi, δ, u)

        n = length(s.stepsize_mu)
        n == length(s.stepsize_phi) == length(s.δ) == length(s.u) ||
            throw(DimensionMismatch("stepsize_mu, stepsize_phi, δ, and u must have the same length"))
        n > 0 || throw(ArgumentError("at least one grid cell is required"))

        valid_step(x) = isfinite(x) && x > 0
        valid = all(valid_step, s.stepsize_mu) &&
                all(valid_step, s.stepsize_phi) &&
                valid_step(s.stepsize_xi)
        valid || throw(ArgumentError("proposal standard deviations must be finite and positive"))

        return s
    end
end

"""
    show(io, sampler::MHSampler; prefix=nothing)
    show(io, ::MIME"text/plain", sampler::MHSampler; prefix="")

Write a summary of `sampler` to `io`. Pass `prefix` to indent the detailed output.
"""
function Base.show(io::IO, sampler::MHSampler; prefix=nothing)
    if isnothing(prefix)
        return Base.show(IOContext(io, :compact => true), MIME"text/plain"(), sampler)
    end
    return Base.show(io, MIME"text/plain"(), sampler; prefix=prefix)
end

function Base.show(io::IO, ::MIME"text/plain", sampler::MHSampler;
                   prefix::AbstractString="")
    print(io, prefix, "MHSampler (", length(sampler.stepsize_mu), " grid cells)")
    get(io, :compact, false) && return nothing

    println(io)
    println(io, prefix, "  stepsize_mu = ", sampler.stepsize_mu)
    println(io, prefix, "  stepsize_phi = ", sampler.stepsize_phi)
    println(io, prefix, "  stepsize_xi = ", sampler.stepsize_xi)
    println(io, prefix, "  δ (standard normal draws) = ", sampler.δ)
    print(io, prefix, "  u (uniform draws) = ", sampler.u)
end


