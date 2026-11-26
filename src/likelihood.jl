
"""
    gev_loglikelihood(y, μ, ϕ, ξ)

Compute the GEV log-likelihood for block maxima `y` at one grid cell,
given location `μ`, log-scale `ϕ`, and shape `ξ`.
"""
function gev_loglikelihood(y::AbstractVector{<:Real}, μ::Real, ϕ::Real, ξ::Real)
    return loglikelihood(GeneralizedExtremeValue(μ, exp(ϕ), ξ), y)
end