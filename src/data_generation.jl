

"""
    generate_gridded_parameters(F; ξ, location_precision, logscale_precision,
                                rng=default_rng())

Draw independent GEV location and log-scale fields on the grid defined by `F`.

Return a dictionary with keys `:F`, `:μ`, `:ϕ`, `:ξ`, `:κ₁`, and `:κ₂`.
The precisions must be finite and positive, and `-0.5 < ξ < 0.5`.
"""
function generate_gridded_parameters(
    F::GMRF.iGMRF;
    ξ::Real,
    location_precision::Real,
    logscale_precision::Real,
    rng::AbstractRNG=Random.default_rng(),
)
    -0.5 < ξ < 0.5 ||
        throw(ArgumentError("ξ must be in (-0.5, 0.5), got $ξ"))
    isfinite(location_precision) && location_precision > 0 ||
        throw(ArgumentError("location_precision must be finite and positive"))
    isfinite(logscale_precision) && logscale_precision > 0 ||
        throw(ArgumentError("logscale_precision must be finite and positive"))

    μ = rand(rng, GMRF.set_precision(F, location_precision))
    ϕ = rand(rng, GMRF.set_precision(F, logscale_precision))

    return Dict(:F => F, :μ => μ, :ϕ => ϕ, :ξ => ξ,
                :κ₁ => location_precision, :κ₂ => logscale_precision)
end

"""
    generate_gridded_blockmaxima(parameters, n; rng=default_rng())

Draw `n` independent GEV block maxima per cell, conditional on the location
vector `:μ`, log-scale vector `:ϕ`, and common shape `:ξ` in `parameters`.

Return a vector whose `j`th entry contains the draws for cell `j`.
"""
function generate_gridded_blockmaxima(
    parameters::AbstractDict, n::Int; rng::AbstractRNG=Random.default_rng(),
)
    n >= 0 || throw(ArgumentError("n must be nonnegative, got $n"))

    μ = parameters[:μ]
    ϕ = parameters[:ϕ]
    ξ = parameters[:ξ]
    axes(μ) == axes(ϕ) ||
        throw(DimensionMismatch("μ and ϕ must have the same cell indices"))

    σ = exp.(ϕ)
    all(s -> isfinite(s) && s > 0, σ) ||
        throw(ArgumentError("exp.(ϕ) must contain only finite positive scales"))

    pd = GeneralizedExtremeValue.(μ, σ, ξ)
    return rand.(rng, pd, n)
end