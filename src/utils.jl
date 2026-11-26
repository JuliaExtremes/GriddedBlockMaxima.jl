
"""
    cvmcriterion(pd::UnivariateDistribution, sample::Vector{<:Real})

Compute the Cramér-von Mises distance between the distribution `pd` and the sample `sample`.
"""
function cvmcriterion(pd::UnivariateDistribution, sample::Vector{<:Real})

    x = sort(sample)
    n = length(x)

    T = 1/12/n + sum( ((2*i-1)/2/n - cdf(pd,x[i]) )^2 for i=1:n)

    ω² = T/n

    return ω²

end