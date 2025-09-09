module DistributionsExt

using Distributions: Distribution, Univariate, Discrete, Categorical
using Random, Statistics

export FiniteDistribution, LabeledCategorical, Constant, SampledDistribution, probability

abstract type FiniteDistribution{T} <: Distribution{Univariate, Discrete} end

struct LabeledCategorical{T} <: FiniteDistribution{T}
    labels::Vector{T}
    dist::Categorical
    dict::Dict{T, Float64}
end

function LabeledCategorical(probs::Dict{T, Float64}) where {T}
    labels = collect(keys(probs))
    weights = collect(values(probs))
    d = Categorical(weights)
    return LabeledCategorical{T}(labels, d, probs)
end

function probability(lc::LabeledCategorical{T}, key::T) where {T}
    if key in lc.labels
        return lc.dict[key]
    else
        return 0.0
    end
end

Base.rand(rng::AbstractRNG, d::LabeledCategorical) = d.labels[rand(rng, d.dist)]
Base.rand(d::LabeledCategorical) = d.labels[rand(d.dist)]

struct Constant{T} <: Distribution{Univariate, Discrete}
    value::T
end

Base.rand(rng::AbstractRNG, d::Constant) = d.value
Base.rand(d::Constant) = d.value

# A distribution defined by a function to sample it
Base.@kwdef struct SampledDistribution{F}
    sampler::F
    expectation_samples::Int = 10000
end

Base.rand(d::SampledDistribution) = d.sampler()

expectation(d::SampledDistribution, f::Function) = sum([f(rand(d)) for _ in 0:d.expectation_samples])/d.expectation_samples

struct Choose{T} <: FiniteDistribution{T}
    options::Array{T}
end

function Base.rand(d::Choose) 
    n = length(d.optionsoptions)
    idx = rand(1:n)
    return d.options[idx]
end

end