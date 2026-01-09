"""
This file extends the Distribution package by adding some distribution specific for our problems and code.
"""
abstract type FiniteDistribution{T} <: Distribution{Univariate, Discrete} end

map(fd::FiniteDistribution, f::Function) = error("The map function is not defined for $(typeof(fd))")

struct LabeledCategorical{T} <: FiniteDistribution{T}
    labels::Vector{T}
    dist::Categorical
    dict::Dict{T, Float64}
end

function LabeledCategorical(probs::Dict{T, Float64}) where {T}
    labels = collect(Base.keys(probs))
    weights = collect(Base.values(probs))
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

function map(fd::LabeledCategorical, f::Function)
    result = DefaultDict(0.0)
    for (x,p) in fd.dict
        result[f(x)] += p
    end

    return LabeledCategorical(result)
end

expectation(lc::LabeledCategorical, f::Function) = sum(f(x)*p for (x, p) in lc.dict)

struct Constant{T} <: FiniteDistribution{T}
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

# Apply a function to the outcomes of a distribution
# f: A -> Distribution{B}
function apply(d::Distribution, f::Function)
    function sample() 
        a = rand(d)
        b_dist = f(a)
        return rand(b_dist)
    end
    return SampledDistribution(; sampler = sample)
end

struct Choose{T} <: FiniteDistribution{T}
    options::Array{T}
end

function Base.rand(d::Choose) 
    n = length(d.options)
    idx = rand(1:n)
    return d.options[idx]
end
