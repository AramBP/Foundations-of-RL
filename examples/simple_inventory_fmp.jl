include("../src/core.jl")

struct InventoryState
    on_hand::Int
    on_order::Int
end

inventory_position(inv_state::InventoryState) = inv_state.on_hand + inv_state.on_order

struct SimpleInventoryFMP 
    capacity::Int
    poisson_lambda::Float64
    poisson_dist::Distributions.Poisson
end

function get_transition_map(si::SimpleInventoryFMP)
    d::Dict{InventoryState, LabeledCategorical{InventoryState}} = Dict()
    n = si.capacity
    for alpha in 0:n
        m = n - alpha
        for beta in 0:m
            state = InventoryState(alpha, beta)
            ip = inventory_position(state)
            k = ip
            beta1 = si.capacity - ip
            state_probs_dict::Dict{InventoryState, Float64} = Dict(
                InventoryState(ip - i, beta1) => (
                    i < ip ? pdf(si.poisson_dist, i) : 1 - cdf(si.poisson_dist, ip- 1)) for i in 0:k)
            d[InventoryState(alpha, beta)] = LabeledCategorical(state_probs_dict)
        end
    end
    return d
end

function SimpleInventoryFMP(capacity::Int, poisson_lambda::Float64)
    return SimpleInventoryFMP(capacity, poisson_lambda, Poisson(poisson_lambda))
end

si_mp = SimpleInventoryFMP(2, 1.0)

fmp = FiniteMarkovProcess(get_transition_map(si_mp))

stationairy_dist = get_stationairy_distribution(fmp).dict
