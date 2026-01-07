include("../src/core.jl")

struct InventoryState
    on_hand::Int
    on_order::Int
end

inventory_position(is::InventoryState) = is.on_order + is.on_hand

struct SimpleInventoryMRPFinite
    capacity::Int
    poisson_lambda::Float64
    holding_cost::Float64
    stockout_cost::Float64
end

function get_transition_reward_map(si::SimpleInventoryMRPFinite)
    d::Dict{InventoryState, LabeledCategorical{Tuple{InventoryState, Float64}}} = Dict()
    poisson_dist = Poisson(si.poisson_lambda)
    n = si.capacity
    for alpha in 0:n
        m = si.capacity - alpha
        for beta in 0:m
            state = InventoryState(alpha, beta)
            ip = inventory_position(state)
            beta1 = si.capacity - ip
            base_reward = -si.holding_cost * state.on_hand
            k = ip - 1
            sr_probs_map::Dict{Tuple{InventoryState, Float64}, Float64} = Dict(
                (InventoryState(ip - i, beta1), base_reward) => pdf(poisson_dist, i) for i in 0:k
            )
            prob = 1 - cdf(poisson_dist, k)
            reward = base_reward - si.stockout_cost * (si.poisson_lambda - ip*(1-pdf(poisson_dist, ip)/prob))
            sr_probs_map[(InventoryState(0, beta1), reward)] = prob
            d[state] = LabeledCategorical(sr_probs_map)
        end
    end
    return d
end

si_fmrp = SimpleInventoryMRPFinite(2, 1.0, 1.0, 10.0)
fmrp = FiniteMarkovRewardProcess(get_transition_reward_map(si_fmrp))
pprintln(get_value_function_vec(fmrp, 0.9))
pprintln(fmrp.reward_function_vec)
