include("../src/core.jl")

struct InventoryState
    on_hand::Int
    on_order::Int
end

inventory_position(is::InventoryState) = is.on_hand + is.on_order

struct SimpleInventoryMDPCap
    capacity::Int
    poisson_lambda::Float64
    holding_cost::Float64
    stockout_cost::Float64
end

function get_action_transition_map(simdp_cap::SimpleInventoryMDPCap)
    d = Dict{InventoryState, Dict{Int, LabeledCategorical}}()
    n = simdp_cap.capacity
    poisson_distr = Poisson(simdp_cap.poisson_lambda)
    for alpha in 0:n
        m = n - alpha
        for beta in 0:m
            state::InventoryState = InventoryState(alpha, beta)
            base_reward = - simdp_cap.holding_cost * alpha
            ip = inventory_position(state)
            d1 = Dict{Int, LabeledCategorical}()
            k = simdp_cap.capacity - ip
            for order in 0:k
                sr_probs_dict = Dict((InventoryState(ip - i, order), base_reward) => pdf(poisson_distr, i) for i in 0:(ip-1))
                probability = 1 - cdf(poisson_distr, ip - 1)
                reward = base_reward - simdp_cap.stockout_cost * (simdp_cap.poisson_lambda - ip * (1 - pdf(poisson_distr, ip)/probability))
                sr_probs_dict[(InventoryState(0, order), reward)] = probability
                d1[order] = LabeledCategorical(sr_probs_dict)
            end
            d[state] = d1
        end
    end
    return d
end

function SimpleInventoryMDPCap_FMDP(simdp_cap::SimpleInventoryMDPCap)::FiniteMarkovDecisionProcess
    return FiniteMarkovDecisionProcess(get_action_transition_map(simdp_cap))
end

user_capacity = 2
user_poisson_lambda = 1.0
user_holding_cost = 1.0
user_stockout_cost = 10.0

si_mdp = SimpleInventoryMDPCap_FMDP(
    SimpleInventoryMDPCap(
        user_capacity, user_poisson_lambda, user_holding_cost, user_stockout_cost
    )
)

fdp = FiniteDeterministicPolicy(Dict(InventoryState(alpha, beta) => user_capacity - (alpha + beta) for alpha in 0:user_capacity for beta in 0:(user_capacity - alpha)))

implied_mrp = apply_finite_policy(si_mdp, fdp.fp)