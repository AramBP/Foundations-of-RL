using RL
using RL.DistributionsExt: Constant, SampledDistribution
using RL.Policies: DeterministicPolicy, action_for, act, Policy
using RL.MarkovProcesses: NonTerminal
using Distributions

struct InventoryState
    on_hand::Int
    on_order::Int
end

inventory_position(is::InventoryState) = is.on_hand + is.on_order

struct SimpleInventoryDeterministicPolicy <: DeterministicPolicy{InventoryState, Int}
    reorder_point::Int
end

function RL.Policies.action_for(si_dp::SimpleInventoryDeterministicPolicy, s::InventoryState)
    return max(si_dp.reorder_point - inventory_position(s), 0)
end

struct SimpleInventoryStochasticPolicy <: Policy{InventoryState, Int}
    reorder_point_poisson_mean::Float64
end

function RL.Policies.act(si_sp::SimpleInventoryStochasticPolicy, state::NonTerminal{InventoryState})
    function action_func(s=state)  
        reorder_sample = rand(Poisson(si_sp.reorder_point_poisson_mean))
        return max(reorder_sample - inventory_position(s.state), 0)
    end

    return SampledDistribution(; sampler = action_func)
end

si_sp = SimpleInventoryStochasticPolicy(8.0)