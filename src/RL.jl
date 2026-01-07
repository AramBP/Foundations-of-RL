module RL

export DistributionsExt, MarkovProcesses, MarkovRewardProcesses, Policies, MarkovDecisionProcesses,
    Iterate

include("DistributionsExt.jl")
include("markov_processes/MarkovProcesses.jl")
include("markov_processes/MarkovRewardProcesses.jl")
include("markov_processes/Policies.jl")
include("markov_processes/MarkovDecisionProcesses.jl")
include("dynamic_programming/Iterate.jl")

end