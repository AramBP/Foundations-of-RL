include("../src/core.jl")

# generate data seqeuence [(x_i, y_i) | 1 <= i <= n] 
# for the model: y = 2 + 10*x_1 + 4 * x_2 - 6 * x_3 + N(0, 0.3) 
function data_seq(n)
    coeffs = (2., 10., 4., -6.)
    d = Normal(0, 0.3)
    f(x::Tuple) = (x, coeffs[1] + dot(coeffs[2:end], x) + rand(d))
    return [f(Tuple(randn(3))) for _ in 1:n]
end

function feature_functions()
    return [_ -> 1., x -> x[1], x -> x[2], x -> x[3]]
end

function adam_gradient()
    return AdamGradient(0.1, 0.9, 0.999)
end

function get_linear_model()
    ffs = feature_functions()
    ag = adam_gradient()
    return create(ffs, ag, 0.)
end

get_linear_model()

