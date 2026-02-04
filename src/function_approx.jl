abstract type FunctionApprox end

function update(f::T, xy_vals_seq) where {T <: FunctionApprox}
    deriv_func(x_seq, y_seq) = evaluate(f, x_seq) - y_seq
    return update_with_gradient(
        f, objective_gradient(f, xy_vals_seq, deriv_func)
    )
end

iterate_updates(f::FunctionApprox, xy_seq) = accumulate((fa, xy) -> update(fa, xy), xy_seq, init = f)

function rmse(f::FunctionApprox, xy_vals_seq)
    x_seq = first.(xy_vals_seq)
    y_seq = last.(xy_vals_seq)
    errors = evaluate(f, x_seq) .- y_seq
    return sqrt(mean(errors .* errors))
end

(f::FunctionApprox)(x) = evaluate(f, [x])
argmax(f::FunctionApprox, xs) = (args = collect(xs); args[Base.argmax(evaluate(f, args))])

struct Gradient
    function_approx::FunctionApprox
end

function +(grad::Gradient, x)
    if isa(x, Gradient)
        return Gradient(grad.function_approx + x.function_approx)
    end
    return grad.function_approx + x
end

*(grad::Gradient, x::Float64) = Gradient(grad.function_approx * x)
zero(grad::Gradient) = Gradient(grad.function_approx * 0.0)

const SMALL_NUM = 1e-6

struct AdamGradient
    learning_rate::Float64
    decay1::Float64
    decay2::Float64
end

function default_settings()
    return AdamGradient(0.001, 0.9, 0.99)
end

struct Weights
    adam_gradient::AdamGradient
    time::Int
    weights::Vector
    adam_cache1::Vector
    adam_cache2::Vector
end

function create(adam_gradient = default_settings(), weights::Vector, adam_cache1 = nothing, adam_cache2 = nothing)
    if adam_cache1 === nothing
        adam_cache1 = zeroslike(weights)
    end
    if adam_cache2 === nothing
        adam_cache2 = zeroslike(weights)
    end

    return Weights(adam_gradient, 0, weights, adam_cache1, adam_cache2)
end

function update(w::Weights, gradient::Vector)
    time = w.time + 1
    new_adam_cache1 = w.adam_gradient.decay1 * w.adam_cache1 .+ ((1 - w.adam_gradient.decay1) * gradient)
    new_adam_cache2 = w.adam_gradient.decay2 * w.adam_cache2 .+ ((1 - w.adam_gradient.decay2) * gradient^2)
    corrected_m = new_adam_cache1 / (1 - w.adam_gradient.decay1^time)
    corrected_v = new_adam_cache2 / (1 - w.adam_gradient.decay2^time)
    new_weights = w.weights .- w.adam_gradient.learning_rate * (corrected_m ./ (sqrt.(corrected_v) + SMALL_NUM))
    return Weights(w.adam_gradient, time, new_weights, new_adam_cache1, new_adam_cache2)
end

function within(w::Weights, other::Weights, tolerance::Float64)
    return all(abs.(w.weights .- other.weights) .<= tolerance)
end

struct LinearFunctionApprox <: FunctionApprox
    feature_functions
    regularization_coeff::Float64
    weights::Weights
    direct_solve::Bool
end

function create(feature_functions, adam_gradient, regularization_coeff, weights = nothing, direct_solve = true)
    if weights === nothing
        weights = create(adam_gradient, zeros(length(feature_functions)))
    end
    return LinearFunctionApprox(feature_functions, regularization_coeff, weights, direct_solve)
end

function get_feature_values(lfa::LinearFunctionApprox, x_values_seq)
    n = length(lfa.feature_functions)
    m = length(x_values_seq)
    mat = zeros(0.0, m, n)
    for i in 1:m
        for j in 1:n
            f = lfa.feature_functions[j]
            x = x_values_seq[i]
            mat[i, j] = f(x)
        end
    end
    return mat
end

function objective_gradient(lfa::LinearFunctionApprox, xy_vals_seq, obj_deriv_out_fun::Function)
    x_seq = first.(xy_vals_seq)
    y_seq = last.(xy_vals_seq)
    obj_deriv_out = obj_deriv_out_fun(x_seq, y_seq)
    features = get_feature_values(lfa, x_seq)
    gradient = transpose(features)*obj_deriv_out / length(obj_deriv_out) .+ lfa.regularization_coeff * lfa.weights.weights
    w = Weights(lfa.weights.adam_gradient, lfa.weights.time, gradient, lfa.weights.adam_cache1, lfa.weights.adam_cache2)
    new_lfa = LinearFunctionApprox(lfa.feature_functions, lfa.regularization_coeff, w, lfa.direct_solve)
    return Gradient(new_lfa)
end

function evaluate(lfa::LinearFunctionApprox, x_values_seq)
    return get_feature_values(lfa, x_values_seq) * lfa.weights.weights
end

function update_with_gradient(lfa::LinearFunctionApprox, gradient::Gradient)
    weights = update(lfa.weights, gradient.function_approx.weights.weights)
    return LinearFunctionApprox(lfa.feature_functions, lfa.regularization_coeff, weights, lfa.direct_solve)
end

function within(lfa::LinearFunctionApprox, other::FunctionApprox, tolerance::Float64)
    if isa(other, LinearFunctionApprox)
        return within(lfa.weights, other.weights, tolerance)
    else
        return false
    end
end

function solve(lfa::LinearFunctionApprox, xy_vals_seq, error_tolerance = nothing)
    if lfa.direct_solve
        x_seq = first.(xy_vals_seq)
        y_seq = last.(xy_vals_seq)
        feature_vals = get_feature_values(lfa, x_seq)
        feature_vals_T = transpose(feature_vals)
        left = (feature_vals_T * feature_vals) .+ (size(feature_vals)[1] * lfa.regularization_coeff * I(length(lfa.weights.weights)))
        right = feature_vals_T * y_seq
        weights = create(lfa.weights.adam_gradient, left / right)
        ret = LinearFunctionApprox(lfa.feature_functions, lfa.regularization_coeff, weights, lfa.direct_solve)
    else
        tol = error_tolerance === nothing ? 1e-6 : error_tolerance
        function done(a::LinearFunctionApprox, b::LinearFunctionApprox, tolerance = tol)
            return within(a, b, tolerance)
        end
        update(x) = iterate_updates(lfa, x)
        ret = converged(Iter(update, done, Iterators.repeated(collect(xy_vals_seq))))
    end
    return ret
end