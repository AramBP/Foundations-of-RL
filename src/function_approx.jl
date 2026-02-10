abstract type FunctionApprox end

function update(f::FunctionApprox, xy_vals_seq)
    x_seq = first.(xy_vals_seq)
    y_seq = last.(xy_vals_seq) 
    deriv_func(x_seq, y_seq) = evaluate(f, x_seq) - y_seq
    return update_with_gradient(
        f, objective_gradient(f, xy_vals_seq, deriv_func)
    )
end

function iterate_updates(f::FunctionApprox, xy_seq)
    accumulate((fa, xy) -> update(fa, xy), xy_seq, init = f)
end

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

function Base.:+(grad::Gradient, x)
    if isa(x, Gradient)
        return Gradient(grad.function_approx + x.function_approx)
    end
    return grad.function_approx + x
end

Base.:*(grad::Gradient, x::Float64) = Gradient(grad.function_approx * x)
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
    weights::AbstractArray
    adam_cache1::AbstractArray
    adam_cache2::AbstractArray
end

function create(
    weights::AbstractArray,
    adam_gradient::AdamGradient = default_settings(),
    adam_cache1::Union{AbstractArray, Nothing} = nothing, 
    adam_cache2::Union{AbstractArray, Nothing} = nothing)
    
    if adam_cache1 === nothing
        adam_cache1 = zero(weights)
    end
    if adam_cache2 === nothing
        adam_cache2 = zero(weights)
    end

    return Weights(adam_gradient, 0, weights, adam_cache1, adam_cache2)
end

function update(w::Weights, gradient::AbstractArray)
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

function create(
    feature_functions, 
    adam_gradient::AdamGradient, 
    regularization_coeff::Float64, 
    weights::Union{Weights, Nothing} = nothing, 
    direct_solve = true)
    if weights === nothing
        weights = create(zeros(length(feature_functions)), adam_gradient)
    end
    return LinearFunctionApprox(feature_functions, regularization_coeff, weights, direct_solve)
end

function get_feature_values(lfa::LinearFunctionApprox, x_values_seq)
    n = length(lfa.feature_functions)
    m = length(x_values_seq)
    mat = zeros(m, n)
    for i in 1:m
        x = x_values_seq[i]
        for j in 1:n
            f = lfa.feature_functions[j]
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

function solve(lfa::LinearFunctionApprox, xy_vals_seq, error_tolerance::Union{Float64, Nothing} = nothing)
    if lfa.direct_solve
        x_seq = first.(xy_vals_seq)
        y_seq = last.(xy_vals_seq)
        feature_vals = get_feature_values(lfa, x_seq)
        feature_vals_T = transpose(feature_vals)
        left = (feature_vals_T * feature_vals) .+ (size(feature_vals)[1] * lfa.regularization_coeff * I(length(lfa.weights.weights)))
        right = feature_vals_T * y_seq
        weights = create(left / right, lfa.weights.adam_gradient)
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

struct DNNSpec
    neurons
    bias::Bool
    hidden_activation::Function
    hidden_activation_deriv::Function
    output_activation::Function
    output_activation_deriv::Function
end


struct DNNApprox <: FunctionApprox
    feature_functions
    dnn_spec::DNNSpec
    regularization_coeff::Float64
    weights
end

function create(
    feature_functions, 
    dnn_spec::DNNSpec, 
    adam_gradient::AdamGradient = default_settings(),
    regularization_coeff = 0.0,
    weights = nothing)

    if weights === nothing
        inputs = append!([length(feature_functions)], [n + (dnn_spec.bias ? 1 : 0) for n in dnn_spec.neurons])
        outputs = push!(collect(dnn_spec.neurons), 1)
        w = [create(randn(output, input) / sqrt(input), adam_gradient) for (input, output) in zip(inputs, outputs)]
    else
        w = weights
    end

    return DNNApprox(feature_functions, dnn_spec, regularization_coeff, w)
end 

function get_feature_values(dnn::DNNApprox, x_values_seq)
    n = length(dnn.feature_functions)
    m = length(x_values_seq)
    mat = zeros(m, n)
    for i in 1:m
        x = x_values_seq[i]
        for j in 1:n
            f = dnn.feature_functions[j]
            mat[i, j] = f(x)
        end
    end
    return mat
end

function forward_propagation(dnn::DNNApprox, x_values_seq)
    input = get_feature_values(dnn, x_values_seq)
    ret = [input]
    for w in dnn.weights[1:end-1]
        output = dnn.dnn_spec.hidden_activation(input*transpose(w.weights))
        if dnn.dnn_spec.bias
            input = hcat(ones(length(output)), output)
        else
            input = output
        end
        push!(ret, dnn.dnn_spec.output_activation(input*transpose(dnn.weights[end].weights))[:, 1])
    end
    return ret
end

function evaluate(dnn::DNNApprox, x_values_seq)
    return forward_propagation(dnn, x_values_seq)[end]
end

function backward_propagation(dnn::DNNApprox, fwd_prop, obj_deriv_out)
    deriv = reshape(permutedims(obj_deriv_out), 1, :)
    back_prop = [(deriv * fwd_prop[end]) / size(deriv, 2)]
    for i in (length(dnn.weights) - 1):1
        deriv = transpose(dnn.weights[i + 1].weights)*deriv .* dnn.dnn_spec.hidden_activation_deriv(transpose(fwd_prop[i+1])) 
        if dnn.dnn_spec.bias
            deriv = deriv[2:end]
        end
        push!(back_prop, (deriv * fwd_prop[i]) / size(deriv, 2))
    end
    return reverse(back_prop)
end

function objective_gradient(dnn::DNNApprox, xy_vals_seq, obj_deriv_out_fun)
    x_vals = first.(xy_vals_seq)
    y_vals = last.(xy_vals_seq)
    obj_deriv_out = obj_deriv_out_fun(x_vals, y_vals)
    fwd_prop = forward_propagation(dnn, x_vals)[end]
    gradient = [x + dnn.regularization_coeff * dnn.weights[i].weights 
        for (i, x) in enumerate(backward_propagation(
            dnn, fwd_prop, obj_deriv_out
        ))] 
    weights = [Weights(w.adam_gradient, w.time, g, w.adam_cache1, w.adam_cache2) for (w, g) in zip(dnn.weights, gradient)]
    return Gradient(DNNApprox(dnn.feature_functions, dnn.dnn_spec, dnn.regularization_coeff, weights))
end

function solve(dnn::DNNApprox, xy_vals_seq, error_tolerance::Union{Float64, Nothing} = nothing)
    tol = error_tolerance === nothing ? 1e-6 : error_tolerance
    done(a::DNNApprox, b::DNNApprox, tolerance = tol) = within(a, b, tolerance)
    update(x) = iterate_updates(dnn, x)
    return converged(Iter(update, done, Iterators.repeated(collect(xy_vals_seq))))
end

function within(dnn::DNNApprox, other::FunctionApprox, tolerance::Float64)
    if isa(other, DNNApprox)
        return all(within(w1, w2, tolerance) for (w1, w2) in zip(dnn.weights, other.weights))
    else
        return false
    end
end