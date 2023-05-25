function deriv_cost_output(output_act, expected_output)

    return 2 * (output_act - expected_output)

end

function deriv_activation_function(weighted_input)

    tmp = activation_function(weighted_input)
    deriv = similar(tmp)
    for i in eachindex(tmp)
        deriv[i] = tmp[i] * (1-tmp[i])
    end

    return deriv

    # tmp = similar(weighted_input)
    # for i in eachindex(weighted_input)
    #     tmp[i] = weighted_input[i] >= 0 ? 1 : 0
    # end
    # return tmp
end



function deriv_weights_biases(neuron_states, weights, biases, expected_output)

    cost_grad_w = similar(weights)
    cost_grad_b = similar(biases)
    delta = similar(neuron_states)
    
    expected_output_vec = exp_out_vec(expected_output)
    

    delta[end] = deriv_cost_output(neuron_states[end], expected_output_vec) .* deriv_activation_function(neuron_states[end])
    #cost_grad_w[end] = delta[end]*neuron_states[end]'
    #cost_grad_b[end] = delta[end]

    for i = length(delta)-1:-1:1
        delta[i] = (delta[i+1]'*weights[i])' .* deriv_activation_function(neuron_states[i])
        cost_grad_b[i] = delta[i+1]
        cost_grad_w[i] = delta[i+1]*neuron_states[i]'
    end

    return cost_grad_w, cost_grad_b
end

function manueller_gradient(input, neuron_states, weights, biases, expected_output, h)

    m_grad_w = [zeros(size(weights[i],1), size(weights[i],2)) for i in eachindex(weights)]
    #m_grad_b = similar(biases)

    h_mats = []
    h_vecs = []
    for i in eachindex(weights)
        push!(h_mats, zeros(size(weights[i],1), size(weights[i],2)))
        #push!(h_vecs, zeros(biases[i]))
    end

    for i in eachindex(weights)
        for j = 1:size(weights[i], 1)
            for k = 1:size(weights[i], 2)
                h_mats[i][j,k] += h
                deriv = current_cost!(input, expected_output, [weights[l]+h_mats[l] for l in eachindex(weights)], biases, neuron_states)-current_cost!(input, expected_output, weights, biases, neuron_states)
                h_mats[i][j,k] -= h
                m_grad_w[i][j,k] = deriv/h
            end
        end
    end

    return m_grad_w

end

