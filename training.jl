function train_batch!(batch, weights, biases, neuron_states, training_set, label_set, learn_rate)
    cost_grad_w = similar(weights)
    cost_grad_b = similar(biases)

    len = length(batch)

    for i in eachindex(batch)
        input_vec = vec(training_set[batch[i]])
        expected_output = label_set[batch[i]]

        calc_output!(input_vec, weights, biases, neuron_states)

        tmp_w, tmp_b = deriv_weights_biases(neuron_states, weights, biases, expected_output)
        if i == 1 
            for i in eachindex(cost_grad_w)
                cost_grad_w[i] = tmp_w[i]
                cost_grad_b[i] = tmp_b[i]
            end 
        else
            for i in eachindex(cost_grad_w)
                cost_grad_w[i] += tmp_w[i]
                cost_grad_b[i] += tmp_b[i]
            end 
        end
    end

    apply_gradient_desc!(learn_rate / len, cost_grad_w, cost_grad_b, weights, biases)

end
