# this function takes as input the activation value of one layer of neurons and returns the 
# activation of the following layer in terms of the current weights and biases
function calc_single_output!(input_vec, weights, biases, ind, neuron_states)
    neuron_states[ind+1] = activation_function(weights[ind] * input_vec + biases[ind])
end


#this function take a training sample and calculates the resulting neuron states for the current weights and biases
function calc_output!(input_vec, weights, biases, neuron_states)
    neuron_states[1] = input_vec
    for i = 1:length(weights)
        calc_single_output!(neuron_states[i], weights, biases, i, neuron_states)
    end
end

#this function return the index of the neuron with the strongest activation in the output layer
function classify_output(neuron_states)
    return argmax(neuron_states[end])
end

#This function with rescale the output of a neuron to be a value in [0,1]
# chosen atm sigmoid, other choice possible -> experiment
function activation_function(weightedInput)
    tmp = similar(weightedInput)
    for i in eachindex(weightedInput)
        tmp[i] = 1 / (1 + exp(-weightedInput[i]))
    end
    return tmp
    # tmp = similar(weightedInput)
    # for i in eachindex(weightedInput)
    #     tmp[i] = max(0, weightedInput[i])
    # end
    # return tmp

end


function current_cost!(input, expected_output, weights, biases, neuron_states)

    calc_output!(vec(input), weights, biases, neuron_states)

    
    expected_output_vec = exp_out_vec(expected_output)

    cost = 0

    for i in eachindex(expected_output_vec)
        err = neuron_states[end][i]-expected_output_vec[i]
        cost += err*err
    end

    return cost

end

function avg_current_cost!(input_arr, dig_arr, weights, biases, neuron_states)
    avg_cost = 0

    for i in eachindex(input_arr)
        avg_cost += current_cost!(input_arr[i], dig_arr[i], weights, biases, neuron_states)
    end

    return avg_cost/length(input_arr)
end


function apply_gradient_desc!(learn_rate, cost_grad_w, cost_grad_b, weights, biases)

    for i in eachindex(weights)
        weights[i] -= cost_grad_w[i] * learn_rate
        biases[i] -= cost_grad_b[i] * learn_rate
    end

end


function create_batches(N)

    arr = [i for i = 1:N]
    shuffle!(arr)
    batches = []

    for i = 1:Int(N / 100)
        push!(batches, arr[100*(i-1)+1:100*i])
    end

    return batches

end

function write_network(weights, biases, num_neurons)
    open("network_v1.txt", "w") do io
        for el in num_neurons
            write(io, string(el) * ",")
        end
        write(io, "\n")
        for i in eachindex(weights)
            for j = 1:size(weights, 1)
                for k = 1:size(weights, 2)
                    write(io, string(weights[i][j, k]))
                    write(io, "\n")
                end
            end
        end
        for i in eachindex(biases)
            for j = 1:length(biases[i])
                write(io, string(biases[i][j]))
                write(io, "\n")
            end
        end
    end
end



function accuracy(training_set, label_set, weights, biases, neuron_states)
    acc = 0
    n = length(training_set)
    for i = 1:n
        current_cost!(training_set[i], label_set[i], weights, biases, neuron_states)
        if argmax(neuron_states[end]) == argmax(exp_out_vec(Int(label_set[i])))#+1
            acc += 1
        end
    end

    return acc / n

end

function performance_test(training_set, label_set, weights, biases, neuron_states)

    @info("Network performance:")
    println("    Average cost: $(avg_current_cost!(training_set, label_set, weights, biases, neuron_states))")
    println("    Accuracy: $(accuracy(training_set, label_set, weights, biases, neuron_states))")

end

function exp_out_vec(expected_output)
    expected_output_vec = [0 for i = 1:10]
    expected_output_vec[Int(expected_output)+1] = 1
    #expected_output_vec = [0, 0]
    #expected_output_vec[expected_output] = 1
    return expected_output_vec
end
