function load_training_data()
    training_set_raw = IDX.load("data/train-images-idx3-ubyte")
    label_set_raw = IDX.load("data/train-labels-idx1-ubyte")

    label_set = label_set_raw[3]
    training_set = Matrix{Float16}[]


    N = size(training_set_raw[3], 3)

    for i = 1:N
        push!(training_set, training_set_raw[3][:, :, i]' ./ 265)
    end

    return training_set, label_set, N
end

# biases exists for any neural except the ones in the input layer. Created here randomly in vectors for each layer
# weights are given as matrices of size num_neurons[i] x num_neurons[i-1] for 0<i<length(num_neurons)
function init_weights_biases(num_neurs)
    weights = Matrix{Float64}[]
    biases = Vector{Float64}[]

    for i = 2:length(num_neurs)
        #if i != 2
        push!(weights, rand(Uniform(-1, 1), num_neurs[i], num_neurs[i-1]))
        push!(biases, [rand(Uniform(-1, 1)) for i = 1:num_neurs[i]])
        #end
    end

    return weights, biases
end

#Gray.(training_set[2])


function setup_all_inc_data(num_hidden_neurons)
    training_set, label_set, N = load_training_data()
    input_size = length(vec(training_set[1]))

    num_neurons = [input_size]
    for el in num_hidden_neurons
        push!(num_neurons, el)
    end
    push!(num_neurons, 10) #10 is number of outputs
    weights, biases = init_weights_biases(num_neurons)

    neuron_states = []
    for el in num_neurons[1:end]
        tmp = [0.0 for i = 1:el]
        push!(neuron_states, tmp)
    end

    return training_set, label_set, N, num_neurons, weights, biases, neuron_states

end

function setup_all_excl_data(training_set, label_set, num_hidden_neurons)
    N = length(training_set)
    input_size = length(training_set[1])

    num_neurons = [input_size]
    for el in num_hidden_neurons
        push!(num_neurons, el)
    end
    push!(num_neurons, 2) #2 is number of outputs in current example
    weights, biases = init_weights_biases(num_neurons)
    display(num_neurons)
    neuron_states = []
    for el in num_neurons[1:end]
        tmp = [0.0 for i = 1:el]
        push!(neuron_states, tmp)
    end

    return N, num_neurons, weights, biases, neuron_states

end
