cd(@__DIR__)
include("libs.jl")


training_set, label_set, N, num_neurons, weights, biases, neuron_states = setup_all_inc_data([10, 10, 10]);
batches = create_batches(N)
output_size = 10

for epoch = 1:10
    for batch in batches
        train_batch!(batch, weights, biases, neuron_states, training_set, label_set, 0.05)
    end
end

accuracy(training_set, label_set, weights, biases, neuron_states)