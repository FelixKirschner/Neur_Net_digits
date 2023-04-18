cd(@__DIR__)
include("NeurNet.jl")
include("IDX.jl")

input = 784
num_neurons = [input, 16, 16, 10]

weights, biases = init_weights_biases(num_neurons)

training_set, label_set, N = load_training_data()

batches = create_batches(N)

batch0 = batches[1]

batch_error(batch0)

weights, biases = train_batch!(batches[20], 1000, weights, biases, training_set, label_set)

batch_error(batches[21])

for iter = 1:100
    for i = 1:length(batches[1:20])
        summe = 0
        for j = 1:i
            summe += batch_error(batches[i], weights, biases, training_set, label_set)
        end
        #@info("Error before epoch no. $i training: $(summe/i)")
        weights, biases = train_batch!(batches[i], 1, weights, biases, training_set, label_set)
        summe2 = 0
        for j = 1:i
            summe2 += batch_error(batches[i], weights, biases, training_set, label_set)
        end
        @info("Error before $(summe/i). Error after no. $i training: $(summe2/i). Difference: $(summe/i-summe2/i)")
    end
end

weights[3]

w,b = backprop(training_set, label_set, weights, biases, 1)
maximum(w[1])
similar(w[1])
Gray.(training_set[batches[1][3]])
eval_instance(weights, training_set[batches[1][3]], biases)[end]

function write_network(weights, biases)
    open("network_v1", "w") do io
        write(io, num_neurons)
        write(io, "\n")
        for i = 1:length(weights)
            for j = 1:size(weights, 1)
                for k = 1:size(weights, 2)
                    write(io, weights[i][j,k])
                    write(io, "\n")
                end
            end
        end
        for i = 1:length(biases)
            for j = 1:length(biases[i])
                write(io, biases[i][j])
                write(io, "\n")
            end
        end
    end
end

