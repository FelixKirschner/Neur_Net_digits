cd(@__DIR__)
# include("IDX.jl")
using Plots
using Images
using Random
using FileIO
using Distributions
Random.seed!(1)



function load_training_data()
    training_set_raw = IDX.load("data/train-images-idx3-ubyte")
    label_set_raw = IDX.load("data/train-labels-idx1-ubyte")

    label_set = label_set_raw[3]
    training_set = Matrix{Float16}[]


    N = size(training_set_raw[3],3)

    for i = 1:N
        push!(training_set, training_set_raw[3][:,:,i]'./265)
    end

    return training_set, label_set, N
end

#Gray.(training_set[1])
#num_neurons = [784, 16, 16, 10]

# weights = Matrix{Float64}[]
# push!(weights, rand(-1:1,16,784))
# push!(weights, rand(-1:1,16,16))
# push!(weights, rand(-1:1,10,16))

# push!(biases, [0.0 for i = 1:16])
# push!(biases, [0.0 for i = 1:16])
# push!(biases, [0.0 for i = 1:10])

function init_weights_biases(num_neurs)
    weights = Matrix{Float64}[]
    biases = Vector{Float64}[]

    for i = 2:length(num_neurons)
        #if i != 2
            push!(weights, rand(Uniform(0.4,0.6),num_neurs[i],num_neurs[i-1]))
            push!(biases, [rand(Uniform(0.4,0.6)) for i = 1:num_neurs[i]])
        #end
    end

    return weights, biases
end

#weights, biases = init_weights_biases(num_neurons)

function sigmoid(vec)
    tmp = []
    for i in eachindex(vec)
        push!(tmp, 1/(1+exp(-vec[i])))
    end
    return tmp
end

function eval_instance(weights, img, biases)
    tmp = []
    push!(tmp, vec(img))
    for i in eachindex(weights)
        push!(tmp, sigmoid(weights[i]*tmp[i]-biases[i]))
    end
    return tmp
end
function eval_quality(res, should_be)
    return sum((res[i]-should_be[i])^2 for i in eachindex(res))
end

function should_be_vec(label_set, ind)
    tmp = [0.0 for i = 1:10]
    tmp[Int(label_set[ind])+1] = 1.0
    return tmp
end

function eval_network(weights, biases, training_set, label_set)
    sum_err = 0.0
    for i in eachindex(training_set)
        sum_err += eval_quality(eval_instance(weights, training_set[i], biases)[end],should_be_vec(label_set,i))
    end
    return sum_err/length(training_set)
end


function backprop(training_set, label_set, weights, biases, ind)
    
    neurons = eval_instance(weights, training_set[ind], biases)
    partial_weights = [similar(weights[i]) for i in eachindex(weights)]
    partial_biases = [similar(biases[i]) for i in eachindex(biases)]
    partial_neurons = [similar(neurons[i]) for i in eachindex(neurons)]

    should_be = should_be_vec(label_set, ind)

    for i = 1:length(neurons)-1# for each layer
        for j = 1:length(neurons[end-i+1]) #for each neuron in that layer
            if i == 1
                partial_neurons[end-i+1][j] = 2*(neurons[end-i+1][j]-should_be[j])
                partial_biases[end-i+1][j] = (neurons[end-i+1][j]-1)*(neurons[end-i+1][j])*partial_neurons[end-i+1][j]
                for k = 1:length(neurons[end-i])
                    partial_weights[end-i+1][j,k] = neurons[end-i][k]*(1/neurons[end-i+1][j]-1)*(1/neurons[end-i+1][j])^(-2)*partial_neurons[end-i+1][j]
                end
            else
                partial_neurons[end-i+1][j] = sum(weights[end-i+2][l,j]*(1/neurons[end-i+2][l]-1)*(1/neurons[end-i+2][l])^(-2)*partial_neurons[end-i+2][l] for l = 1:length(neurons[end-i+2]))
                partial_biases[end-i+1][j] = (1/neurons[end-i+1][j]-1)*(1/neurons[end-i+1][j])^(-2)*partial_neurons[end-i+1][j]
                for k = 1:length(neurons[end-i])
                    partial_weights[end-i+1][j,k] = neurons[end-i][k]*(1/neurons[end-i+1][j]-1)*(1/neurons[end-i+1][j])^(-2)*partial_neurons[end-i+1][j]
                end
            end
        end
    end

    return partial_weights, partial_biases
end

function train_batch!(ind_batch, num_it, weights, biases, training_set, label_set)
    for j = 1:num_it
        for i in eachindex(ind_batch)
        res = backprop(training_set, label_set, weights, biases, ind_batch[i])
            for k in eachindex(weights)
                weights[k] .-= res[1][k]
                biases[k] .-= res[2][k]
            end
        end
    end
    return weights, biases
end

function create_batches(N)

    arr = [i for i = 1:N]
    shuffle!(arr)
    batches = []

    for i = 1:Int(N/100)
        push!(batches, arr[100*(i-1)+1:100*i])
    end

    return batches

end

function batch_error(batch, weights, biases, training_set, label_set)
    summe = 0
    for el in batch
        tmp = eval_instance(weights, training_set[el], biases)[end];
        summe += eval_quality(tmp, should_be_vec(label_set, el))
    end
    return summe/length(batch)
end


function write_network(weights, biases, num_neurons)
    open("network_v1.txt", "w") do io
        for el in num_neurons
            write(io, string(el)*",")
        end
        write(io, "\n")
        for i in eachindex(weights)
            for j = 1:size(weights, 1)
                for k = 1:size(weights, 2)
                    write(io, string(weights[i][j,k]))
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
