#this function outputs N data points with corresponding labels 
function generate_data_to_learn(N)
    training_set = []
    label_set = []
    for i = 1:N
        x1 = rand(Uniform(-1, 1))
        x2 = rand(Uniform(-1, 1))
        x3 = rand(Uniform(-1, 1))

        push!(training_set, [x1,x2,x3])

        label = sqrt(x1^2+x2^2+x3^2) < 1 ? 1 : 2

        push!(label_set, label)
    end
    return training_set, label_set
end

