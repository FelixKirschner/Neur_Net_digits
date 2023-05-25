include("IDX.jl")
include("auxfunctions.jl")
include("setup.jl")
include("partial_derivatives.jl")
include("training.jl")
include("gen_data.jl")
using Plots
using Images
using Random
using FileIO
using Distributions
Random.seed!(1)