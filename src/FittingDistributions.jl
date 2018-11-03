module FittingDistributions
using LinearAlgebra , Statistics, Calculus
using Distributions, Random, StatsBase # mostly for testing
using CmdStan

# defines the struct of fitting problem, and some oprations on it
include("fittingStructs.jl")

# fit by KL minimization
include("klmimimization.jl")

# use probabilistic inference on stan
# (to be compared with analytic approaches)
include("fitonstan.jl")

# in order to make tests, I need to have data!
include("mock_data.jl")


end #of module Distr_fit
