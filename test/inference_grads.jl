#=
Testing routines for inference gradients
=#

using Revise
using FittingDistributions
using Random, Distributions
using StatsBase , LinearAlgebra, Statistics

const D = FittingDistributions

##

n = 8
x_test=randn(n)
q1_test = let q = rand(n)
    q./=sum(q)
end
q2_test = let q = rand(n)
    q./=sum(q)
end
Q_test = hcat(q1_test,q2_test)

##


D.cost_dot_x(x_test,q1_test)
D.cost_dot_x(x_test,q2_test) + D.cost_dot_x(x_test,q1_test)
D.cost_dot_x(x_test,Q_test)
D.dcost_dot_x(x_test,q1_test)
D.dcost_dot_x(x_test,q2_test) + D.dcost_dot_x(x_test,q1_test)
D.dcost_dot_x(x_test,Q_test)

D.cost_dot_x(x_test,Q_test)

## test Dirichlet
D.cost_dirich_x(x_test,q1_test)
D.cost_dirich_theta_full(D.theta_of_x(x_test),q1_test)

D.dcost_dirich_x(x_test,q1_test)

_test = D.gradient_test_cost_dirich(x_test,q1_test)
_test.err

##
alphas_test = fill(0.1,n)
alloc =  D.DirichObjAlloc(Q_test)
D.objective_dirmodel(true,x_test,Q_test,alphas_test,alloc)


_gtest = D.gradient_test_dirmodel(x_test,Q_test,alphas_test)
_gtest.err
