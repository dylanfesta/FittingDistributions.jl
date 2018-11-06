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

## test Dirichlet, a dry run and a gradient test
alphas_test = fill(0.1,n)
alloc =  D.DirichObjAlloc(Q_test)
D.objective_dirmodel(true,x_test,Q_test,alphas_test,alloc)
_gtest = D.gradient_test_dirmodel(x_test,Q_test,alphas_test)
_gtest.err


#####
## Now grad test of optimizer for more realistic data!

n_reps= 5
n_models= 12
n_views= 3
alphas_test = fill(0.1,n_models)
bins,qvals,mock_data= D.mock_fit_problem(n_reps,n_models, n_views;
        ground_d = 7)

fit_problem = D.SpkToFit(mock_data,bins,qvals,D.DFitDirich(alphas_test))

x_test = randn(n_models)

##
fit_problem
Q_test = D.get_Q_ofdata(fit_problem)

_grad_test = D.gradient_test_dirmodel(x_test,fit_problem)
_grad_test.err

# so far so good... solve the objective!
theta_start = fill(1/n_models,n_models)
whatevs = D.optimize_dirichlet_mixture(theta_start, fit_problem ;
    function_tolerance=1E-2)

D.theta_of_x([0.0,0,0,0,0,0])
