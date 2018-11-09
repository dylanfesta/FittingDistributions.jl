using Revise
using FittingDistributions
using Random, Distributions
using Statistics,LinearAlgebra, StatsBase

const D = FittingDistributions

##
n_reps_train=33
n_reps_test = 31
n_reps = n_reps_train + n_reps_test
n_models= 8
n_views=10


alphas_test = fill(0.1,n_models)
ground_idx=6
bins,qvals,mock_data= D.mock_fit_problem(n_reps,n_models, n_views;
        ground_d = ground_idx)

mock_data_train = mock_data[1:n_reps_train,:]
mock_data_test = mock_data[n_reps_train+1:end,:]

fit_problem = D.SpkToFit(mock_data_train,bins,qvals,D.DFitDirich(alphas_test))

q_data = D.get_Q_ofdata(fit_problem)
D.set_stan_folder("/home/dfesta/.cmdstan-2.18.0")
alphas = D.sample_posterior(fit_problem,1000)

alphas_guess = mean(alphas;dims=2)|> vec

##
using Plots
bar(alphas_guess)

##

# LL 0 model (sucks!)
D.logprob_data_uniform(mock_data_test,bins)
# LL oracle (best!)
D.logprob_data(mock_data_test,bins, qvals[:,ground_idx,:])
fake_theta = let  o = fill(0.0,n_models)
      o .+= 0.00001
      o[ground_idx]=1.0
      o ./= sum(o)
end
D.logprob_data(mock_data_test,bins, qvals, fake_theta ,
      fit_problem.fittype)
# LL guess
D.logprob_data(mock_data_test,fit_problem,alphas_guess,D.NoPrior())
D.logprob_data(mock_data_test,fit_problem,alphas_guess,fit_problem.fittype)
