
using Revise
using FittingDistributions
using Random, Distributions
using StatsBase

const D = FittingDistributions

##
n_reps=60
n_models= 8
n_views=30
bins,qvals,mock_data= D.mock_fit_problem(n_reps,n_models,n_views)

fit_problem = D.SpkToFit(mock_data,bins,qvals)

q_data = D.get_Q_ofdata(fit_problem)
D.set_stan_folder("/home/dfesta/.cmdstan-2.18.0")
alphas = D.sample_posterior(fit_problem,1000;
              dir_prior=0.01)

##
using Plots

pp = let
  plot()
  for p=1:8
    histogram!(alphas[p,1:5:end],opacity=0.5,leg=false)
  end
end

histogram(alphas[1,:],opacity=0.5,leg=false)
histogram!(alphas[2,:],opacity=0.5,leg=false)
histogram!(alphas[3,:],opacity=0.5,leg=false)
histogram!(alphas[4,:],opacity=0.5,leg=false)
histogram!(alphas[5,:],opacity=0.5,leg=false)
histogram!(alphas[6,:],opacity=0.5,leg=false)
histogram!(alphas[7,:],opacity=0.5,leg=false)
histogram!(alphas[8,:],opacity=0.5,leg=false)
