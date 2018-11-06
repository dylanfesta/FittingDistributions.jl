
using Revise
using FittingDistributions
using Random, Distributions
using Statistics,LinearAlgebra, StatsBase

const D = FittingDistributions

##
n_reps=5
n_models= 8
n_views=30

alphas_test = fill(0.1,n_models)
bins,qvals,mock_data= D.mock_fit_problem(n_reps,n_models, n_views;
        ground_d = 3)

fit_problem = D.SpkToFit(mock_data,bins,qvals,D.DFitDirich(alphas_test))

q_data = D.get_Q_ofdata(fit_problem)
D.set_stan_folder("/home/dfesta/.cmdstan-2.18.0")
alphas = D.sample_posterior(fit_problem,1000;
              dir_prior=0.01)

##
using Plots

bins = LinRange(0,1.2,20) |> collect

pp = let
  plot()
  for p=1:8
    h = fit(Histogram, selectdim(alphas,1,p), bins, closed=:right) |> normalize
    bar!(h,opacity=0.3)
  end
  plot!()
end
plot(pp)

histogram(alphas[2,:],opacity=0.5,leg=false,nbins=10)
histogram!(alphas[2,:],opacity=0.5,leg=false)
histogram!(alphas[3,:],opacity=0.5,leg=false)
histogram!(alphas[4,:],opacity=0.5,leg=false)
histogram!(alphas[5,:],opacity=0.5,leg=false)
histogram!(alphas[6,:],opacity=0.5,leg=false)
histogram!(alphas[7,:],opacity=0.5,leg=false)
histogram!(alphas[8,:],opacity=0.5,leg=false)
