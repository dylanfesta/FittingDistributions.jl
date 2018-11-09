
using Revise
using FittingDistributions
using Random, Distributions
using Statistics,LinearAlgebra, StatsBase

const D = FittingDistributions

##
n_reps=25
n_models= 8
n_views=10

alphas_test = fill(0.1,n_models)
bins,qvals,mock_data= D.mock_fit_problem(n_reps,n_models, n_views;
        ground_d = 6)

mock_data_miss =  let mk = convert(Array{Union{Float64,Missing}},mock_data)
  mk[3:end,[1,4,6]].=missing
  mk
end

fit_problem = D.SpkToFit(mock_data_miss,bins,qvals,D.DFitDirich(alphas_test))

q_data = D.get_Q_ofdata(fit_problem)
D.set_stan_folder("/home/dfesta/.cmdstan-2.18.0")
alphas = D.sample_posterior(fit_problem,1000)

alphas_guess = mean(alphas;dims=2)|> vec

##
using Plots

bins = LinRange(0,1.2,20) |> collect

pp = let
  plot()
  for p=1:8
    h = fit(Histogram, selectdim(alphas,1,p), bins, closed=:left) |> normalize
    bar!(h,opacity=0.3)
  end
  plot!()
end

bar(alphas_guess)
##
