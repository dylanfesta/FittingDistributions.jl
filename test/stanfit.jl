
using Revise
using FittingDistributions
using Random, Distributions
using StatsBase

const D = FittingDistributions

##

bins,qvals,mock_data= D.mock_fit_problem(33,8,21)

fit_problem = D.SpkToFit(mock_data,bins,qvals)


q_data = D.get_Q_ofdata(fit_problem)
