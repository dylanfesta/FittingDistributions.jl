using Revise
using FittingDistributions
using Random, Distributions
using StatsBase

const D = FittingDistributions

##
bins = LinRange(-5,5,51) |> collect
binsc = midpoints(bins)

d1 = Normal(0,0.6)
d2 = Rayleigh(1.2)
d3 = Normal(0,0.2)


qmat = let temp = [pdf.(dd,binsc) for dd in [d1,d2,d3] ],
  temp = hcat(temp...)
  broadcast!(/,temp,temp, sum(temp;dims=1))
end

targ = Normal(0.5,0.6)
p_targ = pdf.(targ,binsc)
broadcast!(/,p_targ,p_targ,sum(p_targ))

## test 0 (sigh)
D.test_gradient_repar(0.11)

## test 1 , objective function higher for right distribution
D.kl_objective_fun(Float64[],[1.,-100.,-100],qmat,p_targ, D.KLStandard())
D.kl_objective_fun(Float64[],[100.,-100.,-100],qmat,p_targ,D.KLStandard())

# test 2 , gradient
betas=[-4.4 , 3.3 , 10.]
testgrad = D.test_gradient(betas,qmat,p_targ)
testgrad.error

## now test gradient of reverse case

betas=[1 , -1. , 33.33]
testgrad = D.test_gradient(betas,qmat,p_targ,D.KLReverse())
testgrad.error


# test 3 , more samples!

d4 = Normal(1,0.6)
d5 = Rayleigh(3.33)
d6 = Normal(-1,2.0)
targ2 = LogNormal(3,2)

qmats = let temp = [pdf.(dd,binsc) for dd in [d4,d5,d6] ],
  temp = hcat(temp...)
  q2 = broadcast!(/,temp,temp, sum(temp;dims=1))
  [qmat,q2]
end
p_targs = let
  v = pdf.(targ,binsc)
  broadcast!(/,v,v,sum(v))
  [p_targ,v]
end


betas=[1.23 , -10.345 , 0.001]
D.kl_objective_fun(Float64[],betas,qmats,p_targs,D.KLStandard())
testgrad = D.test_gradient(betas,qmats,p_targs,D.KLStandard())
testgrad.error
testgrad = D.test_gradient(betas,qmats,p_targs,D.KLReverse())
testgrad.error
