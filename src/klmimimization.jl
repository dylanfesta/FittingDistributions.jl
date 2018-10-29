

#=
Notation:
alphas are the coefficients of fitting  distributions, betas are non-bound
reparametrizations of alphas
=#

abstract type KLCostType end
struct KLStandard <: KLCostType end
struct KLReverse <: KLCostType end
struct KLSymmetric <: KLCostType end

abstract type AlphaRegus end
struct AlphaNone <: AlphaRegus end
struct AlphaSparse <: AlphaRegus
  c::Float64
end


function _alpha_regularizer(g,alphas,regu::AlphaNone)
  return 0.0 # meh
end

# 1-norm, it's positive, so it's just the sum, and gradient is 1.0 always
function _alpha_regularizer(g,alphas,regu::AlphaSparse)
  error("Work in progress!")
end

function repar(β::AbstractFloat)
    log(1+exp(β))  #this is beta
end
function drepar(β::AbstractFloat)
    inv(1+exp(-β))
end
function invrepar(α::AbstractFloat)
    log(exp(α)-1)
end

function test_gradient_repar(x)
  g_num = Calculus.gradient(repar,x)
  g_an = drepar(x)
  (g_num,g_an,g_num-g_an)
end


function columns_weighted_sum(mat,col_coef)
  out = fill(zero(eltype(mat)),size(mat,1)) # same as fill(0.0, etc ... )
  for (k,c) in enumerate(col_coef)
      m=view(mat,:,k)
      BLAS.axpy!(c,m,out)
  end
  out
end



"""
Main objective function , KL can be standard or reverse (both not implemented yet)
"""
function kl_objective_fun(g,betas::Vector,Q::Matrix,p_target::Vector ,
    (kltype::KLCostType)=KLStandard(),
    (alpharegu::AlphaRegus)=AlphaNone() )
  alphas=repar.(betas)
  dalphas = drepar.(betas)
  qfit = columns_weighted_sum(Q,alphas)
  # this function is different depending on the kltype
  cost = _kl_objective(g,qfit,alphas,Q, p_target,kltype)
  # add alpha regularizer
  alpha_cost = _alpha_regularizer(g,alphas,alpharegu)
  # reparametrization
  !isempty(g) && broadcast!(*,g,g,dalphas)
  cost+alpha_cost
end

function _kl_objective(g,qfit,alphas,Q,p_target, kltype::KLStandard)
  asum=sum(alphas)
  # scale by sum and take the log  q_fit <- log(q_fit)
  q_fit_log = @. log(qfit/asum)
  # sum over elements of p_target
  cost = dot(p_target,q_fit_log)
  for k in 1:length(g) # can probably be vectorized
    q_ik_nrm = Q[:,k] ./ qfit
    g[k] = dot(p_target, q_ik_nrm) - inv(asum)
  end
  cost
end

# reverse KL case
function _kl_objective(g,qfit,alphas,Q,p_target, kltype::KLReverse)
  asum=sum(alphas)
  q_fit_nrm = qfit ./ asum
  q_norm_targ = q_fit_nrm ./ p_target
  cost= sum(@. q_fit_nrm*log( q_norm_targ))
  if !isempty(g)
      _mult = @. (log(q_norm_targ)+1)/asum^2
      _rest=@. Q*asum - qfit
      mul!(g, _rest' , _mult)
  end
  cost
end

function test_gradient(betas,Q,p_target,
      (kltype::KLCostType)=KLStandard() )
  g_an=zero(betas)
  kl_objective_fun(g_an,betas,Q,p_target,kltype)
  f_grad(bb) = kl_objective_fun(Float64[],bb,Q,p_target,kltype)
  g_num = Calculus.gradient(f_grad,betas)
  g_err =@.  2abs(g_an-g_num)/(g_an+g_num+eps(100.0))
  (analytic=g_an,numerical=g_num,error=g_err)
end


# Now there are several targets , and several train examples, all in vectors
function kl_objective_fun(g,betas,Qs::Vector{Matrix{T}},
        p_targets::Vector{Vector{T}},
        kltype::KLCostType,
        ) where T<:AbstractFloat
  cost=0
  fill!(g,0.0)
  g_add=zero(g)
  g_temp=similar(g_add)
  for (Q,p_targ) in zip(Qs,p_targets)
      cost += kl_objective_fun(g_temp,betas,Q,p_targ,kltype)
      g .+= g_temp # if g is empty, nothing will happen
  end
  return cost
end

#
# # function get_Qdims(Q::Vector)
# #     size(Q[1],2)
# # end
# # function get_Qdims(Q::Matrix)
# #     size(Q,2)
# # end
# #
# function get_qdistr(alphas,Q::Matrix)
#     alphas_norm=alphas./sum(alphas)
#     Q*alphas_norm
# end
# function get_qdistr(alphas,Q::Vector)
#     map(qq->get_qdistr(alphas,qq),Q)
# end
#
# function minimize_KL_optim(Q,targ ; par_start=[])
#     npar=get_Qdims(Q)
#     ftomin(pars) = -kl_cost_fun([],pars,Q,targ)
#     function ftomin_grad(g,pars)
#         kl_cost_fun(g,pars,Q,targ)
#         g.*=-1
#         nothing
#     end
#     x_start= isempty(par_start) ? log.(rand(npar)*0.1) : par_start
#     out=Optim.optimize(ftomin,ftomin_grad,x_start,BFGS())
#     exp.(out.minimizer)
# end
#
#
# function minimize_KL_nlopt(Q,targ ; par_start=[])
#     npar=get_Qdims(Q)
#     opt=Opt(:LD_LBFGS,npar)
#     ftomax(g,pars) = kl_cost_fun(g,pars,Q,targ)
#     max_objective!(opt,ftomax)
#     x_start= isempty(par_start) ? log.(rand(npar)*0.1) : par_start
#     (optf,optx,ret)=NLopt.optimize(opt,x_start)
#     @show ret
#     exp.(optx)
# end
#
#
# function minimize_KL_rev(Q,targ;par_start=[])
#     npar=get_Qdims(Q)
#     ftomin(pars) = kl_cost_fun_rev([],pars,Q,targ)
#     ftomin_grad(g,pars)=kl_cost_fun_rev(g,pars,Q,targ)
#     x_start= isempty(par_start) ? log.(rand(npar)*0.1) : par_start
#     out=Optim.optimize(ftomin,ftomin_grad,x_start,BFGS())
#     exp.(out.minimizer)
# end
#
# function KL_divergence(p,q;forward=true)
#     forward ? sum( @. p*log(p/q)) : KL_divergence(q,p;forward=true)
# end
#
# function information_loss(target,guess,uniform)
#     KL_divergence(target,guess) / KL_divergence(target,uniform)
# end
#
# function information_loss(Qs,Qs_test,targets,targets_test,uniform)
#     alphas=minimize_KL_optim(Qs,targets)
#     guesses_test=get_qdistr(alphas,Qs_test)
#     losses= map(tg -> information_loss(tg[1],tg[2],uniform),
#                     zip(targets_test,guesses_test) )
#     mean(losses)
# end
