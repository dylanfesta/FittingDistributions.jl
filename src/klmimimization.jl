

#=
Notation:
alphas are the coefficients of fitting  distributions, betas are non-bound
reparametrizations of alphas

=#

function repar(β::AbstractFloat)
    log(1+exp(β))  #this is beta
end
function drepar(β::AbstractFloat)
    inv(1+exp(-β))
end
function invrepar(α::AbstractFloat)
    log(exp(α)-1)
end


"""
Forward KL cost function with gradient.
g will be filled with the gradient
betas are the coefficients
Q has fit distributions as columns
"""
function kl_cost_fun(g,betas::Vector,Q::Matrix,p_target::Vector)
  alphas=repar.(betas)
  asum=sum(alphas)
  q_fit = similar(p_target)
  for (k,alpha) in enumerate(alphas)
      qk=view(Q,:,k)
      BLAS.axpy!(alpha,qk,q_fit)
  end
  # scale by sum and take the log  q_fit <- log(q_fit)
  q_fit_log = broadcast!(q->log(q/asum),q_fit)
  # sum over elements of p_target
  cost = dot(p_target,q_fit_log)
  for k in 1:length(g)
    q_ik_nrm = Q[:,k] ./ q_fit
    g[k] = dot(p_target, q_ik_nrm) - inv(asum)
    g .*= drepar.(betas)
  end
  return cost
end

function test_gradient(g,betas,Q,p_target)
  g_an=similar(p_target)
  kl_cost_fun(g_an,betas,Q,p_target)
  f_grad(bb) = kl_cost_fun(Float64[],bb,Q,p_target)
  g_num = Calculus.gradient(f_grad,betas)
  g_err =@.  2(g_an-g_num)/(g_an+g_num+eps(100.0))
  (analytic=g_an,numerical=g_num,error=g_err)
end
#
#
# function kl_cost_fun(g,betas,Qs::Vector{Matrix},p_targets::Vector{Vector})
#     @assert length(Qs) == length(p_targets)
#     cost=0
#     g_add=zeros(length(g))
#     _g_temp=copy(g_add)
#     for (Q,p_targ) in zip(Qs,p_targets)
#         cost +=kl_cost_fun(_g_temp,betas,Q,p_targ)
#         if length(g)>0
#             g_add.+=_g_temp
#         end
#     end
#     g[:]=g_add
#     return cost
# end
#
# function kl_cost_fun_rev(g,betas,Q::Matrix,target::Vector)
#     alphas=exp.(betas)
#     _Qa=Q*alphas
#     _asum=sum(alphas)
#     qis=_Qa./_asum
#     cost= sum(@. qis*log( qis/target))
#     if length(g)>0
#         _mult = @. (log(qis)-log(target)+1)/_asum^2
#         _rest=broadcast(-,Q.*_asum,_Qa)
#         g[1:end]=_mult'*_rest
#         g.*=alphas
#     end
#     return cost
# end
#
# function kl_cost_fun_rev(g,betas,Qs::Vector{Matrix},p_targets::Vector{Vector})
#     @assert length(Qs) == length(p_targets)
#     cost=0
#     g_add=zeros(length(g))
#     _g_temp=copy(g_add)
#     for (Q,p_targ) in zip(Qs,p_targets)
#         cost +=kl_cost_fun_rev(_g_temp,betas,Q,p_targ)
#         if length(g)>0
#             g_add.+=_g_temp
#         end
#     end
#     g[:]=g_add
#     return cost
# end
#
# function get_Qdims(Q::Vector)
#     size(Q[1],2)
# end
# function get_Qdims(Q::Matrix)
#     size(Q,2)
# end
#
# function kl_cost_test_grad(idx_test,Q,targ; xstart=[])
#     dims=get_Qdims(Q)
#     kl_cost_test_grad(idx_test,dims,Q,targ,xstart)
# end
# function kl_cost_test_grad(idx_test,dims,Q,targ,xstart)
#     betas= isempty(xstart) ? log.(rand(dims)) : xstart
#     g=Vector{Float64}(dims)
#     betas_p= let x=copy(betas)
#         x[idx_test]+=1E-6
#         x
#     end
#     plus=kl_cost_fun([],betas_p,Q,targ)
#     minus=kl_cost_fun(g,betas,Q,targ)
#     (plus-minus)/1E-6 ,g[idx_test]
# end
#
# function kl_cost_rev_test_grad(idx_test,Q,targ; xstart=[])
#     dims=get_Qdims(Q)
#     kl_cost_rev_test_grad(idx_test,dims,Q,targ,xstart)
# end
# function kl_cost_rev_test_grad(idx_test,dims,Q,targ,xstart)
#     betas= isempty(xstart) ? log.(rand(dims)) : xstart
#     g=Vector{Float64}(dims)
#     betas_p= let x=copy(betas)
#         x[idx_test]+=1E-6
#         x
#     end
#     plus=kl_cost_fun_rev([],betas_p,Q,targ)
#     minus=kl_cost_fun_rev(g,betas,Q,targ)
#     (plus-minus)/1E-6 ,g[idx_test]
# end
#
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
