#=

Use probabilistic inference and Dirichle distribution,
write all the gradients manually for constrained maximization of the objective
=#

theta_of_x(xi::AbstractFloat,expsum::AbstractFloat) = exp(xi)/expsum
function theta_of_x!(xin::AbstractVector,x::AbstractVector)
  expsum = mapreduce(exp,+,x)
  broadcast!(theta_of_x,xin,x,expsum)
end
theta_of_x(x::AbstractVector) = theta_of_x!(similar(x),x)

x_of_theta(theta::AbstractVector) = log.(theta)

function dtheta_of_x!(mat_in,x)
  expsum=mapreduce(exp,+,x)
  expsmsq=expsum*expsum
  broadcast!( (xi,xj)-> -exp(xi)*exp(xj)/expsmsq,
      mat_in, x,transpose(x))
  for (i,_x) in enumerate(x)
    mat_in[i,i] += exp(_x)/expsum
  end
  mat_in
end

function dtheta_of_x(x)
  n = length(x)
  dtheta_of_x!(Matrix{eltype(x)}(undef,n,n),x)
end

function gradient_test_theta_of_x(x)
  n = length(x)
  out_an = dtheta_of_x(x)
  out_num = Matrix{eltype(x)}(undef,n,n)
  x_alloc=similar(x)
  for k in 1:n
    fk(x) = theta_of_x!(x_alloc,x)[k]
    out_num[k,:] = Calculus.gradient(fk,x)
  end
  (num=out_num,an=out_an, err = (@. 2.0(out_num-out_an)/(out_num+out_an+eps(100.0))) )
end

## Full cost , and gradient!
"""
  - `h` : is the sum over  models weighted over `th` for  each stimulus
"""
struct DirichObjAlloc{M,V}
  th_alloc::V
  h_alloc::V
  dth_alloc::M
  dmat_alloc::M
  grad_alloc::V
end

function DirichObjAlloc(Q::AbstractMatrix)
  n_models,n_stims = size(Q)
  dmat=similar(Q)
  h = Vector{Float64}(undef,n_stims)
  th = Vector{Float64}(undef,n_models)
  grad = Vector{Float64}(undef,n_models)
  dth = Matrix{Float64}(undef,n_models,n_models)
  DirichObjAlloc(th,h,dth, dmat ,grad)
end

"""
Unormalized log probability of coefficients taken as a single
extraction from a Dirichlet with prior `alphas` and
turned into a mixture of the model which have probability
for the data expressed by `Q`
  -  rows of `Q` are modes, columns are stimuli, values represent the probability
    of each stimulus in the model corresponding to the row
"""
function objective_dirmodel( withgradient::Bool,
    x , Q::AbstractMatrix , alphas, p::DirichObjAlloc)
  th = theta_of_x!(p.th_alloc,x)
  h_vec = mul!(p.h_alloc,transpose(Q),th)
  cost = 0.0
  cost += mapreduce(log,+,h_vec)
  # now the unnormalized Diriclet part
  for (_th,al) in zip(th,alphas)
    cost += (al-1.0)*log(_th)
  end
  if !withgradient
    return cost
  end
  # now the gradient
  dth = dtheta_of_x!(p.dth_alloc,x)

  # comparison to Q part
  mul!(p.dmat_alloc, dth, Q)
  grad = p.grad_alloc
  for (t,h) in enumerate(h_vec)
    BLAS.axpy!(inv(h),view(p.dmat_alloc,:,t) , grad)
  end
  # Dirichlet prior part
  invvect = @. (alphas-1.0)/th
  @assert all(isfinite.(invvect))
  BLAS.gemv!('N',1.0,dth,invvect,1.0,grad)

  cost
end

function gradient_test_dirmodel(x,Q,alphas)
  alloc = DirichObjAlloc(Q)
  an = let
    _ = objective_dirmodel(true,x,Q,alphas,alloc)
    copy(alloc.grad_alloc)
  end
  fg(_x) = objective_dirmodel(false,_x,Q,alphas,alloc)
  num = Calculus.gradient(fg,x)
  (num=num,an=an,
      err = (@. 2.0(num-an)/(num+an+eps(100.0))) )
end

#  minimizer using ... NLOpt ?

function optimize_dirichlet_mixture_costfun(g,x,Q,alphas,alloc::DirichObjAlloc)
  withgrad=!isempty(g)
  c = objective_dirmodel(withgrad,x,Q,alphas,alloc)
  withgrad && copyto!(g,alloc.grad_alloc)
  c
end

function optimize_dirichlet_mixture(theta_start, Q , alphas;
    max_time=-1,function_tolerance=1E-5)
  x_start = x_of_theta(theta_start)
  nd=length(x_start)
  alloc = DirichObjAlloc(Q)
  _obj_fun(g,x) = optimize_dirichlet_mixture_costfun(g,x,Q,alphas,alloc)
  opt = Opt(:LD_LBFGS,nd)
  max_time >0 && maxtime!(opt,60.0*max_time)
  # lower_bounds!(opt,fill(-30.0,nd) )
  # upper_bounds!(opt,fill(30.0,nd))
  # vector_storage!(opt,10)
  ftol_abs!(opt,function_tolerance)
  max_objective!(opt,_obj_fun)
  minf,minx,ret = optimize(opt,x_start)
  minth = theta_of_x(minx)
  (minf,minth,minx,ret)
end

# using the fit problem object !
function optimize_dirichlet_mixture(theta_start, fit_problem::SpkToFit;
      max_time=-1,function_tolerance=1E-5)
  Q = get_Q_ofdata(fit_problem)
  alphas = fit_problem.fittype.alpha
  optimize_dirichlet_mixture(theta_start, Q , alphas;
  max_time=max_time,function_tolerance=function_tolerance)
end

# meh, should test the gradient
function gradient_test_dirmodel(x,fit_problem::SpkToFit)
  Q = get_Q_ofdata(fit_problem)
  alphas = fit_problem.fittype.alpha
  alloc = DirichObjAlloc(Q)
  _cost_fun(g,_x) = optimize_dirichlet_mixture_costfun(g,_x,Q,alphas,alloc)
  an=similar(x)
  _ = _cost_fun(an,x)
  fg(_x) = _cost_fun([],_x)
  num=Calculus.gradient(fg,x)
  (num=num,an=an,
      err = (@. 2.0(num-an)/(num+an+eps(100.0))) )
end
