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





cost_dot_theta(theta,q) = log(dot(theta,q))
function cost_dot_x!(th_alloc,x,q)
  theta_of_x!(th_alloc,x)
  cost_dot_theta(th_alloc,q)
end
cost_dot_x(x,q) = cost_dot_x!(similar(x),x,q)

function dcost_dot_x!(dmat_alloc,th_alloc,
      x::AbstractVector, q::AbstractVector)
  theta_of_x!(th_alloc,x)
  dot_thq = dot(th_alloc,q)
  dtheta_of_x!(dmat_alloc,x)
  (dmat_alloc * q) .* inv(dot_thq)
end
function dcost_dot_x(x,q)
  n=length(x)
  theta_alloc=similar(x)
  dmat_alloc = Matrix{eltype(x)}(undef,n,n)
  dcost_dot_x!(dmat_alloc,theta_alloc,x,q)
end

function gradient_test_cost_dot(x::AbstractVector,q::AbstractVector)
  an = dcost_dot_x(x,q)
  th_alloc = similar(x)
  ftest(_x) = cost_dot_x!(th_alloc,_x,q)
  num = Calculus.gradient(ftest,x)
  (num=num,an=an,
    err = (@. 2.0(num-an)/(num+an+eps(100.0))) )
end

# consider Q as a matrix!
# too lazy to be memory efficient
function cost_dot_x(x,Q::AbstractMatrix)
  th = theta_of_x(x)
  h_vec = transpose(Q) * th
  mapreduce(log,+,h_vec)
end

function dcost_dot_x!(dmat_alloc::AbstractMatrix,
    x::AbstractVector,Q::AbstractMatrix)
  th = theta_of_x(x)
  h_vec_inv = inv.(transpose(Q) * th)
  Th = dtheta_of_x(x)
  mul!(dmat_alloc, Th,Q)
  n_stims = size(Q,2)
  out = zero(th)
  for (t,h_inv) in enumerate(h_vec_inv)
    BLAS.axpy!(h_inv,view(dmat_alloc,:,t) , out)
  end
  out
end
dcost_dot_x(x::AbstractVector,Q::AbstractMatrix) = dcost_dot_x!(
        similar(Q),x,Q)

function gradient_test_cost_dot(x::AbstractVector,Q::AbstractMatrix)
  an = dcost_dot_x(x,Q)
  ftest(_x) = cost_dot_x(_x,Q)
  num = Calculus.gradient(ftest,x)
  (num=num,an=an,
    err = (@. 2.0(num-an)/(num+an+eps(100.0))) )
end


# now the LogDirichlet part of the cost

function cost_dirich_theta(theta::AbstractVector,alphas)
  out=0.0
  for (th,al) in zip(theta,alphas)
    out += (al-1.0)*log(th)
  end
  out
end
function cost_dirich_theta_full(theta::AbstractVector,alphas)
  d=Dirichlet(alphas)
  logpdf(d,theta)
end
cost_dirich_x(x,alphas) = cost_dirich_theta( theta_of_x(x),alphas )

function dcost_dirich_x(x,alphas)
  _th=theta_of_x(x)
  broadcast!( (th,alph)->  (alph-1.0)/th , _th,_th,alphas)
  dth = dtheta_of_x(x)
  dth*_th
end

function gradient_test_cost_dirich(x,alphas)
  an=dcost_dirich_x(x,alphas)
  th_alloc=similar(x)
  fg(_x) = cost_dirich_theta_full(theta_of_x!(th_alloc,_x),alphas)
  num = Calculus.gradient(fg,x)
  (num=num,an=an,
      err = (@. 2.0(num-an)/(num+an+eps(100.0))) )
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
