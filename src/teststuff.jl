#=
Scripts that test the quality of the
fit of the mixture model
(this should really be part of FittingDistributions)
=#



# log probability, given the Q matrix computed on data!
function logprob_Qmat(thetas, Q::AbstractMatrix , prior::DFitType,normalize::Bool)
  h_vec = transpose(Q)*thetas
  cost = 0.0
  cost += mapreduce(log,+,h_vec)
  cost += logprob_logprior(thetas,prior)
  if normalize
    cost /= length(h_vec)
  else
    cost
  end
end

logprob_logprior(thetas,p::NoPrior) = 0.0
function logprob_logprior(thetas,p::DFitDirich)
    di=Dirichlet(p.alpha)
    logpdf(di,thetas)
end

# when  no thetas, take first
function logprob_Qmat(Q::AbstractMatrix,prior::DFitType,normalize::Bool)
  @assert size(Q,1)==1 "Need to specify the model!"
  logprob_Qmat([1.0], Q , prior,normalize)
end

"""
function logprob_data(spikecounts,bins,discrete_models,
    (theta::Vector{Float64})=[],
    (prior::DFitType)=NoPrior() ;
    normalize=false)

# inputs
  - `spikecounts` : rows are trials, columns are distinct stimuli
  - `bins` : bins used for the discrete probabilities
  - `discrete_models` : dim 1 is bins , dim 2 is models, dim 3 is stimulus.
          if dim 3 is missing, dim 2 is considered as stimulus.
          if dim 2 is missing, we assume stimulus-independence
  - `theta` mixing components of models , ignored if only 1 model is present
  - `prior` prior over theta
  - `normalize` whether to divide by number of datapoints or not
# output
Log probability of data

"""
function logprob_data(spikecounts,bins,discrete_models,
    (theta::Vector{Float64})=Float64[] ,
    (prior::DFitType)=NoPrior() ; (normalize::Bool)=false )
  Q = get_Q_ofdata(spikecounts,discrete_models,bins)
  size(Q,1)==1 ? logprob_Qmat(Q,prior,normalize) : logprob_Qmat(theta,Q,prior,normalize)
end

"""
  function logprob_data(spikecounts, s::SpkToFit ,
      theta,
      prior_theta = NoPrior ;
      normalize=false)

Takes the bins and the distributions from fit problem object
if no prior is specified, it takes that prior, too.
spikecounts need to be provided (it's a cross validation)
along with the solution (that must sum to 1)
"""
function logprob_data(spikecounts, s::SpkToFit ,
    theta, prior_theta=D.NoPrior() ;
    normalize=false)
    logprob_data(spikecounts ,s.spk_bins, s.model_distr,theta, prior_theta ;
      normalize = normalize )
end

function logprob_data_uniform(spikecounts,bins; normalize=false)
  nb = length(bins)-1
  qdistr = fill(1/nb,nb)
  logprob_data(spikecounts,bins,qdistr,[1.0],NoPrior(); normalize = normalize )
end


"""
    function loglik_score_vs_uni(bins, data_test, target_distrtrib, oracle_distrib,
        theta_guess, prior=NoPrior())
Computes the cross-validation score of the fit, as in
(ll_guess-ll_uniform)/(ll_oracle-ll_uniform)
"""
function loglik_score_vs_uni(bins, data_test, target_distr, oracle_distr,
    theta_guess, prior=NoPrior())
    theta_guess ./= sum(theta_guess)
    ll_guess = logprob_data(data_test,bins, target_distr, theta_guess ,prior)
    ll_uniform  =  logprob_data_uniform(data_test,bins)
    ll_oracle = logprob_data(data_test,bins,oracle_distr)
    (ll_guess-ll_uniform)/(ll_oracle-ll_uniform)
end


function loglik_score_vs_model(bins,data_test, target_distr, null_distr,oracle_distr,
  theta_guess,prior=NoPrior())
  theta_guess ./= sum(theta_guess)
  ll_guess = logprob_data(data_test,bins, target_distr, theta_guess ,prior)
  ll_null  =  logprob_data(data_test,bins,null_distr)
  ll_oracle = logprob_data(data_test,bins,oracle_distr)
  (ll_guess-ll_null)/(ll_oracle-ll_null)
end
