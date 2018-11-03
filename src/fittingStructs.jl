

"""
    struct SpkToFit
      spks::Matrix{Float64}
      spk_bins::Vector{Float64}
      model_distr:: Array{Float64,3}
    end
# variables
  - `spks` :  spike counts, dim 1 are single trial spike counts, dim 2 are inputs
             if number of trials can vary, use missing
  - `spk_bins` : boundaries of each bin, must be ordered, histograms include the left
  - `model_distr` : output distributions for each model for each stimulus
       dim 1 are probabilities for each bin, dim 2 are models,
      dim 3 are unique inputs (views). The first dimension must sum to 1
"""
struct SpkToFit{T}
  spks::Matrix{T}
  spk_bins::Vector{Float64}
  model_distr:: Array{Float64,3}
  function  SpkToFit(spks,spk_bins,model_distr)
    tot_prob =  sum(model_distr,dims=1)
    @assert all( isapprox.(tot_prob,1.0,atol=1E-5) ) "probability must sum to 1 ! "
    #@assert all( spks .>= 0.0 ) "spike counts should be positive"
    new{eltype(spks)}(spks,spk_bins,model_distr)
  end
end
n_models(s::SpkToFit) = size(s.model_distr,2)
n_bins(s::SpkToFit) = length(spk_bins)-1
"""
    function discrete_distribution(data::Vector,bins::Vector)

Bins the data, returning the probability of an element in the interval
 [bins[i-1] , bins[1] )   (closed on the left)
"""
function discrete_distribution(data::Vector,bins::Vector)
    h=fit(Histogram, bins, closed = :left )
    hn = normalize(h,mode=:probability)
    hn.weights
end

# simply returns the index of the bin where each datapoint falls
# if there is missing data, it just ignores it
# also ignores data outside the boundaries
function bin_idx(data::AbstractVector{T},bins::Vector{Float64}) where T
  r_max = bins[end]
  _data = filter(data) do d
    !ismissing(d) && d < r_max
  end
  out=similar(_data,Int32)
  for (i,dat) in enumerate(_data)
   out[i] = findfirst( b -> b>dat, bins)-1
  end
  out
end

# spikecounts for a single image, distributions are over K total models
#
function get_Q_ofdata(spikecounts::AbstractVector{T},
  q_img::AbstractMatrix,bins::AbstractVector) where T
  mybins = bin_idx(spikecounts,bins)
  @assert size(q_img,1) == length(bins)-1
  q_img[mybins,:] #over all models and all trials!
end

function get_Q_ofdata(spikecounts_all::AbstractMatrix{T} ,
      q_all::AbstractArray{Float64,3},
      bins::AbstractVector) where T
  n_bins,n_models,n_views = size(q_all)
  out = map(1:n_views) do vv
    get_Q_ofdata(
      view(spikecounts_all,:,vv) ,
      view(q_all,:,:,vv), bins)
  end
  n_trials = length(out[1])
  out_t = Matrix{Float64}(undef,n_models,n_trials)
  for i in 1:n_models
    out_t[i,:]=out[i]
  end
  out_t
end

get_Q_ofdata(s::SpkToFit) = get_Q_ofdata(s.spks,s.model_distr,s.spk_bins)
