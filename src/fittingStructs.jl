


"""
This instructs how to fit, and contains the prior informtion
"""
abstract type DFitType end
struct DFitDirich <: DFitType
   alpha::Vector{Float64}
end
struct NoPrior <: DFitType end



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
struct SpkToFit{T,DD}
  spks::Matrix{T}
  spk_bins::Vector{Float64}
  model_distr:: Array{Float64,3}
  fittype::DD
  function  SpkToFit(spks,spk_bins,model_distr,fittype::DFitType)
    tot_prob =  sum(model_distr,dims=1)
    @assert all( isapprox.(tot_prob,1.0,atol=1E-5) ) "probability must sum to 1 ! "
    #@assert all( spks .>= 0.0 ) "spike counts should be positive"
    new{eltype(spks),typeof(fittype)}(spks,spk_bins,model_distr,fittype)
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
  r_min,r_max = bins[1],bins[end]
  _data = filter(data) do d
    !ismissing(d) && (r_min <= d < r_max)
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
  @assert size(q_img,1) == length(bins)-1 "careful size q is  $(size(q_img)) and bins are $(length(bins)) "
  mybins = bin_idx(spikecounts,bins)
  q_img[mybins,:] #over all models and all trials!
end


"""
  function get_Q_ofdata(spikecounts_all::AbstractMatrix{T} ,
        q_all::AbstractArray{Float64,3},
        bins::AbstractVector) where T

# Inputs
  - `spikecounts_all` :  neural (or mock) data, rows are trials, columns are different
     unique views
  - `q_all` : discrete  probability distributions to match, dim 1 is bins, dim 2 is
         models , dim 3 is views
  - `bins` : bins
# output
Matrix Q ,  rows are models, columns are the spikecounts that have been accepted
(total trials). Spikecounts are not accepted if they are missing, or out of the
bounds of bins
"""
function get_Q_ofdata(spikecounts_all::AbstractMatrix{T} ,
      q_all::AbstractArray{Float64,3},
      bins::AbstractVector) where T
  n_bins,n_models,n_views = size(q_all)
  out = map(1:n_views) do vv
    get_Q_ofdata(
      selectdim(spikecounts_all,2,vv) ,
      selectdim(q_all,3,vv), bins)
  end
  # just collect and transpose
  permutedims(vcat(out...))
end

# for generality, here is a version with a single model
# the model is still repreated for each image
function get_Q_ofdata(spikecounts_all::AbstractMatrix{T} ,
      q_all::AbstractArray{Float64,2},
      bins::AbstractVector) where T
  (r,c) = size(q_all)
  q3 = reshape(q_all,r,1,c)
  get_Q_ofdata(spikecounts_all,q3,bins) # this will be a row vector!
end

# meh , the model is the same for all views
# (may as well convert spikecounts into a single vector!)
function get_Q_ofdata(spikecounts_all::AbstractMatrix{T} ,
      q_all::AbstractArray{Float64,1},
      bins::AbstractVector) where T
  n_views = size(spikecounts_all,2)
  q_allm =repeat(q_all,outer=(1,n_views))
  get_Q_ofdata(spikecounts_all,q_allm,bins) # this calls the one above
end


get_Q_ofdata(s::SpkToFit) = get_Q_ofdata(s.spks,s.model_distr,s.spk_bins)
