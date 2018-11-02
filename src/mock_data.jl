

# not general enough... but it will do... :-(
function mock_fit_problem(n_samples,n_train,n_views;
    n_bins = 50,
    ground_d = 3,
    uniregu=1E-3)
  stds = repeat(  rand(TruncatedNormal(1.2,0.5,0.0,Inf),n_train) ; outer=(1,n_views))
  means =  [ rand(Normal(i,0.6))  for i in 1:n_train , img in 1:n_views ]
  bins=collect(LinRange(-3,8,n_bins+1))
  binsc=midpoints(bins)
  qvals = let out=Array{Float64}(undef,n_bins,n_train,n_views)
    for i in 1:n_train , j in 1:n_views
      mu,si = means[i,j],stds[i,j]
      temp =  [ pdf.(Normal(mu,si),bc)  for bc in binsc ]
      temp ./= sum(temp)
      temp .+= uniregu
      temp ./= sum(temp)
      out[:,i,j] =  temp
    end
    out
  end

  mock_data  = let _g=ground_d,
      out = Matrix{Float64}(undef,n_samples,n_views)
      for im in 1:n_views
          mu,si = means[_g,im],stds[_g,im]
          out[:,im] = rand(Normal(mu,si),n_samples)
      end
      out
  end
  bins,qvals,mock_data
end
