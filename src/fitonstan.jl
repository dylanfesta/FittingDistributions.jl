


global stan_folder = ""

function set_stan_folder(folder)
    global stan_folder=folder
    set_cmdstan_home!(folder)
end


# general functions to read data of any dimesion from Stan
function get_stan_data_fun(data::Array{Float64},field_names::Vector{String})
  function out(str::String)
    idx=findfirst(s->s==str,field_names)
    idx ==0 && error("cannot read data, the name $str is not in the database")
    vec(data[:,idx,:])
  end
end

"""
    get_data_all(data::Array{Float64,3}, datanames::Vector{String})

Reads all elements or a matrix or a vector with indexes specified
by dims.
# Example
```julia
    get_data = get_data_all(standata,stannames)
    alpha = get_data("alpha",100)
    some_matrix = get_data("some_matrix",50,100)
```
"""
function get_data_all(data::Array{Float64,3},datanames::Vector{String})
    get_data=get_stan_data_fun(data,datanames)
    function f_out(data_name::String,dims::Integer...)
        nd=length(dims)
        n_sampl=get_data(data_name * ".1"^nd) |> length
        out=Array{Float64}(undef,dims...,n_sampl)
        to_iter = Iterators.product([(1:d) for d in dims]...)
        for ijk in to_iter
            _str=data_name
            for i in ijk
                _str*=".$i"
            end
            out[ijk...,:] = get_data(_str)
        end
        out
    end
end
# this stan model requires the pre computation of the discrete probability of
# each datapoint by binning.
# to avoid singularities, make sure to add at least one uniform model to the Qs
const stan_discrete_distr_fit = """
    data {
        int<lower=1> K; // number of models
        int<lower=0> T; // number of spike counts
        // Q_kt is the probability of spike count t under model k
        matrix<lower=0>[K,T] Q ;
        vector<lower=0>[K] theta; // Dirichelet prior
    }
    parameters {
        simplex[K] alphas; // coefficients for each model
    }
    model {
        target += dirichlet_lpdf(alphas|theta); // flipped notation Vs Stan documentation
        for (t in 1:T)
            target += log( dot_product(alphas,Q[:,t]));
    }
"""

# function

function sample_posterior(fit_problem::SpkToFit, n_samples ;
        dir_prior=0.1,
        thin_val=2,
        nchains=4,
        pdir=joinpath(@__DIR__(),"../tmp") )
    @assert !isempty(stan_folder) "Please set the folder of cmdstan using set_stan_folder"
    println("the following temporary directory will be used" * pdir)
    n_mod = n_models(fit_problem)
    q_vals = get_Q_ofdata(fit_problem)
    @assert n_mod == size(q_vals,1)
    n_trials = size(q_vals,2)
    dir_prior = fill(dir_prior,n_mod)
    Data=[  Dict("K"=>n_mod,"T"=>n_trials,
                    "Q"=>q_vals,"theta"=>dir_prior ) for i in (1:nchains) ]
    stanmodel = Stanmodel(num_samples=n_samples,
            thin=thin_val, name="dirich_fit",
            model=stan_discrete_distr_fit ,
            pdir=pdir)
    sim = stan(stanmodel, Data)
    get_data = get_data_all(sim[2],sim[3])
    get_data("alphas",n_mod)
end
