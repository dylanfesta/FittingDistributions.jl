


global stan_folder = ""

function set_stan_folder(folder)
    global stan_folder=folder
    set_cmdstan_home!(folder)
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
    @assert n_mod == size(q_vals,2)
    n_trials = size(q_vals,1)
    dir_prior = fill(dir_prior,n_mod)
    Data=[  Dict("K"=>n_mod,"T"=>n_trials,
                    "Q"=>q_vals,"theta"=>dir_prior )]
    stanmodel = Stanmodel(num_samples=n_samples,
            thin=thin_val, name="dirich_fit",
            model=stan_discrete_distr_fit ,
            pdir=pdir)
    sim = stan(stanmodel, Data)
    # get_data = get_data_all(sim[2],sim[3])
    # gs = get_data("g",dim,num_data)
    # vs = get_data("v",num_data)
    # (gs=gs,vs=vs)
end
