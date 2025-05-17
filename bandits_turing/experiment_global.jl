# Environment setup
using Pkg
Pkg.activate(".")
Pkg.add(["Turing", "Distributions", "StatsPlots", "Random", "ReverseDiff"])

# Imports
using Turing
using Turing: MH
using Distributions
using StatsPlots
using Random
using ReverseDiff

Random.seed!(42)

# ==================== MODEL DEFINITIONS ====================

# Advanced Model: Global HMM State Across Arms
@model function hmm_bandit_a1(y)
    N = length(y)
    mu0 ~ Beta(1, 1)
    mu1 ~ Beta(1, 1)
    pi0 ~ Beta(1, 1)
    pi1 ~ Beta(1, 1)

    z = Vector{Int}(undef, N)
    z[1] ~ Bernoulli(0.5)
    for t in 2:N
        z[t] ~ Bernoulli(z[t-1] == 1 ? pi1 : pi0)
    end

    for t in 1:N
        y[t] ~ Bernoulli(z[t] == 1 ? mu1 : mu0)
    end
end

@model function hmm_bandit_a2(y)
    N = length(y)

    # Prior from previous posterior estimates could be something like this:
    mu0 ~ Beta(3.5, 1.5)  # Previously: mean about 0.7
    mu1 ~ Beta(1.5, 3.5)  # Previously: mean about 0.3
    pi0 ~ Beta(8, 2)  # Tendency to stay in state 0
    pi1 ~ Beta(8, 2)  # Tendency to stay in state 1

    z = Vector{Int}(undef, N)
    z[1] ~ Bernoulli(0.5)
    for t in 2:N
        z[t] ~ Bernoulli(z[t-1] == 1 ? pi1 : pi0)
    end

    for t in 1:N
        y[t] ~ Bernoulli(z[t] == 1 ? mu1 : mu0)
    end
end

# ==================== THOMPSON SAMPLING ALGORITHMS ====================

# Common Dynamics Model (Advanced TS)
function run_adv_ts_experiment(K, N, mu, pi_mat)
    z_true = zeros(Int, N)
    z_true[1] = rand(Bernoulli(0.5))
    for t in 2:N
        z_true[t] = rand(Bernoulli(pi_mat[z_true[t-1]+1, 2]))
    end

    y_true = zeros(Int, K, N)
    for k in 1:K, t in 1:N
        y_true[k, t] = rand(Bernoulli(mu[k, z_true[t]+1]))
    end

    reward = Vector{Float64}(undef, N)
    regret = Vector{Float64}(undef, N)
    observed = [Int[] for _ in 1:K]

    for t in 1:N
        sampled_thetas = zeros(K)
        for k in 1:K
            if length(observed[k]) % 20 == 0 && length(observed[k]) >= 20
            #if length(observed[k]) >= 5 && t % 5 == 0 # Update model every 5 rounds
            #if length(observed[k]) >= 2
                model = hmm_bandit_a2(observed[k])
                chain = sample(model, MH(), 1200; progress=false)
                burned_chain = chain[1001:end, :, :]
                mu0_post = mean(burned_chain[:mu0])
                mu1_post = mean(burned_chain[:mu1])
                z_pred = rand(Bernoulli(0.5))
                for _ in 1:3
                    z_pred = rand(Bernoulli(z_pred == 1 ? mean(burned_chain[:pi1]) : mean(burned_chain[:pi0])))
                end
                sampled_thetas[k] = z_pred == 1 ? mu1_post : mu0_post
            else
                sampled_thetas[k] = rand()
            end
        end

        arm = argmax(sampled_thetas)
        r = y_true[arm, t]
        push!(observed[arm], r)
        reward[t] = r

        expected_rewards = [mu[k, z_true[t]+1] for k in 1:K]
        regret[t] = maximum(expected_rewards) - mu[arm, z_true[t]+1]
    end

    return cumsum(reward), cumsum(regret)
end

# Standard Thompson Sampling (No Latent States)
function run_standard_ts(K, N, mu, pi_mat)
    z_true = zeros(Int, N)
    z_true[1] = rand(Bernoulli(0.5))
    for t in 2:N
        z_true[t] = rand(Bernoulli(pi_mat[z_true[t-1]+1, 2]))
    end

    y_true = zeros(Int, K, N)
    for k in 1:K, t in 1:N
        y_true[k, t] = rand(Bernoulli(mu[k, z_true[t]+1]))
    end

    alpha, beta = ones(K), ones(K)
    reward = Vector{Float64}(undef, N)
    regret = Vector{Float64}(undef, N)

    for t in 1:N
        thetas = rand.(Beta.(alpha, beta))
        arm = argmax(thetas)
        r = y_true[arm, t]
        reward[t] = r
        alpha[arm] += r
        beta[arm] += 1 - r

        expected_rewards = [mu[k, z_true[t]+1] for k in 1:K]
        regret[t] = maximum(expected_rewards) - mu[arm, z_true[t]+1]
    end

    return cumsum(reward), cumsum(regret)
end

# ==================== RUN & PLOT ====================

num_runs = 1
#K, N = 3, 1000
#mu = [0.1 0.95; 0.3 0.75; 0.95 0.1]
K, N = 2, 5000
mu = [0.05 0.95; 0.95 0.05]
pi_mat = [0.9 0.1; 0.1 0.9]

reward_adv = zeros(Float64, num_runs, N)
regret_adv = zeros(Float64, num_runs, N)
reward_std = zeros(Float64, num_runs, N)
regret_std = zeros(Float64, num_runs, N)

for i in 1:num_runs
    r_adv, g_adv = run_adv_ts_experiment(K, N, mu, pi_mat)
    r_std, g_std = run_standard_ts(K, N, mu, pi_mat)
    reward_adv[i, :] .= r_adv
    regret_adv[i, :] .= g_adv
    reward_std[i, :] .= r_std
    regret_std[i, :] .= g_std
end

avg_reward_adv = mean(reward_adv, dims=1)[:]
avg_regret_adv = mean(regret_adv, dims=1)[:]
avg_reward_std = mean(reward_std, dims=1)[:]
avg_regret_std = mean(regret_std, dims=1)[:]

# Plotting
plot(1:N, avg_regret_adv, label="Common Dynamic TS", xlabel="Time", ylabel="Cumulative Regret", lw=2)
plot!(1:N, avg_regret_std, label="Baseline Model TS", lw=2)
savefig("avg_cum_regret_comparison.png")

plot(1:N, avg_reward_adv, label="Common Dynamic TS", xlabel="Time", ylabel="Cumulative Reward", lw=2)
plot!(1:N, avg_reward_std, label="Baseline Model TS", lw=2)
savefig("avg_cum_reward_comparison.png")
