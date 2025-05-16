# Environment setup
using Pkg
Pkg.activate(".")
Pkg.add(["Turing", "Distributions", "StatsPlots", "Random", "ReverseDiff"])

# Imports
using Turing
using Turing: MH
#using Turing: NUTS, MCMCThreads
using Distributions
using StatsPlots
using Random
using ReverseDiff

#Turing.setadbackend(:reversediff)  # Best for discrete models

# optional, we could do this:
#NUTS(0.65; adtype=ReverseDiffAD())
# or this:
#NUTS(0.65; adtype=ForwardDiffAD())

# Set the number of threads to use (match your physical/virtual cores) 
n_chains = Threads.nthreads()  # Check how many are currently active

# Set seed
Random.seed!(42)

# Problem setup
K, N = 3, 1000
z_true = zeros(Int, N)
z_true[1] = rand(Bernoulli(0.5))
pi_mat = [0.9 0.1; 0.1 0.9]
mu = [0.1 0.9; 0.45 0.6; 0.9 0.1]

for t in 2:N
    z_true[t] = rand(Bernoulli(pi_mat[z_true[t-1]+1, 2]))
end

y_true = zeros(Int, K, N)
for k in 1:K, t in 1:N
    y_true[k, t] = rand(Bernoulli(mu[k, z_true[t]+1]))
end

# Turing model
@model function hmm_bandit(y)
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

# Thompson Sampling loop
counts, rewards = zeros(Int, K), zeros(Int, K)
observed = [Int[] for _ in 1:K]
selected_arms, cum_reward = Int[], Int[]

cum_regret = Int[]

for t in 1:N
    sampled_thetas = zeros(K)
    for k in 1:K
        if length(observed[k]) >= 5 && t % 5 == 0 # Update model every 5 rounds
            model = hmm_bandit(observed[k])
            #chain = sample(
                #model,
                #NUTS(0.65),
                #MCMCThreads(),
                #150, # samples per chain
                #12; # chains
                #num_warmup = 500,
                #progress = false
            #)
            # -> HMC with 12 threads/parallel chains and initial 150 samples, initial step size 0.65
            # -> Discard first 100 samples
            # -> Use MCMCThreads for parallel sampling this will lead to a total of 600 usable samples
            #mu0_post = mean(chain[:mu0])
            #mu1_post = mean(chain[:mu1])
            #z_pred = rand(Bernoulli(0.5))
            #for _ in 1:3
                #z_pred = rand(Bernoulli(z_pred == 1 ? mean(chain[:pi1]) : mean(chain[:pi0])))
            #end
            chain = sample(model, MH(), 550; progress=false) # 100 samples would be more accurate
            burned_chain = chain[501:end, :, :] # Discard first 500 samples
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
    push!(selected_arms, arm)
    r = y_true[arm, t]
    # Compute instantaneous regret
    expected_rewards = [mu[k, z_true[t] + 1] for k in 1:K]
    optimal = maximum(expected_rewards)
    regret = optimal - mu[arm, z_true[t] + 1]
    push!(cum_regret, (t == 1 ? regret : cum_regret[end] + regret))
    # Update observed data
    push!(observed[arm], r)
    rewards[arm] += r
    counts[arm] += 1
    push!(cum_reward, sum(rewards))
end

# Save final plots
#plot(cum_reward, label="Cumulative Reward", xlabel="Time", ylabel="Reward")
#savefig("cumulative_reward_plot.png")

plot(1:N, cum_reward, label="Cumulative Reward", xlabel="Time", ylabel="Value", lw=2)
plot!(1:N, cum_regret, label="Cumulative Regret", lw=2)
savefig("reward_and_regret.png")

