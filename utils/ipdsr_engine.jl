
# =====================================================================
# IPDSR Core Mathematical Engine
# =====================================================================

function aggregate_objectives(F::Vector{Float64}, gamma::Vector{Float64}, N_prime::Int)
    N = length(F)
    # If we have fewer scenarios than bins, no aggregation needed
    if N <= N_prime
        return F, gamma, [[i] for i in 1:N]
    end
    
    sorted_idx = sortperm(F)
    F_sorted = F[sorted_idx]
    gamma_sorted = gamma[sorted_idx]

    F_agg = zeros(Float64, N_prime)
    gamma_agg = zeros(Float64, N_prime)
    mapping = Vector{Vector{Int}}(undef, N_prime)

    chunk_size = div(N, N_prime)
    remainder = N % N_prime

    start_idx = 1
    for j in 1:N_prime
        end_idx = start_idx + chunk_size - 1 + (j <= remainder ? 1 : 0)

        # Track which original sorted indices belong to this bin
        mapping[j] = sorted_idx[start_idx:end_idx]

        gamma_agg[j] = sum(gamma_sorted[start_idx:end_idx])

        if gamma_agg[j] > 0
            F_agg[j] = sum(F_sorted[start_idx:end_idx] .* gamma_sorted[start_idx:end_idx]) / gamma_agg[j]
        else
            F_agg[j] = F_sorted[start_idx]
        end

        start_idx = end_idx + 1
    end

    return F_agg, gamma_agg, mapping
end

function solve_ipdsr_mip(F::Vector{Float64}, gamma::Vector{Float64}, K::Int, lambda::Float64, alpha::Float64, seed::Int = 42)
    N = length(F)

    # 1. Calculate original VaR (v_xi_alpha)
    cum_prob = 0.0
    v_xi_alpha = F[end]
    for i in 1:N
        cum_prob += gamma[i]
        if cum_prob >= alpha
            v_xi_alpha = F[i]
            break
        end
    end

    # 2. Pre-calculate the C_ij matrix
    C = zeros(Float64, N, N)
    for i in 1:N
        for j in 1:N
            pos_i = max(0.0, F[i] - v_xi_alpha)
            pos_j = max(0.0, F[j] - v_xi_alpha)
            C[i,j] = F[i] - F[j] + (lambda / (1 - alpha)) * (pos_i - pos_j)
        end
    end

    # 3. Build JuMP Model
    model = JuMP.Model(Gurobi.Optimizer)
    JuMP.set_silent(model)
    
    JuMP.set_optimizer_attribute(model, "Threads", IPDSR_THREADS) 
    JuMP.set_optimizer_attribute(model, "TimeLimit", IPDSR_TIME_LIMIT) 
    JuMP.set_optimizer_attribute(model, "MIPGap", IPDSR_MIP_GAP) 
    JuMP.set_optimizer_attribute(model, "Seed", seed)

    @variable(model, u[1:N], Bin)
    @variable(model, v[1:N, 1:N], Bin)
    @variable(model, n[1:N], Bin)
    @variable(model, h[1:N], Bin)
    @variable(model, W[1:N, 1:N], Bin)
    @variable(model, Z >= 0)           

    @constraint(model, [i=1:N, j=1:N], v[i,j] <= u[j])
    @constraint(model, [j=1:N], v[j,j] == u[j])
    @constraint(model, [i=1:N], sum(v[i,:]) == 1)
    @constraint(model, sum(u) == K)
    @constraint(model, sum(n) == 1)
    @constraint(model, [j=1:N], n[j] <= u[j])

    @constraint(model, [j=1:N], h[j] <= u[j])
    @constraint(model, [j=1:N], h[j] <= 1 - sum(n[p] for p in 1:j))
    @constraint(model, [j=1:N], h[j] >= u[j] - sum(n[p] for p in 1:j))

    @constraint(model, [i=1:N, j=1:N], W[i,j] <= h[j])
    @constraint(model, [i=1:N, j=1:N], W[i,j] <= v[i,j])
    @constraint(model, [i=1:N, j=1:N], W[i,j] >= h[j] + v[i,j] - 1)

    @constraint(model, sum(W[i,j] * gamma[i] for i in 1:N for j in 1:N) <= alpha)
    @constraint(model, [k=1:N], 
        sum(W[i,j] * gamma[i] for i in 1:N for j in 1:N) + 
        sum((v[i,k] - W[i,k]) * gamma[i] for i in 1:N) >= alpha * (u[k] - h[k])
    )

    X = sum(v[i,j] * gamma[i] * C[i,j] for i in 1:N for j in 1:N) + 
        lambda * v_xi_alpha - lambda * sum(n[j] * F[j] for j in 1:N)

    @constraint(model, Z >= X)
    @constraint(model, Z >= -X)
    @objective(model, Min, Z)

    JuMP.optimize!(model)

    status = JuMP.termination_status(model)
    if status != JuMP.OPTIMAL && status != JuMP.TIME_LIMIT
        @warn "MIP did not find an optimal solution. Status: $status"
        empty!(model) # Free memory
        GC.gc()       # Force Julia garbage collection
        return Int[], Float64[]
    end

    selected_indices = findall(x -> x > 0.5, JuMP.value.(u))
    new_weights = [sum(JuMP.value.(v)[i, j] * gamma[i] for i in 1:N) for j in selected_indices]

    # Wipe the C-pointers and trigger Julia's GC immediately
    empty!(model)
    GC.gc()

    return selected_indices, new_weights
end

function verify_ipdsr_mip()
    println("--- Starting IPDSR Verification ---")
    dummy_F = [100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 800.0, 1200.0, 2000.0]
    dummy_gamma = fill(0.1, 10)
    K = 3
    lambda = 0.5
    alpha = 0.8 

    println("Pre-aggregation check... Shrinking N=10 to N'=5")
    F_agg, gamma_agg, mapping = aggregate_objectives(dummy_F, dummy_gamma, 5) # Updated signature
    
    println("\nSolving MIP...")
    selected_idx, new_weights = solve_ipdsr_mip(F_agg, gamma_agg, K, lambda, alpha)

    println("\n--- Verification Results ---")
    println("Selected Aggregated Indices: ", selected_idx)
    println("New Weights: ", new_weights)
    println("-----------------------------------")
end