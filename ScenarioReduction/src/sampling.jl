# Quasi-random Gaussian sampling for SD-screening.
#
# Standalone — depends only on QuasiMonteCarlo, SpecialFunctions, LinearAlgebra,
# Statistics, Random. No JuMP/Tulipa/DuckDB so it can be unit-tested without
# spinning up a solver. `src/utils.jl` includes this file so existing
# `include(.../utils.jl)` callers pick the functions up unchanged.

using QuasiMonteCarlo: QuasiMonteCarlo
using LinearAlgebra: cholesky, Symmetric, Diagonal, diag, I
using Random: Random, Xoshiro
using Statistics: Statistics

# Beasley-Springer-Moro inverse standard-normal CDF.
# Pure base-Julia rational approximation, accurate to ~1e-9. Used here because
# the root project does not have SpecialFunctions as a direct dependency (it's
# only transitively in the Manifest), so `using SpecialFunctions: erfinv` does
# not resolve when this file is included from main.jl / test_stochastic_dominance.jl.
# Reference: Boyle, Broadie & Glasserman, "Monte Carlo Methods for Security
# Pricing" (1997), eqn. (8). Standard QMC textbook implementation.
const _BSM_A = (
    -3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
     1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00,
)
const _BSM_B = (
    -5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
     6.680131188771972e+01, -1.328068155288572e+01,
)
const _BSM_C = (
    -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
    -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00,
)
const _BSM_D = (
     7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
     3.754408661907416e+00,
)
const _BSM_PLOW  = 0.02425
const _BSM_PHIGH = 1.0 - _BSM_PLOW

function _norminvcdf(p::Real)
    # Lower tail (rational approximation in q = √(−2 ln p)).
    if p < _BSM_PLOW
        q = sqrt(-2.0 * log(p))
        return (((((_BSM_C[1]*q + _BSM_C[2])*q + _BSM_C[3])*q + _BSM_C[4])*q + _BSM_C[5])*q + _BSM_C[6]) /
               ((((_BSM_D[1]*q + _BSM_D[2])*q + _BSM_D[3])*q + _BSM_D[4])*q + 1.0)
    end
    # Upper tail (mirror of the lower tail).
    if p > _BSM_PHIGH
        q = sqrt(-2.0 * log(1.0 - p))
        return -(((((_BSM_C[1]*q + _BSM_C[2])*q + _BSM_C[3])*q + _BSM_C[4])*q + _BSM_C[5])*q + _BSM_C[6]) /
                ((((_BSM_D[1]*q + _BSM_D[2])*q + _BSM_D[3])*q + _BSM_D[4])*q + 1.0)
    end
    # Central region (rational approximation in r = (p − 0.5)²).
    q = p - 0.5
    r = q * q
    return (((((_BSM_A[1]*r + _BSM_A[2])*r + _BSM_A[3])*r + _BSM_A[4])*r + _BSM_A[5])*r + _BSM_A[6]) * q /
           (((((_BSM_B[1]*r + _BSM_B[2])*r + _BSM_B[3])*r + _BSM_B[4])*r + _BSM_B[5])*r + 1.0)
end

"""
    sobol_gaussian_samples(N, μ, Σ; scramble=true, seed=1, jitter=1e-8)

Generate `N` quasi-random samples from `N(μ, Σ)` using Sobol' points pushed
through the standard-normal inverse CDF and a Cholesky transform. Returns a
`d × N` matrix; each column is one sample (matches the existing
`eachcol(samples)` consumer in `stochastic_dominance.jl`).

Pipeline:
  1. Sobol' uniforms `U ∈ [0,1]^{d×N}`, optionally Owen-scrambled.
  2. Inverse normal CDF via `√2·erfinv(2u − 1)`; `u` is clamped away from
     `{0, 1}` to avoid `±Inf`.
  3. Cholesky factor `L` of `Symmetric(Σ + jitter·I)` (jitter protects against
     numerical PSD violations on empirical covariances).
  4. `Z = μ .+ L * W`.
"""
function sobol_gaussian_samples(
    N::Int,
    μ::AbstractVector,
    Σ::AbstractMatrix;
    scramble::Bool=true,
    seed::Int=1,
    jitter::Real=1e-8,
)
    d = length(μ)
    size(Σ) == (d, d) || error("Σ must be $d×$d, got $(size(Σ))")

    sampler = scramble ?
        QuasiMonteCarlo.SobolSample(;
            R=QuasiMonteCarlo.OwenScramble(base=2, pad=32, rng=Xoshiro(seed)),
        ) :
        QuasiMonteCarlo.SobolSample()
    U = QuasiMonteCarlo.sample(N, zeros(d), ones(d), sampler)

    eps_safe = 1e-12
    W = _norminvcdf.(clamp.(U, eps_safe, 1 - eps_safe))

    L = cholesky(Symmetric(Matrix(Σ) + jitter * I)).L

    return μ .+ L * W
end

"""
    sobol_gaussian_samples_nonneg(N, μ, Σ; mode=:clip, ub=nothing, ...)

Wrapper around [`sobol_gaussian_samples`](@ref) enforcing non-negativity (and
optional per-coordinate upper bound `ub`).

Modes:
  * `:clip`    — clamp coord-wise to `[0, ub_k]`. Fast; distorts the marginals
                 (mass that would have been negative piles up at 0).
  * `:reflect` — reflect via `z ↦ |z|`. Preserves total probability but
                 distorts the distribution.
  * `:reject`  — over-generate by a factor of `oversample`, drop any sample
                 with a negative or out-of-bounds coordinate, keep the first
                 `N` survivors. Preserves the conditional distribution exactly
                 but loses Sobol' low-discrepancy structure of the survivors.
"""
function sobol_gaussian_samples_nonneg(
    N::Int,
    μ::AbstractVector,
    Σ::AbstractMatrix;
    mode::Symbol=:clip,
    ub::Union{Nothing,AbstractVector}=nothing,
    scramble::Bool=true,
    seed::Int=1,
    jitter::Real=1e-8,
    oversample::Real=3.0,
)
    if mode === :clip
        Z = sobol_gaussian_samples(N, μ, Σ; scramble, seed, jitter)
        Z .= max.(Z, 0.0)
        isnothing(ub) || (Z .= min.(Z, ub))
        return Z
    elseif mode === :reflect
        Z = sobol_gaussian_samples(N, μ, Σ; scramble, seed, jitter)
        Z .= abs.(Z)
        isnothing(ub) || (Z .= min.(Z, ub))
        return Z
    elseif mode === :reject
        Ngen = ceil(Int, oversample * N)
        Z = sobol_gaussian_samples(Ngen, μ, Σ; scramble, seed, jitter)
        keep = trues(Ngen)
        @inbounds for k in 1:Ngen
            col = view(Z, :, k)
            if any(<(0.0), col)
                keep[k] = false
                continue
            end
            if !isnothing(ub) && any(i -> col[i] > ub[i], eachindex(col))
                keep[k] = false
            end
        end
        idx = findall(keep)
        length(idx) < N && error(
            "Only $(length(idx))/$N feasible samples after oversample=$oversample; " *
            "increase oversample or relax bounds.",
        )
        return Z[:, idx[1:N]]
    else
        throw(ArgumentError("mode must be :clip, :reflect, or :reject (got :$mode)"))
    end
end

"""
    shrink_covariance(Σ_emp; α=0.2, target=:diagonal)

Linear shrinkage of an empirical covariance toward a structured target.
Returns `(1 − α) · Σ_emp + α · T`.

  * `target=:diagonal` → `T = Diagonal(diag(Σ_emp))` (zeros off-diagonals).
  * `target=:identity` → `T = mean(diag(Σ_emp)) · I` (also equalises variances).

With ~144 samples in 7 dimensions the off-diagonals of the empirical Σ are
noisy; a moderate `α ∈ [0.1, 0.3]` dampens those without destroying the
marginal-variance information.
"""
function shrink_covariance(Σ_emp::AbstractMatrix; α::Real=0.2, target::Symbol=:diagonal)
    T = if target === :diagonal
        Matrix(Diagonal(diag(Σ_emp)))
    elseif target === :identity
        Statistics.mean(diag(Σ_emp)) * Matrix(I, size(Σ_emp, 1), size(Σ_emp, 1))
    else
        throw(ArgumentError("target must be :diagonal or :identity (got :$target)"))
    end
    return (1 - α) * Σ_emp + α * T
end
