# ==============================================================================
# src/ic_diagnostics.jl
#
# Initial condition (IC) diagnostics.
#
# Goal:
# - Provide a deterministic, solver-independent analysis of IC smoothness and
#   admissibility (floors, causality, etc.).
# - Report first/second derivatives and simple oscillation indicators.
# - Write results to CSV for quick plotting/triage.
#
# Dependencies: Base + CSV/Tables (already used in src/io.jl).
# ==============================================================================

using CSV
using Tables
using Printf
using LinearAlgebra


# ------------------------------------------------------------
# Finite differences (uniform grid)
# ------------------------------------------------------------
function _d1_uniform!(df::Vector{Float64}, f::Vector{Float64}, dr::Float64)
    N = length(f)
    @assert length(df) == N
    inv2dr = 1.0 / (2dr)
    invdr  = 1.0 / dr

    if N == 1
        df[1] = 0.0
        return df
    end

    df[1] = (f[2] - f[1]) * invdr
    @inbounds for i in 2:(N-1)
        df[i] = (f[i+1] - f[i-1]) * inv2dr
    end
    df[N] = (f[N] - f[N-1]) * invdr
    return df
end

function _d2_uniform!(d2f::Vector{Float64}, f::Vector{Float64}, dr::Float64)
    N = length(f)
    @assert length(d2f) == N
    invdr2 = 1.0 / (dr*dr)

    if N == 1
        d2f[1] = 0.0
        return d2f
    elseif N == 2
        # Minimal consistent choice: curvature 0 for both endpoints.
        d2f[1] = 0.0
        d2f[2] = 0.0
        return d2f
    end

    # One-sided second derivative at endpoints.
    d2f[1] = (f[3] - 2f[2] + f[1]) * invdr2
    @inbounds for i in 2:(N-1)
        d2f[i] = (f[i+1] - 2f[i] + f[i-1]) * invdr2
    end
    d2f[N] = (f[N] - 2f[N-1] + f[N-2]) * invdr2
    return d2f
end

# ------------------------------------------------------------
# Smoothed derivatives (uniform grid, Savitzky–Golay)
# ------------------------------------------------------------
function _sgolay_coeffs(halfwidth::Int, order::Int, deriv::Int)
    @assert halfwidth >= 1
    @assert order >= deriv
    x = collect(-halfwidth:halfwidth)
    m = length(x)
    # Vandermonde: A[j,k] = x[j]^(k-1), k=1..order+1
    A = Matrix{Float64}(undef, m, order + 1)
    @inbounds for j in 1:m
        A[j, 1] = 1.0
        for k in 2:(order + 1)
            A[j, k] = A[j, k - 1] * x[j]
        end
    end
    # Pseudoinverse row for the desired derivative at x=0:
    # c = e_{deriv}^T * (A^T A)^{-1} A^T
    ATA = A' * A
    pinvA = ATA \ A'
    c = pinvA[deriv + 1, :]
    # Scale by factorial(deriv) to convert polynomial coefficient to derivative.
    fact = 1.0
    for k in 2:deriv
        fact *= k
    end
    return fact .* c
end

function _sgolay_deriv_uniform!(df::Vector{Float64}, f::Vector{Float64}, dr::Float64;
                                deriv::Int=1, halfwidth::Int=5, order::Int=3)
    N = length(f)
    @assert length(df) == N
    fill!(df, NaN)
    if N < (2*halfwidth + 1)
        # Not enough points; fall back.
        if deriv == 1
            return _d1_uniform!(df, f, dr)
        elseif deriv == 2
            return _d2_uniform!(df, f, dr)
        else
            return df
        end
    end

    c = _sgolay_coeffs(halfwidth, order, deriv)
    invdr = 1.0 / (dr^deriv)
    @inbounds for i in (halfwidth + 1):(N - halfwidth)
        s = 0.0
        for j in 1:length(c)
            s += c[j] * f[i + (j - (halfwidth + 1))]
        end
        df[i] = s * invdr
    end

    # Simple boundary fill: copy nearest interior value.
    @inbounds begin
        for i in 1:halfwidth
            df[i] = df[halfwidth + 1]
            df[N - i + 1] = df[N - halfwidth]
        end
    end

    return df
end

@inline function _total_variation(f::Vector{Float64})
    N = length(f)
    N <= 1 && return 0.0
    tv = 0.0
    @inbounds for i in 1:(N-1)
        tv += abs(f[i+1] - f[i])
    end
    return tv
end

function _sign_changes(df::Vector{Float64}; eps::Float64=0.0)
    # Count sign flips ignoring near-zero values.
    N = length(df)
    N <= 2 && return 0

    prev = 0
    n = 0
    @inbounds for i in 1:N
        x = df[i]
        if !isfinite(x) || abs(x) <= eps
            continue
        end
        s = x > 0 ? 1 : -1
        if prev != 0 && s != prev
            n += 1
        end
        prev = s
    end
    return n
end

function _field_summary(name::AbstractString, f::Vector{Float64}, df::Vector{Float64}, d2f::Vector{Float64};
                        dr::Float64,
                        floor_val::Union{Nothing,Float64}=nothing)
    n = length(f)

    nf = 0
    fmin = Inf
    fmax = -Inf
    nfloor = 0

    @inbounds for i in 1:n
        x = f[i]
        if !isfinite(x)
            nf += 1
            continue
        end
        fmin = min(fmin, x)
        fmax = max(fmax, x)
        if floor_val !== nothing && x <= (floor_val::Float64) * (1 + 1e-14)
            nfloor += 1
        end
    end

    maxabs_df  = 0.0
    maxabs_d2f = 0.0
    @inbounds for i in 1:n
        x1 = df[i]
        x2 = d2f[i]
        if isfinite(x1)
            maxabs_df = max(maxabs_df, abs(x1))
        end
        if isfinite(x2)
            maxabs_d2f = max(maxabs_d2f, abs(x2))
        end
    end

    # Dimensionless curvature indicator: |f''| * dr^2 / (|f| + eps)
    max_kappa = 0.0
    @inbounds for i in 1:n
        x = f[i]
        x2 = d2f[i]
        if isfinite(x) && isfinite(x2)
            denom = abs(x) + 1e-300
            max_kappa = max(max_kappa, abs(x2) * (dr*dr) / denom)
        end
    end

    return (
        name = String(name),
        n = n,
        n_nonfinite = nf,
        min = isfinite(fmin) ? fmin : NaN,
        max = isfinite(fmax) ? fmax : NaN,
        tv = _total_variation(f),
        max_abs_d1 = maxabs_df,
        max_abs_d2 = maxabs_d2f,
        max_dimless_curv = max_kappa,
        d1_sign_changes = _sign_changes(df; eps=0.0),
        n_at_floor = nfloor,
    )
end

# ------------------------------------------------------------
# Primitive extraction at τ0 (for diagnostics)
# ------------------------------------------------------------
function extract_ic_primitives(U, grid, τ, model)
    L = layout(model)
    ng = grid.nghost
    i0 = ng + 1
    iL = size(U,2) - ng
    Np = iL - i0 + 1

    r   = Vector{Float64}(undef, Np)
    T   = Vector{Float64}(undef, Np)
    mu  = Vector{Float64}(undef, Np)
    alpha = Vector{Float64}(undef, Np)
    phi   = Vector{Float64}(undef, Np)
    ur  = Vector{Float64}(undef, Np)
    v   = Vector{Float64}(undef, Np)
    n   = Vector{Float64}(undef, Np)
    e   = Vector{Float64}(undef, Np)
    P   = Vector{Float64}(undef, Np)
    D   = Vector{Float64}(undef, Np)
    ok  = Vector{Bool}(undef, Np)

    Dtau = Vector{Float64}(undef, Np)
    Sr   = Vector{Float64}(undef, Np)
    EE   = Vector{Float64}(undef, Np)
    nur_stored = Vector{Float64}(undef, Np)

    mhq = hq_mass(model.eos)

    Threads.@threads for i in i0:iL
        j = i - ng
        prim = cons_to_prim_col(U, i, grid.rC[i], τ, model)
        uτ = sqrt(1 + prim.ur^2)
        vv = safe_div(prim.ur, uτ)

        @inbounds begin
            r[j]     = grid.rC[i]
            T[j]     = prim.T
            mu[j]    = prim.mu
            alpha[j] = safe_div(prim.mu, prim.T)
            phi[j]   = safe_div((prim.mu - mhq), prim.T)
            ur[j]    = prim.ur
            v[j]     = vv
            n[j]     = prim.n
            e[j]     = prim.e
            P[j]     = prim.P
            D[j]     = safe_div(U[L.iDtau, i], τ)
            ok[j]    = prim.ok

            Dtau[j] = U[L.iDtau, i]
            Sr[j]   = U[L.iSr,   i]
            EE[j]   = U[L.iE,    i]
            nur_stored[j] = (L.hasNur ? U[L.iNur, i] : 0.0)
        end
    end

    return (; r, T, mu, alpha, phi, ur, v, n, e, P, D, ok, Dtau, Sr, E=EE, nur_stored)
end

# ------------------------------------------------------------
# dt/courant-related IC stats
# ------------------------------------------------------------
function ic_dt_stats(work::Work1D, grid, τ, model::IdealDiffViscModel; CFL::Float64=0.2, CFLτ::Float64=0.05)
    ng = grid.nghost
    amax_hyp = 1e-30
    amax_used = 1e-30
    kmax = 0.0

    τπ_min = Inf
    τΠ_min = Inf

    @inbounds for i in (ng+1):(length(work.yT)-ng)
        work.ok[i] || continue

        T  = exp(work.yT[i])
        μ  = work.mu[i]
        ur = sinh(work.y[i])

        λm, λp = wavespeeds_from_prim(T, μ, ur, model.eos)
        a = max(abs(λm), abs(λp))

        amax_hyp = max(amax_hyp, a)

        if model.enable_diff
            uτ = sqrt(1 + ur^2)
            if model.diffusion_drive === :alpha
                κ  = diff_kappa(T, μ, work.n[i], model)
                if model.charge_mode === :density_frame
                    # Match the production cap in main.jl `compute_dt_from_work`:
                    # the density frame's explicit parabolic flux is limited by
                    # D_eff = κ(u^τ)²/(∂n/∂α), not by the bare κ(u^τ)².
                    dndα = diff_dn_dalpha(T, work.alpha[i], model)
                    dndα > 0.0 || (dndα = max(work.n[i], TINY))   # fail safe, see main.jl
                    kmax = max(kmax, κ * (uτ^2) / dndα)
                else
                    kmax = max(kmax, κ * (uτ^2))
                end
            elseif model.diffusion_drive === :n
                DsT = model.kappa_coeff
                D = safe_div(DsT, T) / fmGeV
                kmax = max(kmax, D)
            else
                throw(ArgumentError("Unknown diffusion_drive=$(model.diffusion_drive). Use :alpha or :n"))
            end
            a = max(a, 0.999999)
        end
        amax_used = max(amax_used, a)

        if model.enable_shear
            τπ = visc_tauShear(T, μ, work.n[i], work.e[i], work.P[i], model)
            if isfinite(τπ) && τπ > 0
                τπ_min = min(τπ_min, τπ)
            end
        end
        if model.enable_bulk
            τΠ = visc_tauPi(T, μ, work.n[i], work.e[i], work.P[i], model)
            if isfinite(τΠ) && τΠ > 0
                τΠ_min = min(τΠ_min, τΠ)
            end
        end
    end

    dt_cfl  = min(CFL * grid.dr/(amax_used + TINY), CFLτ * τ)
    dt_diff = (model.enable_diff && kmax > 0) ? (model.diff_dt_coeff * grid.dr^2/(kmax + TINY)) : Inf
    dt_shear = (model.enable_shear && isfinite(τπ_min) && model.shear_dt_coeff > 0) ? (model.shear_dt_coeff * τπ_min) : Inf
    dt_bulk  = (model.enable_bulk  && isfinite(τΠ_min) && model.bulk_dt_coeff > 0) ? (model.bulk_dt_coeff * τΠ_min) : Inf

    dt = min(dt_cfl, dt_diff, dt_shear, dt_bulk)

    return (
        dt = dt,
        dt_cfl = dt_cfl,
        dt_diff = isfinite(dt_diff) ? dt_diff : NaN,
        dt_shear = isfinite(dt_shear) ? dt_shear : NaN,
        dt_bulk = isfinite(dt_bulk) ? dt_bulk : NaN,
        amax = amax_used,
        amax_hyp = amax_hyp,
        amax_used = amax_used,
        kmax = kmax,
        taupi_min = isfinite(τπ_min) ? τπ_min : NaN,
        tauPi_min = isfinite(τΠ_min) ? τΠ_min : NaN,
    )
end

# ------------------------------------------------------------
# Main entry: analyze & write CSV
# ------------------------------------------------------------
function write_ic_diagnostics(outdir::AbstractString, U, grid, τ, model;
                              label::AbstractString = "",
                              CFL::Float64=0.2,
                              CFLτ::Float64=0.05)
    mkpath(outdir)

    # Use the same primitive recovery helper as snapshot output for consistency.
    prim = extract_ic_primitives(U, grid, τ, model)

    # Derivatives
    dr = grid.dr

    dT  = similar(prim.T)
    d2T = similar(prim.T)

    dα  = similar(prim.alpha)
    d2α = similar(prim.alpha)

    dφ  = similar(prim.phi)
    d2φ = similar(prim.phi)

    dμ  = similar(prim.mu)
    d2μ = similar(prim.mu)

    dμ_sg  = similar(prim.mu)
    d2μ_sg = similar(prim.mu)

    dn  = similar(prim.n)
    d2n = similar(prim.n)

    de  = similar(prim.e)
    d2e = similar(prim.e)

    de_sg  = similar(prim.e)
    d2e_sg = similar(prim.e)

    dv  = similar(prim.v)
    d2v = similar(prim.v)

    dur  = similar(prim.ur)
    d2ur = similar(prim.ur)

    dP  = similar(prim.P)
    d2P = similar(prim.P)

    dD  = similar(prim.D)
    d2D = similar(prim.D)

    dDtau  = similar(prim.Dtau)
    d2Dtau = similar(prim.Dtau)

    dSr  = similar(prim.Sr)
    d2Sr = similar(prim.Sr)

    dE  = similar(prim.E)
    d2E = similar(prim.E)

    _d1_uniform!(dT,  prim.T,     dr)
    _d2_uniform!(d2T, prim.T,     dr)

    _d1_uniform!(dα,  prim.alpha, dr)
    _d2_uniform!(d2α, prim.alpha, dr)

    _d1_uniform!(dφ,  prim.phi,   dr)
    _d2_uniform!(d2φ, prim.phi,   dr)

    _d1_uniform!(dμ,  prim.mu,    dr)
    _d2_uniform!(d2μ, prim.mu,    dr)

    # Smoothed derivatives (helps visually diagnose true structure vs. FD noise)
    _sgolay_deriv_uniform!(dμ_sg,  prim.mu, dr; deriv=1, halfwidth=6, order=3)
    _sgolay_deriv_uniform!(d2μ_sg, prim.mu, dr; deriv=2, halfwidth=6, order=3)

    _sgolay_deriv_uniform!(de_sg,  prim.e, dr; deriv=1, halfwidth=6, order=3)
    _sgolay_deriv_uniform!(d2e_sg, prim.e, dr; deriv=2, halfwidth=6, order=3)

    _d1_uniform!(dn,  prim.n,     dr)
    _d2_uniform!(d2n, prim.n,     dr)

    _d1_uniform!(de,  prim.e,     dr)
    _d2_uniform!(d2e, prim.e,     dr)

    _d1_uniform!(dv,  prim.v,     dr)
    _d2_uniform!(d2v, prim.v,     dr)

    _d1_uniform!(dur,  prim.ur,    dr)
    _d2_uniform!(d2ur, prim.ur,    dr)

    _d1_uniform!(dP,  prim.P,     dr)
    _d2_uniform!(d2P, prim.P,     dr)

    _d1_uniform!(dD,  prim.D,     dr)
    _d2_uniform!(d2D, prim.D,     dr)

    _d1_uniform!(dDtau,  prim.Dtau,  dr)
    _d2_uniform!(d2Dtau, prim.Dtau,  dr)

    _d1_uniform!(dSr,  prim.Sr,  dr)
    _d2_uniform!(d2Sr, prim.Sr,  dr)

    _d1_uniform!(dE,  prim.E,  dr)
    _d2_uniform!(d2E, prim.E,  dr)

    # Work cache (consistent with solver path). We also use this to compute the
    # FV-limited solver-style gradient ∂r α used by the diffusion sector.
    work = make_work(U)
    prime_work_from_U!(work, U, grid, τ, model)

    alpha_smooth = similar(prim.alpha)
    gradAlpha_fv = similar(prim.alpha)
    fill!(alpha_smooth, NaN)
    fill!(gradAlpha_fv, NaN)
    try
        epsα = clamp(model.alpha_filter_eps + eps_from_len(model.alpha_smooth_len, grid.dr), 0.0, 0.24)
        smooth_alpha!(work, grid, epsα)

        ng = grid.nghost
        @inbounds for g in 1:ng
            il = ng + 1 - g
            ir = ng + g
            work.alpha[il] = work.alpha[ir]
        end

        # Outer boundary: use a zero-gradient (copy) fill for ghost cells so the
        # FV gradient does not see an artificial jump to 0 at r=rmax.
        @inbounds begin
            iL = length(work.alpha) - ng
            for g in 1:ng
                work.alpha[iL + g] = work.alpha[iL]
            end
        end

        compute_alpha_grad_fv!(work, grid; limiter=mc_limiter)

        i0 = ng + 1
        iL = size(U,2) - ng
        Threads.@threads for i in i0:iL
            j = i - ng
            @inbounds begin
                alpha_smooth[j] = work.alpha[i]
                gradAlpha_fv[j] = work.gradAlpha[i]
            end
        end
    catch err
        @warn "Failed to compute solver-style gradAlpha; writing NaNs" label=label exception=(err, catch_backtrace())
    end

    # Per-cell table
    tbl = (
        r = prim.r,
        ok = prim.ok,
        T = prim.T,
        mu = prim.mu,
        dmu_sg = dμ_sg,
        d2mu_sg = d2μ_sg,
        alpha = prim.alpha,
        alpha_smooth = alpha_smooth,
        gradAlpha_fv = gradAlpha_fv,
        phi = prim.phi,
        ur = prim.ur,
        v = prim.v,
        n = prim.n,
        e = prim.e,
        de_sg = de_sg,
        d2e_sg = d2e_sg,
        P = prim.P,
        D = prim.D,
        Dtau = prim.Dtau,
        Sr = prim.Sr,
        E = prim.E,
        nur_stored = prim.nur_stored,
        dT = dT,
        d2T = d2T,
        dmu = dμ,
        d2mu = d2μ,
        dalpha = dα,
        d2alpha = d2α,
        dphi = dφ,
        d2phi = d2φ,
        dur = dur,
        d2ur = d2ur,
        dn = dn,
        d2n = d2n,
        de = de,
        d2e = d2e,
        dP = dP,
        d2P = d2P,
        dD = dD,
        d2D = d2D,
        dDtau = dDtau,
        d2Dtau = d2Dtau,
        dSr = dSr,
        d2Sr = d2Sr,
        dE = dE,
        d2E = d2E,
        dv = dv,
        d2v = d2v,
    )

    fields_csv = joinpath(outdir, "ic_fields.csv")
    CSV.write(fields_csv, tbl)

    # Summary rows
    summaries = Any[
        _field_summary("T", prim.T, dT, d2T; dr=dr, floor_val=T_MIN),
        _field_summary("mu", prim.mu, dμ, d2μ; dr=dr),
        _field_summary("alpha", prim.alpha, dα, d2α; dr=dr),
        _field_summary("phi", prim.phi, dφ, d2φ; dr=dr),
        _field_summary("ur", prim.ur, dur, d2ur; dr=dr),
        _field_summary("P", prim.P, dP, d2P; dr=dr),
        _field_summary("n", prim.n, dn, d2n; dr=dr),
        _field_summary("e", prim.e, de, d2e; dr=dr, floor_val=E_FLOOR),
        _field_summary("D", prim.D, dD, d2D; dr=dr),
        _field_summary("Dtau", prim.Dtau, dDtau, d2Dtau; dr=dr),
        _field_summary("Sr", prim.Sr, dSr, d2Sr; dr=dr),
        _field_summary("E", prim.E, dE, d2E; dr=dr),
        _field_summary("v", prim.v, dv, d2v; dr=dr),
    ]

    ok_count = count(prim.ok)
    bad_count = length(prim.ok) - ok_count

    last_good = 0
    @inbounds for j in 1:length(prim.ok)
        if prim.ok[j]
            last_good = j
        else
            break
        end
    end
    rmax_good = if last_good >= 1
        # Use outer face of last good cell as max usable r.
        grid.rF[grid.nghost + last_good + 1]
    else
        NaN
    end

    vabs_max = 0.0
    vabs_near = 0
    @inbounds for i in eachindex(prim.v)
        x = abs(prim.v[i])
        if isfinite(x)
            vabs_max = max(vabs_max, x)
            if x >= 0.999
                vabs_near += 1
            end
        end
    end

    # dt stats use the work-cache (consistent with compute_dt_from_work)
    dtstats = ic_dt_stats(work, grid, τ, model; CFL=CFL, CFLτ=CFLτ)

    header = (
        label = String(label),
        tau = Float64(τ),
        Nr = Int(grid.Nr),
        rmax = Float64(grid.rmax),
        dr = Float64(grid.dr),
        ok_cells = ok_count,
        bad_cells = bad_count,
        ok_prefix_cells = last_good,
        Nr_good = last_good,
        rmax_good = rmax_good,
        vabs_max = vabs_max,
        vabs_near0999 = vabs_near,
        dt = dtstats.dt,
        dt_cfl = dtstats.dt_cfl,
        dt_diff = dtstats.dt_diff,
        dt_shear = dtstats.dt_shear,
        dt_bulk = dtstats.dt_bulk,
        amax = dtstats.amax,
        amax_hyp = dtstats.amax_hyp,
        amax_used = dtstats.amax_used,
        kmax = dtstats.kmax,
        taupi_min = dtstats.taupi_min,
        tauPi_min = dtstats.tauPi_min,
        enable_diff = model.enable_diff,
        enable_shear = model.enable_shear,
        enable_bulk = model.enable_bulk,
        CFL = CFL,
        CFLtau = CFLτ,
    )

    # Write summary as a single CSV with header rows expanded
    # - ic_run_summary.csv: one row with global IC stats
    # - ic_field_summary.csv: one row per field
    CSV.write(joinpath(outdir, "ic_run_summary.csv"), [header])
    CSV.write(joinpath(outdir, "ic_field_summary.csv"), summaries)

    return (
        outdir = String(outdir),
        fields_csv = String(fields_csv),
        ok_cells = ok_count,
        bad_cells = bad_count,
        dt = dtstats.dt,
    )
end
