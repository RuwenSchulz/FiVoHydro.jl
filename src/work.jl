# ==============================================================================
# src/work.jl
#
# Work arrays and initialization.
# ==============================================================================

mutable struct Work1D
    yT::Vector{Float64}
    phi::Vector{Float64}
    mu::Vector{Float64}
    alpha::Vector{Float64}
    y::Vector{Float64}
    y_prev::Vector{Float64}
    P::Vector{Float64}
    n::Vector{Float64}
    e::Vector{Float64}
    ok::BitVector

    x0_yT::Vector{Float64}
    x0_phi::Vector{Float64}
    x0_y::Vector{Float64}

    σp::Matrix{Float64}
    ULp::Matrix{Float64}
    URp::Matrix{Float64}

    ULc::Matrix{Float64}
    URc::Matrix{Float64}
    Fh::Matrix{Float64}

    S::Matrix{Float64}
    k::Matrix{Float64}
    U1::Matrix{Float64}
    U2::Matrix{Float64}

    bad::BitVector
    bad_tmp::BitVector

    tmpFL::Vector{Vector{Float64}}
    tmpFR::Vector{Vector{Float64}}

    theta::Vector{Float64}
    vC::Vector{Float64}
    vF::Vector{Float64}
    dvdr::Vector{Float64}

    alphaF::Vector{Float64}
    gradAlpha::Vector{Float64}

    alpha_tmp::Vector{Float64}
    nur_tmp::Vector{Float64}
    n_tmp::Vector{Float64}

    visc_tmp1::Vector{Float64}
    visc_tmp2::Vector{Float64}
    visc_tmp3::Vector{Float64}

    primfail_tls::Vector{Int}
    primfail_i_tls::Vector{Int}
    primfail_tau_tls::Vector{Float64}
    primfail_Dtau_tls::Vector{Float64}
    primfail_Sr_tls::Vector{Float64}
    primfail_E_tls::Vector{Float64}
    primfail_nur_stored_tls::Vector{Float64}
    primfail_Pi_stored_tls::Vector{Float64}
    primfail_piR_stored_tls::Vector{Float64}
    primfail_piEta_stored_tls::Vector{Float64}
    primfail_reason_tls::Vector{PrimRecReason}
    primfail_iters_tls::Vector{Int}
    primfail_resnorm_tls::Vector{Float64}
    bad_tls::Vector{Bool}
    amax_tls::Vector{Float64}
    kmax_tls::Vector{Float64}

    # Previous-substep charm fugacity α (analogous to y_prev) for the temporal part of the
    # covariant Navier–Stokes drive ∇^⟨r⟩α = u_τ²∂rα + u^r u^τ ∂τα in the diffusion relaxation.
    # This is part of Fluidum's covariant gradient projection — required for FiVo's first-order
    # diffusion to match Fluidum's IS2 charm current (≈2× under-driven without the ∂τα term).
    alpha_prev::Vector{Float64}

    # Previous-step temperature, for the ∂_τT of the consistent first moment's sources
    # (`consistent_fm`, 2026-09-11): the pressure-gradient T channel and D ln h need DT.
    # NaN-seeded like alpha_prev; the ∂_τT pieces are dropped on the first step.
    T_prev::Vector{Float64}

    # ∂_τ u^r, built ONCE per relaxation substep from (y, y_prev) and then shared by the
    # charge and viscous blocks, which used to recompute it independently.  Held as its
    # own field because it is BAND-LIMITED before use when `dtau_u_smooth_len > 0` — see
    # the long note in `relax_dissipative!` (src/dissipation.jl) for why a raw pointwise
    # ∂_τ u^r is short-wavelength unstable in an operator-split scheme.
    durdtau::Vector{Float64}
    durdtau_tmp::Vector{Float64}
end

function make_work(U)
    Nvars, Ntot = size(U)
    # Cover the full thread-id range (default + interactive pools), since
    # Threads.@threads may schedule onto the interactive thread (threadid() can
    # exceed Threads.nthreads()).  Under-sizing here BoundsErrors the threaded loops.
    nt = Threads.nthreads() + Threads.nthreads(:interactive)
    return Work1D(
        # yT, phi, mu, alpha, y, y_prev — y_prev NaN-seeded: no ∂_τu^r on the first step
        zeros(Ntot), zeros(Ntot), zeros(Ntot), zeros(Ntot), zeros(Ntot), fill(NaN, Ntot),
        zeros(Ntot), zeros(Ntot), zeros(Ntot), falses(Ntot),

        fill(log(0.25), Ntot), zeros(Ntot), zeros(Ntot),

        zeros(3, Ntot),
        zeros(3, Ntot-1),
        zeros(3, Ntot-1),

        zeros(Nvars, Ntot-1),
        zeros(Nvars, Ntot-1),
        zeros(Nvars, Ntot-1),

        zeros(Nvars, Ntot),
        zeros(Nvars, Ntot),
        similar(U),
        similar(U),

        falses(Ntot),
        falses(Ntot),

        [zeros(Nvars) for _ in 1:nt],
        [zeros(Nvars) for _ in 1:nt],

        zeros(Ntot),
        zeros(Ntot),
        zeros(Ntot),
        zeros(Ntot),

        zeros(Ntot-1),
        zeros(Ntot),

        zeros(Ntot),
        zeros(Ntot),
        zeros(Ntot),

        zeros(Ntot),
        zeros(Ntot),
        zeros(Ntot),

        zeros(Int, nt),
        zeros(Int, nt),
        zeros(nt),
        zeros(nt),
        zeros(nt),
        zeros(nt),
        zeros(nt),
        zeros(nt),
        zeros(nt),
        zeros(nt),
        fill(PRR_UNSET, nt),
        zeros(Int, nt),
        fill(NaN, nt),
        fill(false, nt),
        fill(1e-30, nt),
        fill(0.0, nt),
        fill(NaN, Ntot),   # alpha_prev — NaN marks "no previous substep yet" (∂τα term skipped then)
        fill(NaN, Ntot),   # T_prev — same convention, for the consistent first moment's ∂τT
        zeros(Ntot), zeros(Ntot),   # durdtau, durdtau_tmp — rebuilt every relaxation substep
    )
end

# ------------------------------------------------------------
# Primitive cache builder (for dt-from-work correctness at t=τ0)
# ------------------------------------------------------------
function prime_work_from_U!(work::Work1D, U, grid, τ, model::IdealDiffViscModel)
    L   = layout(model)
    eos = model.eos
    Ntot = size(U,2)

    Threads.@threads for i in 1:Ntot
        tid = Threads.threadid()
        wpr = model.primrec.work[tid]

        D  = safe_div(U[L.iDtau, i], τ)
        Sr = U[L.iSr,i]
        E  = U[L.iE,i]

        nur_phys = (L.hasNur ? phys_from_stored(U[L.iNur,i]) : 0.0)
        Pi_phys  = (L.hasPi  ? phys_from_stored(U[L.iPi,i])  : 0.0)
        piR_phys = (L.hasPiR ? phys_from_stored(U[L.iPiR,i]) : 0.0)
        piEta_phys = (L.hasPiEta ? phys_from_stored(U[L.iPiEta,i]) : 0.0)

        T, μ, ur, n, e, P, ok = cons_to_prim_ideal_phi_diff_visc!(
            wpr, D, Sr, E, nur_phys, Pi_phys, piR_phys, piEta_phys, grid.rC[i], τ, eos;
            yT0=work.x0_yT[i], φ0=work.x0_phi[i], y0=work.x0_y[i],
            maxit=80
        )

        if ok
            Tm = max(T, T_MIN)
            φ  = clamp((μ - hq_mass(eos)) / Tm, -PHI_CAP, PHI_CAP)

            work.yT[i]    = log(Tm)
            work.phi[i]   = φ
            work.mu[i]    = μ
            work.alpha[i] = μ / Tm
            work.y[i]     = asinh(ur)

            work.P[i]     = P
            work.n[i]     = n
            work.e[i]     = e
            work.ok[i]    = true

            uτ = sqrt(1 + ur^2)
            work.vC[i]    = safe_div(ur, uτ)

            work.x0_yT[i]  = work.yT[i]
            work.x0_phi[i] = work.phi[i]
            work.x0_y[i]   = work.y[i]
        else
            # consistent vacuum fallback
            Tv = T_MIN
            μv = 0.0
            Pv, _, ev = eos_Pne(Tv, μv, eos)

            work.yT[i]    = log(Tv)
            work.phi[i]   = 0.0
            work.mu[i]    = μv
            work.alpha[i] = 0.0
            work.y[i]     = 0.0

            work.P[i]     = Pv
            work.n[i]     = 0.0
            work.e[i]     = max(ev, 0.0)
            work.ok[i]    = true

            work.vC[i]    = 0.0

            # Reset Newton initial guess caches to safe values.
            work.x0_yT[i]  = work.yT[i]
            work.x0_phi[i] = 0.0
            work.x0_y[i]   = 0.0
        end
    end

    # axis symmetry cache (safe even for ghosts)
    i0 = grid.nghost + 1
    work.y[i0]  = 0.0
    work.vC[i0] = 0.0
    return nothing
end
