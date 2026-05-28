# ==============================================================================
# src/debugging.jl
#
# Debugging + failure dump helpers.
# This file is included from main.jl only after core types (e.g. IdealDiffViscModel,
# Work1D, DiagCounters) are defined.
# ==============================================================================

# ------------------------------------------------------------
# Debug: shear -> conserved consistency check
#   Assumes piR_phys is LRF diagonal shear scalar (π̂_rr).
#   Then: π^{τr} = (uτ*ur) π̂_rr,  π^{ττ} = (ur^2) π̂_rr
# ------------------------------------------------------------
const _TCONS_CHECK_COUNTER = Ref(0)

@inline function _debug_check_shear_conserved!(U::AbstractMatrix, grid, τ::Float64,
                                              model::IdealDiffViscModel, work::Work1D;
                                              diag::Union{Nothing,DiagCounters}=nothing)
    flags = hydro_flags()
    flags.check_tcons || return nothing
    model.enable_shear || return nothing

    _TCONS_CHECK_COUNTER[] += 1
    (_TCONS_CHECK_COUNTER[] % flags.check_tcons_every == 0) || return nothing

    L  = layout(model)
    ng = grid.nghost
    i0 = ng + 1
    iL = size(U,2) - ng

    i_env = flags.check_tcons_i
    i = (i_env < i0 || i_env > iL) ? ((i0 + iL) >>> 1) : i_env

    (i0 <= i <= iL) || return nothing
    work.ok[i] || return nothing

    # reconstructed prims
    ur = sinh(work.y[i])
    uτ = sqrt(1 + ur^2)
    e  = work.e[i]
    P  = work.P[i]

    # physical dissipatives from U
    Π   = (L.hasPi    ? phys_from_stored(U[L.iPi,i])    : 0.0)
    πr  = (L.hasPiR   ? phys_from_stored(U[L.iPiR,i])   : 0.0)   # assumed π̂_rr
    # piEta does NOT mix into ττ/τr for a pure boost in r, so not used here.

    Sr_expected = (e + P + Π) * (uτ*ur) + (uτ*ur) * πr
    E_expected  = (e + P + Π) * (uτ^2)  - (P + Π) + (ur^2) * πr

    Sr_stored = U[L.iSr,i]
    E_stored  = U[L.iE,i]

    tol_rel = flags.check_tcons_tol_rel
    tol_abs = flags.check_tcons_tol_abs

    errSr = Sr_stored - Sr_expected
    errE  = E_stored  - E_expected

    denomSr = max(max(abs(Sr_expected), abs(Sr_stored)), 1.0)
    denomE  = max(max(abs(E_expected),  abs(E_stored)),  1.0)

    badSr = (abs(errSr) > tol_abs) && (abs(errSr)/denomSr > tol_rel)
    badE  = (abs(errE)  > tol_abs) && (abs(errE) /denomE  > tol_rel)

    if badSr || badE
        r = grid.rC[i]
        v = safe_div(ur, uτ)
        kv = (; τ=τ, i=i, r=r, v=v,
            Sr=Sr_stored, Sr_expected=Sr_expected, errSr=errSr, relSr=(abs(errSr)/denomSr),
            E=E_stored,   E_expected=E_expected,   errE=errE,   relE=(abs(errE)/denomE),
            Pi=Π, piR=πr, e=e, P=P, uτ=uτ, ur=ur)

        @warn "Tcons/shear mismatch (check your prim<->cons shear wiring)" kv...

        if diag !== nothing
            diag.last_misc_i = i
        end
    end

    return nothing
end


# ------------------------------------------------------------
# Failure debugging (window dump)
# ------------------------------------------------------------
@inline function _bad_reason(U::AbstractMatrix, grid, i::Int, τ::Float64, model::IdealDiffViscModel, work;
                            Emin::Float64=E_FLOOR,
                            χ::Float64=χ_SrE,
                            v_eps::Float64=1e-14,
                            Rmax::Float64=10.0)
    L = model.layout
    Dtau = U[L.iDtau,i]
    Sr   = U[L.iSr,i]
    E    = U[L.iE,i]

    if !(isfinite(Dtau) & isfinite(Sr) & isfinite(E))
        return :nonfinite_conserved
    end
    if Dtau < 0.0
        return :negative_Dtau
    end
    if E < Emin
        return :E_below_floor
    end

    if L.hasNur   && !isfinite(U[L.iNur,i]);   return :nonfinite_nur; end
    if L.hasPi    && !isfinite(U[L.iPi,i]);    return :nonfinite_Pi; end
    if L.hasPiR   && !isfinite(U[L.iPiR,i]);   return :nonfinite_piR; end
    if L.hasPiEta && !isfinite(U[L.iPiEta,i]); return :nonfinite_piEta; end

    if work !== nothing && work.ok[i]
        v = work.vC[i]
        if !(isfinite(v) && abs(v) < (1 - v_eps))
            return :superluminal_v
        end

        cap = max(work.P[i] + work.e[i], 0.0)
        if cap > 0
            if model.enable_bulk && L.hasPi
                Π = abs(phys_from_stored(U[L.iPi,i]))
                if Π > Rmax * cap
                    return :Pi_too_large
                end
            end
            if model.enable_shear
                if L.hasPiR
                    pr = abs(phys_from_stored(U[L.iPiR,i]))
                    if pr > Rmax * cap
                        return :piR_too_large
                    end
                end
                if L.hasPiEta
                    pe = abs(phys_from_stored(U[L.iPiEta,i]))
                    if pe > Rmax * cap
                        return :piEta_too_large
                    end
                end
            end
        end

        if model.enable_diff && L.hasNur
            y = work.y[i]
            if isfinite(y)
                uτ = cosh(y)
                ncap = max(work.n[i] * uτ, 0.0)
                if ncap > 0
                    ν = abs(phys_from_stored(U[L.iNur,i]))
                    if ν > Rmax * ncap
                        return :nur_too_large
                    end
                end
            end
        end
    end

    # Cheap Sr admissibility gate before running full primitive recovery.
    Pguess = 0.0
    if work !== nothing && work.ok[i]
        Ptmp = work.P[i]
        if isfinite(Ptmp)
            Pguess = max(Ptmp, 0.0)
        end
    end
    Π  = (L.hasPi    ? abs(phys_from_stored(U[L.iPi,i]))    : 0.0)
    pr = (L.hasPiR   ? abs(phys_from_stored(U[L.iPiR,i]))   : 0.0)
    pe = (L.hasPiEta ? abs(phys_from_stored(U[L.iPiEta,i])) : 0.0)
    χeff = sr_margin() * χ
    Srmax = χeff*(E + Pguess + Π + pr + pe)
    if abs(Sr) <= Srmax
        return :unknown
    end
    if E < 100Emin
        return :unknown
    end

    prim = cons_to_prim_col(U, i, grid.rC[i], τ, model)
    return prim.ok ? :unknown : :prim_recovery_failed
end

function _find_first_bad(U, grid, τ, model::IdealDiffViscModel, work; Emin::Float64=E_FLOOR, χ::Float64=χ_SrE)
    ng = grid.nghost
    for i in (ng+1):(size(U,2)-ng)
        why = _bad_reason(U, grid, i, τ, model, work; Emin=Emin, χ=χ)
        if why != :unknown
            return i, why
        end
    end
    return 0, :none
end

function _dump_failure_window_csv(path::String, U, grid, τ, model::IdealDiffViscModel, work, i::Int; radius::Int=5)
    L = model.layout
    ng = grid.nghost
    N  = size(U,2)
    ilo = max(i - radius, ng+1)
    ihi = min(i + radius, N-ng)

    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "τ,i,rC,Dtau,Sr,E,nur,Pi,piR,piEta,yT,phi,mu,y,vC,n,e,P,ok")
        for j in ilo:ihi
            rC = grid.rC[j]
            Dtau = U[L.iDtau,j]
            Sr   = U[L.iSr,j]
            E    = U[L.iE,j]
            nur  = (L.hasNur   ? phys_from_stored(U[L.iNur,j])   : 0.0)
            Pi   = (L.hasPi    ? phys_from_stored(U[L.iPi,j])    : 0.0)
            piR  = (L.hasPiR   ? phys_from_stored(U[L.iPiR,j])   : 0.0)
            piEta= (L.hasPiEta ? phys_from_stored(U[L.iPiEta,j]) : 0.0)

            haswork = (work !== nothing)
            yT  = (haswork ? work.yT[j]  : NaN)
            phi = (haswork ? work.phi[j] : NaN)
            mu  = (haswork ? work.mu[j]  : NaN)
            y   = (haswork ? work.y[j]   : NaN)
            vC  = (haswork ? work.vC[j]  : NaN)
            n   = (haswork ? work.n[j]   : NaN)
            e   = (haswork ? work.e[j]   : NaN)
            P   = (haswork ? work.P[j]   : NaN)
            ok  = (haswork ? work.ok[j]  : true)

            # Low-overhead CSV row writing (debugging may run in tight loops).
            print(io, τ);   print(io, ',')
            print(io, j);   print(io, ',')
            print(io, rC);  print(io, ',')
            print(io, Dtau);print(io, ',')
            print(io, Sr);  print(io, ',')
            print(io, E);   print(io, ',')
            print(io, nur); print(io, ',')
            print(io, Pi);  print(io, ',')
            print(io, piR); print(io, ',')
            print(io, piEta);print(io, ',')
            print(io, yT);  print(io, ',')
            print(io, phi); print(io, ',')
            print(io, mu);  print(io, ',')
            print(io, y);   print(io, ',')
            print(io, vC);  print(io, ',')
            print(io, n);   print(io, ',')
            print(io, e);   print(io, ',')
            print(io, P);   print(io, ',')
            println(io, ok)
        end
    end

    @debug "wrote failure window CSV" file=path τ=τ i=i ilo=ilo ihi=ihi
    return nothing
end

function _dump_flux_decomp_csv(path::String, U, grid, τ, model::IdealDiffViscModel, work::Work1D, i::Int;
                               Emin::Float64=E_FLOOR)
    L   = model.layout
    eos = model.eos
    ng  = grid.nghost
    Ntot = size(U,2)

    if !(ng+1 <= i <= Ntot-ng)
        return nothing
    end

    prime_work_from_U!(work, U, grid, τ, model)
    reconstruct_muscl_prims!(work.ULp, work.URp, work.σp, work.yT, work.phi, work.y, grid)

    mkpath(dirname(path))
    open(path, "w") do io
        names = L.names
        hdr = String[]
        append!(hdr, ["UL_"*String(s) for s in names])
        append!(hdr, ["UR_"*String(s) for s in names])
        append!(hdr, ["FL_"*String(s) for s in names])
        append!(hdr, ["FR_"*String(s) for s in names])
        append!(hdr, ["Fh_"*String(s) for s in names])
        append!(hdr, ["S_"*String(s) for s in names])
        append!(hdr, ["div_"*String(s) for s in names])
        append!(hdr, ["dU_"*String(s) for s in names])
        println(io, "kind,τ,i,col,rC,rF,method,okL,okR,λmL,λpL,λmR,λpR,sL,sR,", join(hdr, ","))

        function compute_face(col::Int)
            tmpFL = zeros(length(names))
            tmpFR = zeros(length(names))
            ULtmp = zeros(length(names))
            URtmp = zeros(length(names))
            Fhtmp = zeros(length(names))

            if !(1 <= col <= Ntot-1)
                return (valid=false, col=col, rF=NaN, method="", okL=false, okR=false,
                        λmL=NaN, λpL=NaN, λmR=NaN, λpR=NaN, sL=NaN, sR=NaN,
                        UL=ULtmp, UR=URtmp, FL=tmpFL, FR=tmpFR, Fh=Fhtmp)
            end

            # reconstructed states
            yTL = work.ULp[1,col]; φL = work.ULp[2,col]; yL = work.ULp[3,col]
            yTR = work.URp[1,col]; φR = work.URp[2,col]; yR = work.URp[3,col]

            nurL_phys   = (L.hasNur   ? phys_from_stored(U[L.iNur, col])   : 0.0)
            nurR_phys   = (L.hasNur   ? phys_from_stored(U[L.iNur, col+1]) : 0.0)
            PiL_phys    = (L.hasPi    ? phys_from_stored(U[L.iPi, col])    : 0.0)
            PiR_phys    = (L.hasPi    ? phys_from_stored(U[L.iPi, col+1])  : 0.0)
            piRL_phys   = (L.hasPiR   ? phys_from_stored(U[L.iPiR, col])   : 0.0)
            piRR_phys   = (L.hasPiR   ? phys_from_stored(U[L.iPiR, col+1]) : 0.0)
            piEtaL_phys = (L.hasPiEta ? phys_from_stored(U[L.iPiEta, col])   : 0.0)
            piEtaR_phys = (L.hasPiEta ? phys_from_stored(U[L.iPiEta, col+1]) : 0.0)

            okL, primL = prim_to_cons_col_ideal_phi_diff_visc!(work.ULc, col,
                                                              yTL, φL, yL,
                                                              nurL_phys, PiL_phys, piRL_phys, piEtaL_phys,
                                                              grid.rC[col],
                                                              τ, eos, L)
            okR, primR = prim_to_cons_col_ideal_phi_diff_visc!(work.URc, col,
                                                              yTR, φR, yR,
                                                              nurR_phys, PiR_phys, piRR_phys, piEtaR_phys,
                                                              grid.rC[col+1],
                                                              τ, eos, L)

            use_pc = okL && okR &&
                     work.ULc[L.iDtau,col] ≥ 0.0 && work.URc[L.iDtau,col] ≥ 0.0 &&
                     work.ULc[L.iE,col]    ≥ Emin && work.URc[L.iE,col]    ≥ Emin

            method = use_pc ? "pc" : "fo"
            if use_pc
                @inbounds for a in 1:length(names)
                    ULtmp[a] = work.ULc[a,col]
                    URtmp[a] = work.URc[a,col]
                end
            else
                TL = exp(work.yT[col]);    μL = work.mu[col];    urL = sinh(work.y[col])
                TR = exp(work.yT[col+1]);  μR = work.mu[col+1];  urR = sinh(work.y[col+1])
                primL = PrimIdealVisc(TL, μL, urL, work.n[col],   work.e[col],   work.P[col],
                                      nurL_phys, PiL_phys, piRL_phys, piEtaL_phys, true)
                primR = PrimIdealVisc(TR, μR, urR, work.n[col+1], work.e[col+1], work.P[col+1],
                                      nurR_phys, PiR_phys, piRR_phys, piEtaR_phys, true)
                @inbounds for a in 1:length(names)
                    ULtmp[a] = U[a,col]
                    URtmp[a] = U[a,col+1]
                end
            end

            flux_cell!(tmpFL, primL, grid.rC[col], τ, model)
            flux_cell!(tmpFR, primR, grid.rC[col+1], τ, model)

            λmL, λpL = wavespeeds_from_prim(primL.T, primL.mu, primL.ur, eos;
                                             Pi_phys=primL.Pi, piR_phys=primL.piR,
                                             e=primL.e, P=primL.P)
            λmR, λpR = wavespeeds_from_prim(primR.T, primR.mu, primR.ur, eos;
                                             Pi_phys=primR.Pi, piR_phys=primR.piR,
                                             e=primR.e, P=primR.P)
            sL = min(λmL, λmR)
            sR = max(λpL, λpR)

            if sL ≥ 0
                @inbounds for a in 1:length(names)
                    Fhtmp[a] = tmpFL[a]
                end
            elseif sR ≤ 0
                @inbounds for a in 1:length(names)
                    Fhtmp[a] = tmpFR[a]
                end
            else
                inv = 1/(sR - sL + TINY)
                @inbounds for a in 1:length(names)
                    Fhtmp[a] = (sR*tmpFL[a] - sL*tmpFR[a] + sR*sL*(URtmp[a] - ULtmp[a])) * inv
                end
            end

            return (valid=true, col=col, rF=grid.rF[col+1], method=method, okL=okL, okR=okR,
                    λmL=λmL, λpL=λpL, λmR=λmR, λpR=λpR, sL=sL, sR=sR,
                    UL=ULtmp, UR=URtmp, FL=tmpFL, FR=tmpFR, Fh=Fhtmp)
        end

        function write_face(face)
            face.valid || return
            rC = grid.rC[i]
            zerosv = fill(0.0, length(names))
            row = Any[
                "face", τ, i, face.col, rC, face.rF,
                face.method, face.okL, face.okR,
                face.λmL, face.λpL, face.λmR, face.λpR, face.sL, face.sR,
                string.(face.UL)...,
                string.(face.UR)...,
                string.(face.FL)...,
                string.(face.FR)...,
                string.(face.Fh)...,
                string.(zerosv)...,
                string.(zerosv)...,
                string.(zerosv)...,
            ]
            println(io, join(string.(row), ","))
        end

        left  = compute_face(i-1)
        right = compute_face(i)
        write_face(left)
        write_face(right)

        # Cell-local decomposition
        Svec = zeros(length(names))
        Smat = reshape(Svec, :, 1)
        Pi_phys    = (L.hasPi    ? phys_from_stored(U[L.iPi,i])    : 0.0)
        piR_phys   = (L.hasPiR   ? phys_from_stored(U[L.iPiR,i])   : 0.0)
        piEta_phys = (L.hasPiEta ? phys_from_stored(U[L.iPiEta,i]) : 0.0)
        source_cell_fast_col!(Smat, 1, U[L.iSr,i], U[L.iE,i],
                              work.P[i], Pi_phys, piR_phys, piEta_phys,
                              sinh(work.y[i]),
                              grid.rC[i], τ, model)

        div = zeros(length(names))
        dU  = zeros(length(names))
        if left.valid && right.valid
            rRp = grid.rF[i+1]
            rRm = grid.rF[i]
            @inbounds for a in 1:length(names)
                div[a] = (rRp*right.Fh[a] - rRm*left.Fh[a]) / grid.dr
                dU[a]  = -(div[a] * safe_inv(grid.rC[i])) + Svec[a]
            end
        end

        nanv = fill(NaN, length(names))
        row = Any[
            "cell", τ, i, 0, grid.rC[i], NaN,
            "", false, false,
            NaN, NaN, NaN, NaN, NaN, NaN,
            string.(nanv)...,
            string.(nanv)...,
            string.(nanv)...,
            string.(nanv)...,
            string.(nanv)...,
            string.(Svec)...,
            string.(div)...,
            string.(dU)...,
        ]
        println(io, join(string.(row), ","))
    end

    return nothing
end


@inline function _maybe_log_sr_clamp!(diag::DiagCounters, U, grid, τ, model::IdealDiffViscModel, work::Work1D, it::Int;
                                     Emin::Float64=E_FLOOR,
                                     χ::Float64=χ_SrE)
    flags = hydro_flags()
    flags.log_srscale || return nothing

    diag.last_Sr_scaled_cells > 0 || return nothing
    i = diag.last_Sr_scaled_max_i
    i == 0 && return nothing

    ratio = diag.last_Sr_scaled_maxratio
    ratio >= (1 + flags.log_srscale_eps) || return nothing

    (flags.log_srscale_every <= 1 || (it % flags.log_srscale_every) == 0) || return nothing

    L = layout(model)
    r = grid.rC[i]

    D = safe_div(U[L.iDtau, i], τ)
    Sr = U[L.iSr,i]
    E  = U[L.iE,i]

    nur = (L.hasNur ? phys_from_stored(U[L.iNur,i]) : 0.0)
    Π   = (L.hasPi  ? phys_from_stored(U[L.iPi,i])  : 0.0)
    pr  = (L.hasPiR ? phys_from_stored(U[L.iPiR,i]) : 0.0)
    pe  = (L.hasPiEta ? phys_from_stored(U[L.iPiEta,i]) : 0.0)

    v = (work.ok[i] ? work.vC[i] : NaN)
    T = (work.ok[i] ? exp(work.yT[i]) : NaN)
    μ = (work.ok[i] ? work.mu[i] : NaN)
    P = (work.ok[i] ? work.P[i] : NaN)
    e = (work.ok[i] ? work.e[i] : NaN)

    Pguess = (work.ok[i] ? max(P, 0.0) : 0.0)
    Peff = Pguess + abs(Π) + abs(pr) + abs(pe)
    Srmax = χ * (E + Peff)
    cap = (work.ok[i] ? (e + P) : NaN)

    @warn "Sr constraint hit" it=it τ=τ i=i r=r ratio=ratio Sr=Sr Srmax=Srmax E=E D=D v=v Pguess=Pguess Peff=Peff cap=cap T=T mu=μ nur=nur Pi=Π piR=pr piEta=pe

    if flags.dump_srscale
        f = joinpath("debug_srsclamp", @sprintf("srsclamp_tau_%06.3f_it_%d_i_%d.csv", τ, it, i))
        _dump_failure_window_csv(f, U, grid, τ, model, work, i; radius=5)
        @info "wrote Sr clamp window" file=f
    end

    return nothing
end

@inline function _abort_on_first_primfail!(diag::DiagCounters, U, grid, τ, model::IdealDiffViscModel, work::Work1D, it::Int;
                                          Emin::Float64=E_FLOOR,
                                          χ::Float64=χ_SrE)
    flags = hydro_flags()
    flags.abort_on_first_primfail || return nothing
    diag.last_prim_fail_cells > 0 || return nothing

    i = diag.last_prim_fail_i
    i == 0 && return nothing

    L = layout(model)
    r = (1 <= i <= length(grid.rC)) ? grid.rC[i] : NaN

    Dtau = diag.last_prim_fail_Dtau
    Sr   = diag.last_prim_fail_Sr
    E    = diag.last_prim_fail_E
    D = safe_div(Dtau, τ)

    prr = diag.last_prim_fail_reason
    prit = diag.last_prim_fail_iters
    prres = diag.last_prim_fail_resnorm

    nur = phys_from_stored(diag.last_prim_fail_nur_stored)
    Π   = phys_from_stored(diag.last_prim_fail_Pi_stored)
    pr  = phys_from_stored(diag.last_prim_fail_piR_stored)
    pe  = phys_from_stored(diag.last_prim_fail_piEta_stored)

    P = (1 <= i <= length(work.P) && work.ok[i]) ? work.P[i] : NaN
    Pguess = (isfinite(P) ? max(P, 0.0) : 0.0)
    Peff = Pguess + abs(Π) + abs(pr) + abs(pe)
    Srmax = χ * (E + Peff)
    Sr_over_Srmax = (isfinite(Sr) && isfinite(Srmax) && Srmax > 0) ? (abs(Sr) / Srmax) : NaN
    Sr_over_E = (isfinite(Sr) && isfinite(E) && E > 0) ? (abs(Sr) / E) : NaN

    yT0  = work.x0_yT[i]
    φ0   = work.x0_phi[i]
    y0   = work.x0_y[i]
    v0   = tanh(y0)

    v = (1 <= i <= length(work.vC) && work.ok[i]) ? work.vC[i] : NaN
    T = (1 <= i <= length(work.yT) && work.ok[i]) ? exp(work.yT[i]) : NaN
    μ = (1 <= i <= length(work.mu) && work.ok[i]) ? work.mu[i] : NaN

    @error "ABORT: primitive recovery failed" it=it τ=τ i=i r=r χ=χ Emin=Emin Dtau=Dtau D=D Sr=Sr E=E Sr_over_E=Sr_over_E Srmax=Srmax Sr_over_Srmax=Sr_over_Srmax primrec_reason=prr primrec_iters=prit primrec_resnorm=prres Pguess=Pguess Peff=Peff nur=nur Pi=Π piR=pr piEta=pe yT0=yT0 phi0=φ0 y0=y0 v0=v0 v=v T=T mu=μ

    f = joinpath("debug_primfail", @sprintf("first_primfail_tau_%06.3f_it_%d_i_%d.csv", τ, it, i))
    _dump_failure_window_csv(f, U, grid, τ, model, work, i; radius=8)
    @error "ABORT: wrote primfail window" file=f

    ff = joinpath("debug_flux", @sprintf("flux_primfail_tau_%06.3f_it_%d_i_%d.csv", τ, it, i))
    _dump_flux_decomp_csv(ff, U, grid, τ, model, work, i; Emin=Emin)
    @error "ABORT: wrote flux decomposition" file=ff

    error("Aborting on first primitive recovery failure (HYDRO_ABORT_ON_FIRST_PRIMFAIL=1).")
end

@inline function _abort_on_first_srscale!(diag::DiagCounters, U, grid, τ, model::IdealDiffViscModel, work::Work1D, it::Int;
                                         Emin::Float64=E_FLOOR,
                                         χ::Float64=χ_SrE)
    flags = hydro_flags()
    flags.abort_on_first_srscale || return nothing
    diag.Sr_scaled_cells > 0 || return nothing

    τ >= flags.abort_srscale_min_tau || return nothing
    diag.last_Sr_scaled_maxratio >= (1 + flags.abort_srscale_eps) || return nothing

    i = diag.last_Sr_scaled_max_i
    if i == 0
        i = diag.last_prim_fail_i
    end

    L = layout(model)
    r = (1 <= i <= length(grid.rC)) ? grid.rC[i] : NaN

    D = (1 <= i <= size(U,2)) ? safe_div(U[L.iDtau, i], τ) : NaN
    Sr = (1 <= i <= size(U,2)) ? U[L.iSr,i] : NaN
    E  = (1 <= i <= size(U,2)) ? U[L.iE,i] : NaN
    ratio_pre  = diag.last_Sr_scaled_maxratio

    nur = (L.hasNur && 1 <= i <= size(U,2)) ? phys_from_stored(U[L.iNur,i]) : 0.0
    Π   = (L.hasPi  && 1 <= i <= size(U,2)) ? phys_from_stored(U[L.iPi,i])  : 0.0
    pr  = (L.hasPiR && 1 <= i <= size(U,2)) ? phys_from_stored(U[L.iPiR,i]) : 0.0
    pe  = (L.hasPiEta && 1 <= i <= size(U,2)) ? phys_from_stored(U[L.iPiEta,i]) : 0.0

    v = (1 <= i <= length(work.vC) && work.ok[i]) ? work.vC[i] : NaN
    T = (1 <= i <= length(work.yT) && work.ok[i]) ? exp(work.yT[i]) : NaN
    μ = (1 <= i <= length(work.mu) && work.ok[i]) ? work.mu[i] : NaN
    P = (1 <= i <= length(work.P) && work.ok[i]) ? work.P[i] : NaN
    e = (1 <= i <= length(work.e) && work.ok[i]) ? work.e[i] : NaN

    Pguess = (isfinite(P) ? max(P, 0.0) : 0.0)
    Peff = Pguess + abs(Π) + abs(pr) + abs(pe)
    Srmax = χ * (E + Peff)
    ratio_post = (isfinite(Sr) && isfinite(Srmax) && Srmax > 0) ? (abs(Sr) / Srmax) : NaN

    (isfinite(ratio_post) && ratio_post > (1 + flags.abort_srscale_post_eps)) || return nothing

    @error "ABORT: Sr constraint triggered" it=it τ=τ χ=χ Emin=Emin SrScaled_total=diag.Sr_scaled_cells SrScaled_last=diag.last_Sr_scaled_cells SrScaledMax_last=diag.last_Sr_scaled_maxratio SrScaledMaxI_last=diag.last_Sr_scaled_max_i i=i r=r ratio_pre=ratio_pre ratio_post=ratio_post Sr=Sr Srmax=Srmax E=E D=D v=v T=T mu=μ P=P Pguess=Pguess Peff=Peff e=e nur=nur Pi=Π piR=pr piEta=pe

    f = joinpath("debug_srsclamp", @sprintf("first_srsclamp_tau_%06.3f_it_%d_i_%d.csv", τ, it, i))
    _dump_failure_window_csv(f, U, grid, τ, model, work, i; radius=8)
    @error "ABORT: wrote Sr clamp window" file=f

    ff = joinpath("debug_flux", @sprintf("flux_srsclamp_tau_%06.3f_it_%d_i_%d.csv", τ, it, i))
    _dump_flux_decomp_csv(ff, U, grid, τ, model, work, i; Emin=Emin)
    @error "ABORT: wrote flux decomposition" file=ff

    error("Aborting on first Sr constraint hit (HYDRO_ABORT_ON_FIRST_SRSCALE=1).")
end

@inline function _maybe_log_primfail!(diag::DiagCounters, U, grid, τ, model::IdealDiffViscModel, work::Work1D, it::Int;
                                     Emin::Float64=E_FLOOR,
                                     χ::Float64=χ_SrE)
    flags = hydro_flags()
    flags.log_primfail || return nothing

    i = diag.last_prim_fail_i
    i == 0 && return nothing

    (flags.log_primfail_every <= 1 || (it % flags.log_primfail_every) == 0) || return nothing

    L = layout(model)
    r = grid.rC[i]

    Dtau = diag.last_prim_fail_Dtau
    Sr   = diag.last_prim_fail_Sr
    E    = diag.last_prim_fail_E

    D = safe_div(Dtau, τ)
    Sr_over_E = safe_div(abs(Sr), E)
    margin_E_S = E - abs(Sr)

    nur = phys_from_stored(diag.last_prim_fail_nur_stored)
    Π   = phys_from_stored(diag.last_prim_fail_Pi_stored)
    pr  = phys_from_stored(diag.last_prim_fail_piR_stored)
    pe  = phys_from_stored(diag.last_prim_fail_piEta_stored)

    yT0  = work.x0_yT[i]
    φ0   = work.x0_phi[i]
    y0   = work.x0_y[i]
    v0   = tanh(y0)

    prr = diag.last_prim_fail_reason
    prit = diag.last_prim_fail_iters
    prres = diag.last_prim_fail_resnorm

    @error "primitive recovery failed" it=it τ=τ i=i r=r χ=χ Emin=Emin Dtau=Dtau D=D Sr=Sr E=E Sr_over_E=Sr_over_E margin_E_minus_absSr=margin_E_S primrec_reason=prr primrec_iters=prit primrec_resnorm=prres nur=nur Pi=Π piR=pr piEta=pe yT0=yT0 phi0=φ0 y0=y0 v0=v0

    if flags.dump_primfail
        f = joinpath("debug_primfail", @sprintf("primfail_tau_%06.3f_it_%d_i_%d.csv", τ, it, i))
        _dump_failure_window_csv(f, U, grid, τ, model, work, i; radius=5)
        @info "wrote primfail window" file=f
    end

    return nothing
end
