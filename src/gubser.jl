# ==============================================================================
# src/gubser.jl
#
# Ideal Gubser flow (boost-invariant, azimuthally symmetric) analytic solution
# in Milne coordinates (τ, r, ϕ, η).
#
# References for formulas:
# - S. S. Gubser, Phys. Rev. D82 (2010) 085027
#
# Conventions used here (ideal conformal):
#   v_r(τ,r) = (2 q^2 τ r) / (1 + q^2 τ^2 + q^2 r^2)
#   T(τ,r)   = Tscale * (2q)^(2/3) / ( τ^(1/3) * Den(τ,r)^(1/3) )
#   Den(τ,r) = 1 + 2 q^2(τ^2+r^2) + q^4(τ^2-r^2)^2
# where q has units 1/fm.
#
# The solver uses ur = u^r (contravariant component), with u^τ = sqrt(1+ur^2)
# and v = ur/u^τ.
# ==============================================================================;

@inline function gubser_den(τ::Float64, r::Float64, q::Float64)
    q2 = q*q
    τ2 = τ*τ
    r2 = r*r
    return 1.0 + 2.0*q2*(τ2 + r2) + (q2*q2)*((τ2 - r2)^2)
end

@inline function gubser_vr(τ::Float64, r::Float64, q::Float64)
    q2 = q*q
    num = 2.0*q2*τ*r
    den = 1.0 + q2*(τ*τ) + q2*(r*r)
    return safe_div(num, den)
end

@inline function gubser_ur(τ::Float64, r::Float64, q::Float64)
    v = clamp(gubser_vr(τ, r, q), -0.999999999, 0.999999999)
    return v / sqrt(1.0 - v*v)
end

@inline function gubser_temperature(τ::Float64, r::Float64, q::Float64, Tscale::Float64)
    Den = gubser_den(τ, r, q)
    # Keep q strictly positive in the scaling to avoid complex (2q)^(2/3)
    qpos = posden(abs(q))
    return Tscale * (2.0*qpos)^(2/3) / (posden(τ)^(1/3) * posden(Den)^(1/3))
end

"""
Compute the normalization Tscale given a desired central temperature Tc0
at (τ0, r=0).

At r=0 one has Den(τ,0) = (1 + q^2 τ^2)^2.
"""
@inline function gubser_Tscale_from_center_T(τ0::Float64, q::Float64, Tc0::Float64)
    qpos = posden(abs(q))
    fac = (posden(τ0)^(1/3)) * (1.0 + (qpos*τ0)^2)^(2/3)
    return posden(Tc0) * fac / (2.0*qpos)^(2/3)
end

"""Return ideal (dissipation-free) primitive state for Gubser flow at (τ,r)."""
@inline function gubser_prim_ideal(τ::Float64, r::Float64, eos;
                                  q::Float64,
                                  Tscale::Float64,
                                  μ::Float64 = 0.0)
    T  = gubser_temperature(τ, r, q, Tscale)
    ur = gubser_ur(τ, r, q)
    P, n, e = eos_Pne(T, μ, eos)
    return PrimIdealVisc(T, μ, ur, n, e, P, 0.0, 0.0, 0.0, 0.0, true)
end

"""Initialize the conservative state U to ideal Gubser flow at τ0.

This sets dissipatives to zero and uses the provided EOS to compute (P,n,e).
Recommended for validation: use a conformal EOS without charge sector, e.g.
`ConformalHQEOS(g_eff=40.0, m_hq=0.0, g_hq=0.0)`.
"""
function initialize_gubser!(U, grid::Grid1D, τ0::Float64, model::IdealDiffViscModel;
                           q::Float64,
                           Tc0::Float64,
                           μ0::Float64 = 0.0,
                           Emin::Float64 = E_FLOOR,
                           χ::Float64 = χ_SrE)

    L  = layout(model)
    ng = grid.nghost
    Ntot = size(U,2)
    fill!(U, 0.0)

    Tscale = gubser_Tscale_from_center_T(τ0, q, Tc0)

    @inbounds for i in (ng+1):(Ntot-ng)
        r = grid.rC[i]
        prim = gubser_prim_ideal(τ0, r, model.eos; q=q, Tscale=Tscale, μ=μ0)

        uτ = sqrt(1.0 + prim.ur^2)
        w  = prim.e + prim.P

        # dissipatives are zero for ideal Gubser
        D = prim.n * uτ

        U[L.iDtau, i] = τ0 * D
        U[L.iSr,  i]  = w * (uτ * prim.ur)
        U[L.iE,   i]  = w * (uτ^2) - prim.P

        if L.hasNur;   U[L.iNur, i]   = 0.0; end
        if L.hasPi;    U[L.iPi, i]    = 0.0; end
        if L.hasPiR;   U[L.iPiR, i]   = 0.0; end
        if L.hasPiEta; U[L.iPiEta, i] = 0.0; end
    end

    apply_bc!(U, grid, τ0, model)
    enforce_floors!(U, grid, τ0, model; Emin=Emin, diag=nothing)
    enforce_Sr_energy_constraint!(U, grid, model; χ=χ, mask=nothing, diag=nothing)
    return nothing
end

"""Compute relative L2 errors (area-weighted) of numeric state vs ideal Gubser analytic.

Returns a named tuple: (err_T, err_e, err_ur).
Weights are proportional to transverse area element: w(r) = r * dr (constant factors cancel).
"""
function gubser_relL2_errors(U, grid::Grid1D, τ::Float64, model::IdealDiffViscModel;
                            q::Float64,
                            Tc0::Float64,
                            τref::Float64,
                            μ0::Float64 = 0.0)

    # normalize analytic using the same reference definition as initialization
    Tscale = gubser_Tscale_from_center_T(τref, q, Tc0)

    ng = grid.nghost
    i0 = ng + 1
    iL = size(U,2) - ng

    numT = 0.0; denT = 0.0
    nume = 0.0; dene = 0.0
    numu = 0.0; denu = 0.0

    @inbounds for i in i0:iL
        r = max(grid.rC[i], 0.0)
        wgt = r * grid.dr

        prim_num = cons_to_prim_col(U, i, grid.rC[i], τ, model)
        prim_ana = gubser_prim_ideal(τ, grid.rC[i], model.eos; q=q, Tscale=Tscale, μ=μ0)

        dT  = prim_num.T  - prim_ana.T
        de  = prim_num.e  - prim_ana.e
        dur = prim_num.ur - prim_ana.ur

        numT += wgt * dT*dT
        denT += wgt * (prim_ana.T*prim_ana.T + TINY)

        nume += wgt * de*de
        dene += wgt * (prim_ana.e*prim_ana.e + TINY)

        numu += wgt * dur*dur
        denu += wgt * (prim_ana.ur*prim_ana.ur + TINY)
    end

    return (
        err_T  = sqrt(numT / posden(denT)),
        err_e  = sqrt(nume / posden(dene)),
        err_ur = sqrt(numu / posden(denu)),
    )
end

"""Relative L2 errors for *conserved* variables U vs analytic ideal Gubser.

This avoids any dependence on primitive recovery and therefore cleanly validates
the flux + source update.

Returns: (err_E, err_Sr, err_Dtau)
"""
function gubser_relL2_errors_cons(U, grid::Grid1D, τ::Float64, model::IdealDiffViscModel;
                                 q::Float64,
                                 Tc0::Float64,
                                 τref::Float64,
                                 μ0::Float64 = 0.0)

    Tscale = gubser_Tscale_from_center_T(τref, q, Tc0)
    L = layout(model)

    ng = grid.nghost
    i0 = ng + 1
    iL = size(U,2) - ng

    numE = 0.0; denE = 0.0
    numS = 0.0; denS = 0.0
    numD = 0.0; denD = 0.0

    # 1-cell scratch for analytic U
    Uana = zeros(Float64, length(L.names), 1)

    @inbounds for i in i0:iL
        r = max(grid.rC[i], 0.0)
        wgt = r * grid.dr

        # Analytic primitives (ideal)
        T  = gubser_temperature(τ, grid.rC[i], q, Tscale)
        ur = gubser_ur(τ, grid.rC[i], q)

        yT = log(max(T, T_MIN))
        y  = asinh(ur)
        φ  = 0.0

        ok, _ = prim_to_cons_col_ideal_phi_diff_visc!(Uana, 1, yT, φ, y,
                                                     0.0, 0.0, 0.0, 0.0,
                                                     grid.rC[i], τ,
                                                     model.eos, L)
        ok || continue

        dE = U[L.iE, i]  - Uana[L.iE, 1]
        dS = U[L.iSr, i] - Uana[L.iSr, 1]
        dD = U[L.iDtau, i] - Uana[L.iDtau, 1]

        numE += wgt * dE*dE
        denE += wgt * (Uana[L.iE,1]^2 + TINY)

        numS += wgt * dS*dS
        denS += wgt * (Uana[L.iSr,1]^2 + TINY)

        numD += wgt * dD*dD
        denD += wgt * (Uana[L.iDtau,1]^2 + TINY)
    end

    return (
        err_E    = sqrt(numE / posden(denE)),
        err_Sr   = sqrt(numS / posden(denS)),
        err_Dtau = sqrt(numD / posden(denD)),
    )
end
