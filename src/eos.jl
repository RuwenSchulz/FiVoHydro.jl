# =========================
# src/eos.jl
# =========================
# Depends on: T_MIN, fmGeV3, safe_besselkx, clamp
# Also depends on: Interpolations (for TabulatedHQEOS constructor)
# ===============================================================

const π2 = π^2

# -------------------------
# Conformal light + HQ Boltzmann
# -------------------------
Base.@kwdef struct ConformalHQEOS
    g_eff::Float64 = 40.0
    m_hq::Float64  = 1.5
    g_hq::Float64  = 6.0

    # If true, multiply the HQ-sector (P,n,e) by fmGeV3 so units match the light sector.
    # Default is false to preserve historical behavior.
    hq_times_fmGeV3::Bool = false
end

@inline hq_mass(eos::ConformalHQEOS) = eos.m_hq
@inline a_SB(eos::ConformalHQEOS) = (π2/90) * eos.g_eff
@inline eos_cs2(::Float64, ::Float64, ::ConformalHQEOS) = 1/3

@inline function eos_Pne(T::Float64, μ::Float64, eos::ConformalHQEOS)
    T  = max(T, T_MIN)

    Plight = a_SB(eos) * T^4 * fmGeV3
    elight = 3.0 * Plight

    m = hq_mass(eos)
    g = eos.g_hq
    x = m / T
    if !(isfinite(x)) || x <= 0.0
        return Plight, 0.0, elight
    end

    A   = g * m^2 / (2π^2)
    K2x = SpecialFunctions.besselkx(2, x)
    K1x = SpecialFunctions.besselkx(1, x)

    exp_arg = (μ - m) / T
    Efac    = exp(exp_arg)

    n_hq = A * T   * Efac * K2x
    P_hq = A * T^2 * Efac * K2x
    e_hq = A * T^2 * Efac * (3.0*K2x + x*K1x)

    if eos.hq_times_fmGeV3
        n_hq *= fmGeV3
        P_hq *= fmGeV3
        e_hq *= fmGeV3
    end

    return Plight + P_hq, n_hq, elight + e_hq
end


# -------------------------
# Running conformal light + HQ Boltzmann
# -------------------------
Base.@kwdef struct RunningConformalHQEOS
    g_eff::Float64 = 40.0
    δ::Float64     = 1e-4   # deformation strength
    T0::Float64    = 0.2   # GeV

    m_hq::Float64  = 1.5
    g_hq::Float64  = 6.0

    # If true, multiply the HQ-sector (P,n,e) by fmGeV3 so units match the light sector.
    # Default is false to preserve historical behavior.
    hq_times_fmGeV3::Bool = false
end

@inline hq_mass(eos::RunningConformalHQEOS) = eos.m_hq
@inline a_SB(eos::RunningConformalHQEOS) = (π2/90) * eos.g_eff

@inline @fastmath function _Plight_elight(T::Float64, eos::RunningConformalHQEOS)
    a = a_SB(eos)
    b = a * eos.δ * eos.T0^2
    T2 = T*T
    T4 = T2*T2
    Plight = (a*T4 + b*T2) * fmGeV3
    elight = (3.0*a*T4 + b*T2) * fmGeV3
    return Plight, elight
end

@inline function eos_Pne(T::Float64, μ::Float64, eos::RunningConformalHQEOS)
    T = max(T, T_MIN)

    Plight, elight = _Plight_elight(T, eos)

    m = hq_mass(eos)
    g = eos.g_hq
    x = m / T
    #x < 1e-100 && @warn "safe_besselkx called with very small x=$(x),T=$(T),m=$(m)" 
    #x > 1e100  && @warn "safe_besselkx called with very large x=$(x),T=$(T),m=$(m)" 
    if !(isfinite(x)) || x <= 0.0
        return Plight, 0.0, elight
    end

    A   = g * m^2 / (2π^2)
    K2x = SpecialFunctions.besselkx(2, x)
    K1x = SpecialFunctions.besselkx(1, x)

    exp_arg = (μ - m) / T
    Efac    = exp(exp_arg)

    n_hq = A * T   * Efac * K2x
    P_hq = A * T^2 * Efac * K2x
    e_hq = A * T^2 * Efac * (3.0*K2x + x*K1x)

    if eos.hq_times_fmGeV3
        n_hq *= fmGeV3
        P_hq *= fmGeV3
        e_hq *= fmGeV3
    end

    return Plight + P_hq, n_hq, elight + e_hq
end


@inline function eos_cs2(T::Float64, μ::Float64, eos::RunningConformalHQEOS;
                         relstep::Float64 = 1e-3,
                         absstep::Float64 = 1e-6)

    # "Domain-safe" only: we avoid evaluating eos_Pne below T_EOS_MIN.
    # This is not a physics correction; it's preventing an invalid EOS call.
    T0 = max(T, T_EOS_MIN)

    h = max(relstep*T0, absstep)

    Pp = Pm = ep = em = NaN
    if T0 - h <= T_EOS_MIN
        P0, _, e0 = eos_Pne(T0,     μ, eos)
        Pp, _, ep = eos_Pne(T0 + h, μ, eos)
        dP = (Pp - P0) / h
        de = (ep - e0) / h
    else
        Pp, _, ep = eos_Pne(T0 + h, μ, eos)
        Pm, _, em = eos_Pne(T0 - h, μ, eos)
        dP = (Pp - Pm) / (2h)
        de = (ep - em) / (2h)
    end

    cs2 = dP / de

    # No clamp, no fallback. Only WARN / THROW.
    bad = (!isfinite(cs2) || !isfinite(dP) || !isfinite(de) || de == 0.0)

    if bad
        if !EOS_WARN_ONCE[]
            EOS_WARN_ONCE[] = true
            @warn "eos_cs2 produced invalid result" T=T μ=μ T0=T0 h=h dP=dP de=de cs2=cs2 Pp=Pp Pm=Pm ep=ep em=em
        end
        if EOS_STRICT_THROW
            error("eos_cs2 invalid: cs2=$(cs2), dP=$(dP), de=$(de), T=$(T), μ=$(μ)")
        end
        return cs2  # likely NaN/Inf
    end

    return cs2
end









# -------------------------
# Tabulated wrapper EOS: interpolates P(T,α), n(T,α), e(T,α)
# -------------------------
struct TabulatedHQEOS{ITP}
    base
    itpP::ITP
    itpn::ITP
    itpe::ITP
    Tmin::Float64
    Tmax::Float64
    αmin::Float64
    αmax::Float64
end

@inline hq_mass(eos::TabulatedHQEOS) = hq_mass(eos.base)
@inline a_SB(eos::TabulatedHQEOS) = a_SB(eos.base)

# src/eos_tab.jl


# cs2 for tabulated EOS: numeric derivative at fixed μ (matches your RunningConformal method)
@inline function eos_cs2(T::Float64, μ::Float64, eos::TabulatedHQEOS;
                        relstep::Float64=1e-3, absstep::Float64=1e-6)
    T0 = max(T, eos.Tmin)
    h  = max(relstep*T0, absstep)

    Pp, _, ep = eos_Pne(T0 + h, μ, eos)
    Pm, _, em = eos_Pne(max(T0 - h, eos.Tmin), μ, eos)

    dP = (Pp - Pm) / (2h)
    de = (ep - em) / (2h)

    return clamp(dP / max(de, eps(Float64)), 0.0, 1.0)
end



function TabulatedHQEOS(base;
                        Tmin::Float64=T_MIN, Tmax::Float64=0.7,
                        αmin::Float64=0.0, αmax::Float64=200.0,
                        NT::Int=500, Nα::Int=5000)

    Ts = collect(range(Tmin, Tmax; length=NT))
    αs = collect(range(αmin, αmax; length=Nα))

    P = Array{Float64}(undef, NT, Nα)
    n = Array{Float64}(undef, NT, Nα)
    e = Array{Float64}(undef, NT, Nα)

    @inbounds for j in 1:Nα, i in 1:NT
        T = Ts[i]
        α = αs[j]
        μ = α*T
        P[i,j], n[i,j], e[i,j] = eos_Pne(T, μ, base)
    end

    itpP = extrapolate(interpolate((Ts, αs), P, Gridded(Linear())), Flat())
    itpn = extrapolate(interpolate((Ts, αs), n, Gridded(Linear())), Flat())
    itpe = extrapolate(interpolate((Ts, αs), e, Gridded(Linear())), Flat())

    return TabulatedHQEOS(base, itpP, itpn, itpe, Tmin, Tmax, αmin, αmax)
end

@inline function eos_Pne(T::Float64, μ::Float64, eos::TabulatedHQEOS)
    Tuse = max(T, eos.Tmin)
    α = safe_div(μ, Tuse)

    if (Tuse < eos.Tmin) || (Tuse > eos.Tmax) || (α < eos.αmin) || (α > eos.αmax)
        return eos_Pne(Tuse, μ, eos.base)   # <-- robust fallback
    end

    return (eos.itpP(Tuse, α), eos.itpn(Tuse, α), eos.itpe(Tuse, α))
end


# -------------------------
# Chiral-modified EOS wrapper (SoftPion two-way coupling).
# Wraps any base EOS with a per-cell additive (ΔP, Δn, Δe) from the soft-pion
# subtract-and-replace sector, precomputed by the coupled driver from the local chiral
# field σ (held frozen during a hydro sub-step — operator splitting). Constructed cheaply
# per cell, so it is thread-safe. With zero deltas it reduces EXACTLY to the base EOS, so
# default behaviour of every production run is untouched (this is a new type + new methods
# only; nothing dispatches here unless a ChiralModifiedEOS is explicitly passed). cs² and
# hq_mass are inherited from the base: the soft-pion shift is small and localized below
# T_pc, so the constant ΔP/Δe leave dP/dT (hence cs²) unchanged to leading order.
# -------------------------
struct ChiralModifiedEOS{B}
    base::B
    dP::Float64
    dn::Float64
    de::Float64
end
ChiralModifiedEOS(base) = ChiralModifiedEOS(base, 0.0, 0.0, 0.0)

@inline hq_mass(eos::ChiralModifiedEOS) = hq_mass(eos.base)
@inline eos_cs2(T::Float64, μ::Float64, eos::ChiralModifiedEOS) = eos_cs2(T, μ, eos.base)
@inline function eos_Pne(T::Float64, μ::Float64, eos::ChiralModifiedEOS)
    P, n, e = eos_Pne(T, μ, eos.base)
    return P + eos.dP, n + eos.dn, e + eos.de
end


# -------------------------
# Generic thermodynamic helper (OK to keep generic)
# -------------------------
@inline function eos_entropy(T::Float64, μ::Float64, n::Float64, e::Float64, P::Float64)
    T = max(T, T_MIN)
    s = (e + P - μ*n) / T
    return max(s, 0.0)
end




# -------------------------
# Fit light-sector + HQ Boltzmann EOS
# -------------------------
Base.@kwdef struct LatticeHRGEOS
    # --- light-sector fit params (from "Heavy_Quark") ---
    a1::Float64 = -15.526548963383643
    a2::Float64 =  18.6159584620131
    a3::Float64 = -10.731808109698516
    a4::Float64 =  2.7413302179949346

    b1::Float64 = -3.3147904483107595
    b2::Float64 =  5.310983721567554
    b3::Float64 = -4.653922019495976
    b4::Float64 =  1.8600649533271152

    c::Float64  = -1.0465330501411811
    d::Float64  =  0.09551531822245873

    # --- HQ Boltzmann params (your sector) ---
    m_hq::Float64 = 1.5
    g_hq::Float64 = 6.0

    # If your HQ formulas are in GeV^4 and light is already fmGeV3-scaled,
    # set this true to multiply HQ (P,n,e) by fmGeV3 as well.
    hq_times_fmGeV3::Bool = true

    # Canonical-ensemble suppression of the charm density, I₁(N/2)/I₀(N/2). Default = the
    # hardcoded N=21.55 value; pass a per-system value (e.g. matching Fluidum's N_total) to make
    # the charm EOS identical to Fluidum's for a given collision system.
    canon_factor::Float64 = _LHRG_CCBAR_FACT
end

const _LHRG_CCBAR = 21.55
const _LHRG_CCBAR_FACT = SpecialFunctions.besseli(1, _LHRG_CCBAR/2) / SpecialFunctions.besseli(0, _LHRG_CCBAR/2)
canonical_factor(N::Real) = SpecialFunctions.besseli(1, N/2) / SpecialFunctions.besseli(0, N/2)

@inline hq_mass(eos::LatticeHRGEOS) = eos.m_hq

# -------------------------
# Light-sector fit: P(T), dP/dT, d²P/dT²
# (algebraically identical to your pressure(T,x::Heavy_Quark))
# -------------------------
@inline function light_P(T::Float64, eos::LatticeHRGEOS)
    T = max(T, T_MIN)

    k0 = 0.0005624486560000001
    k1 = 0.003652264
    k2 = 0.023716
    k3 = 0.154
    k4 = 5.208957878352717

    # exp[ -(0.01 d^2)/T^2 - (0.1 c^2)/T ]
    A  = 0.01 * eos.d^2
    B  = 0.1  * eos.c^2
    E  = -(A/(T*T)) - (B/T)
    E  = clamp(E, -700.0, 700.0)
    expE = exp(E)

    N = k0*eos.a4 + k1*eos.a3*T + k2*eos.a2*T^2 + k3*eos.a1*T^3 + k4*T^4
    D = k0*eos.b4 + k1*eos.b3*T + k2*eos.b2*T^2 + k3*eos.b1*T^3 + T^4

    return fmGeV3 * expE * (T^4) * (N/D)
end

@inline function light_dP_dT(T::Float64, eos::LatticeHRGEOS)
    T = max(T, T_MIN)

    k0 = 0.0005624486560000001
    k1 = 0.003652264
    k2 = 0.023716
    k3 = 0.154
    k4 = 5.208957878352717

    A  = 0.01 * eos.d^2
    B  = 0.1  * eos.c^2
    E  = -(A/(T*T)) - (B/T)
    E  = clamp(E, -700.0, 700.0)
    expE = exp(E)

    Ep = (2A/(T^3)) + (B/(T^2))          # dE/dT

    N  = k0*eos.a4 + k1*eos.a3*T + k2*eos.a2*T^2 + k3*eos.a1*T^3 + k4*T^4
    D  = k0*eos.b4 + k1*eos.b3*T + k2*eos.b2*T^2 + k3*eos.b1*T^3 + T^4
    Np = k1*eos.a3 + 2k2*eos.a2*T + 3k3*eos.a1*T^2 + 4k4*T^3
    Dp = k1*eos.b3 + 2k2*eos.b2*T + 3k3*eos.b1*T^2 + 4*T^3

    # F(T) = T^4 * N / D
    G  = T^4 * N
    H  = D
    Gp = 4*T^3*N + T^4*Np
    Hp = Dp

    F  = G / H
    Fp = (Gp*H - G*Hp) / (H*H)

    # P = fmGeV3 * exp(E) * F
    return fmGeV3 * expE * (Fp + F*Ep)
end

@inline function light_d2P_dT2(T::Float64, eos::LatticeHRGEOS)
    T = max(T, T_MIN)

    k0 = 0.0005624486560000001
    k1 = 0.003652264
    k2 = 0.023716
    k3 = 0.154
    k4 = 5.208957878352717

    A  = 0.01 * eos.d^2
    B  = 0.1  * eos.c^2
    E  = -(A/(T*T)) - (B/T)
    E  = clamp(E, -700.0, 700.0)
    expE = exp(E)

    Ep  = (2A/(T^3)) + (B/(T^2))          # E'
    Epp = (-6A/(T^4)) - (2B/(T^3))        # E''

    N   = k0*eos.a4 + k1*eos.a3*T + k2*eos.a2*T^2 + k3*eos.a1*T^3 + k4*T^4
    D   = k0*eos.b4 + k1*eos.b3*T + k2*eos.b2*T^2 + k3*eos.b1*T^3 + T^4

    Np  = k1*eos.a3 + 2k2*eos.a2*T + 3k3*eos.a1*T^2 + 4k4*T^3
    Dp  = k1*eos.b3 + 2k2*eos.b2*T + 3k3*eos.b1*T^2 + 4*T^3

    Npp = 2k2*eos.a2 + 6k3*eos.a1*T + 12k4*T^2
    Dpp = 2k2*eos.b2 + 6k3*eos.b1*T + 12*T^2

    # F(T) = G/H with G=T^4 N, H=D
    G   = T^4 * N
    H   = D

    Gp  = 4*T^3*N + T^4*Np
    Hp  = Dp

    Gpp = 12*T^2*N + 8*T^3*Np + T^4*Npp
    Hpp = Dpp

    F   = G / H
    Fp  = (Gp*H - G*Hp) / (H*H)

    # F'' using Q = (Gp*H - G*Hp), Q' = Gpp*H - G*Hpp
    Q   = (Gp*H - G*Hp)
    Qp  = (Gpp*H - G*Hpp)
    Fpp = (Qp*H - 2Q*Hp) / (H^3)

    # P = C exp(E) F
    # P'  = C exp(E) (F' + F E')
    # P'' = C exp(E) (F'' + 2F' E' + F (E'' + (E')^2))
    return fmGeV3 * expE * (Fpp + 2*Fp*Ep + F*(Epp + Ep*Ep))
end

# Light energy density (μ_light = 0): e = -P + T dP/dT
@inline function light_e(T::Float64, eos::LatticeHRGEOS)
    P  = light_P(T, eos)
    dP = light_dP_dT(T, eos)
    return -P + T*dP
end

# cs^2 = dP/de = P'/(T P'')  (since e' = T P'')
@inline function eos_cs2(T::Float64, ::Float64, eos::LatticeHRGEOS)
    T  = max(T, T_MIN)
    Pp = light_dP_dT(T, eos)
    Ppp = light_d2P_dT2(T, eos)
    return (Ppp == 0.0) ? (1/3) : (Pp / (T*Ppp))
end

# -------------------------
# Total interface: (P, n, e) = light + HQ Boltzmann
# -------------------------
@inline function eos_Pne(T::Float64, μ::Float64, eos::LatticeHRGEOS)
    T = max(T, T_MIN)

    # light sector
    Pl = light_P(T, eos)
    el = light_e(T, eos)

    # HQ Boltzmann sector (your previous implementation)
    m = hq_mass(eos)
    g = eos.g_hq
    x = m / T
    x = min(x, 10^5)
    if !(isfinite(x)) || x <= 0.0
        return Pl, 0.0, el
    end
   
    # This factor is independent of (T, μ); cache it to avoid per-call besseli allocations.
    A   = eos.canon_factor*g * m^2 / (2π^2)
    K2x = SpecialFunctions.besselkx(2, x)

    z   = (μ - m) / T
    z   = clamp(z, -700.0, 700.0)
    Efac = exp(z)

    n_hq = A * T   * Efac * K2x
    # P_hq and e_hq are currently disabled; avoid computing besselkx(1, x) unless re-enabled.
    P_hq = 0.0
    e_hq = 0.0

    if eos.hq_times_fmGeV3
        n_hq *= fmGeV3
        P_hq *= fmGeV3
        e_hq *= fmGeV3
    end

    return Pl + P_hq, n_hq, el + e_hq
end
