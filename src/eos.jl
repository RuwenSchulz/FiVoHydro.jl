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
    K2x = safe_besselkx(2, x)
    K1x = safe_besselkx(1, x)

    exp_arg = (μ - m) / T
    Efac    = exp(exp_arg)

    n_hq = A * T   * Efac * K2x
    P_hq = A * T^2 * Efac * K2x
    e_hq = A * T^2 * Efac * (3.0*K2x + x*K1x)

    # NOTE: if your light sector is in fm units via fmGeV3, check if you also want:
    # n_hq *= fmGeV3; P_hq *= fmGeV3; e_hq *= fmGeV3

    return Plight + P_hq, n_hq, elight + e_hq
end


# -------------------------
# Running conformal light + HQ Boltzmann
# -------------------------
Base.@kwdef struct RunningConformalHQEOS
    g_eff::Float64 = 40.0
    δ::Float64     = 1e-5   # deformation strength
    T0::Float64    = 0.2   # GeV

    m_hq::Float64  = 1.5
    g_hq::Float64  = 6.0
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
    K2x = safe_besselkx(2, x)
    K1x = safe_besselkx(1, x)

    exp_arg = (μ - m) / T
    Efac    = exp(exp_arg)

    n_hq = A * T   * Efac * K2x
    P_hq = A * T^2 * Efac * K2x
    e_hq = A * T^2 * Efac * (3.0*K2x + x*K1x)

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
    α = μ / max(Tuse, 1e-50)

    if (Tuse < eos.Tmin) || (Tuse > eos.Tmax) || (α < eos.αmin) || (α > eos.αmax)
        return eos_Pne(Tuse, μ, eos.base)   # <-- robust fallback
    end

    return (eos.itpP(Tuse, α), eos.itpn(Tuse, α), eos.itpe(Tuse, α))
end


# -------------------------
# Generic thermodynamic helper (OK to keep generic)
# -------------------------
@inline function eos_entropy(T::Float64, μ::Float64, n::Float64, e::Float64, P::Float64)
    T = max(T, T_MIN)
    s = (e + P - μ*n) / T
    return max(s, 0.0)
end
