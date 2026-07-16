# =============================================================================
# FiVo-local second-moment builder.
#
# The copied Fluidum algebra is kept as `_build_IS2_system_legacy_basis!`, but
# FiVo evolves the physical variables (piQr, piQperp, PiQ). The legacy matrix is
# written in the transformed basis (-piQr, -piQperp / r^4, -PiQ), so the public
# `build_IS2_system!` maps into that basis and transforms the derivative columns
# back to the physical variables.
# =============================================================================

function _build_IS2_system_legacy_basis!(
    At::Matrix{Float64}, Ax::Matrix{Float64}, src::Vector{Float64},
    U::NTuple{5,Float64},
    τ::Float64, r::Float64,
    ur::Float64, T::Float64, dtT::Float64, drT::Float64, drur::Float64, dtur::Float64,
    n::Float64, dn_dα::Float64, dn_dT::Float64,
    κ::Float64, τn::Float64, τM::Float64, ηM::Float64, cM::Float64,
    dmn_eps::Float64 = 1e-6,
)
    α_val, νr, πQr, πQperp, PiQ = U

    uτ   = sqrt(1.0 + ur^2)
    uτ2  = 1.0 + ur^2
    uτ15 = uτ * uτ2
    r4   = r^4
    t4   = τ^4
    r3   = r^3
    t3   = τ^3

    dn_da = dn_dα + dmn_eps

    fill!(At, 0.0)
    At[1,1] = uτ * dn_da
    At[2,1] = κ * ur * uτ
    At[1,2] = ur / uτ
    At[2,2] = uτ * τn
    At[3,2] = ηM * ur * (-6 + r4 + t4) / (3.0 * uτ)
    At[4,2] = ηM * ur * ( 3 - 2*r4 + t4) / (3.0 * uτ)
    At[5,2] = -ηM * ur * (3 + r4 + t4) / (3.0 * uτ)
    At[2,3] = cM * ur * uτ
    At[3,3] = uτ * τM * (2 + t4) / 3.0
    At[4,3] = uτ * τM * (-1 + t4) / 3.0
    At[5,3] = -uτ * τM * (-1 + t4) / 3.0
    At[3,4] = -uτ * τM * (r4 - t4) / 3.0
    At[4,4] = uτ * τM * (2*r4 + t4) / 3.0
    At[5,4] = uτ * τM * (r4 - t4) / 3.0
    At[2,5] = cM * ur * uτ
    At[3,5] = -uτ * τM * (-2 + r4 + t4) / 3.0
    At[4,5] = uτ * τM * (-1 + 2*r4 - t4) / 3.0
    At[5,5] = uτ * τM * (1 + r4 + t4) / 3.0

    fill!(Ax, 0.0)
    Ax[1,1] = ur * dn_da
    Ax[2,1] = κ * uτ2
    Ax[1,2] = 1.0
    Ax[2,2] = ur * τn
    Ax[3,2] = ηM * (-6 + r4 + t4) / 3.0
    Ax[4,2] = ηM * ( 3 - 2*r4 + t4) / 3.0
    Ax[5,2] = -ηM * (3 + r4 + t4) / 3.0
    Ax[2,3] = cM * uτ2
    Ax[3,3] = ur * τM * (2 + t4) / 3.0
    Ax[4,3] = ur * τM * (-1 + t4) / 3.0
    Ax[5,3] = -ur * τM * (-1 + t4) / 3.0
    Ax[3,4] = ur * τM * (-r4 + t4) / 3.0
    Ax[4,4] = ur * τM * (2*r4 + t4) / 3.0
    Ax[5,4] = ur * τM * (r4 - t4) / 3.0
    Ax[2,5] = cM * uτ2
    Ax[3,5] = -ur * τM * (-2 + r4 + t4) / 3.0
    Ax[4,5] = ur * τM * (-1 + 2*r4 - t4) / 3.0
    Ax[5,5] = ur * τM * (1 + r4 + t4) / 3.0

    src[1] = n * (drur + dtur*ur/uτ + ur/r + uτ/τ) +
             νr * (1.0/r + (ur + ur^3 + dtur*τ) / (uτ15*τ)) +
             (drT*ur + dtT*uτ) * dn_dT

    src[2] = (1.0 - dtur*ur*τn/uτ + drur*(-1.0 + 1.0/uτ2)*τn) * νr +
             cM * (drur*ur + dtur*uτ) * (πQr + PiQ)

    src[3] = (
        3.0*uτ2*(
            2*uτ*πQr
            + 2*ur*uτ*τM*r3*πQperp - uτ*r4*πQperp
            + 2*uτ*PiQ
            + 2*ur*uτ*τM*r3*PiQ - uτ*r4*PiQ
            - 2*uτ2*τM*(πQr + πQperp - PiQ)*t3
            + uτ*πQr*t4 + uτ*πQperp*t4 - uτ*PiQ*t4
        ) + ηM*νr*(
            5*dtur*(-2 + r4 + t4)
            + 2*dtur*ur^2*(4 + r4 + t4)
            + 2*drur*ur*uτ*(4 + r4 + t4)
        )
    ) / (9.0 * uτ15)

    src[4] = (
        -3.0*uτ2*(
            uτ*πQr
            + 4*ur*uτ*τM*r3*πQperp - 2*uτ*r4*πQperp
            + uτ*PiQ
            + 4*ur*uτ*τM*r3*PiQ - 2*uτ*r4*PiQ
            + 2*uτ2*τM*(πQr + πQperp - PiQ)*t3
            - uτ*πQr*t4 - uτ*πQperp*t4 + uτ*PiQ*t4
        ) + ηM*νr*(
            2*dtur*ur^2*(-2 - 2*r4 + t4)
            + 2*drur*ur*uτ*(-2 - 2*r4 + t4)
            + 5*dtur*(1 - 2*r4 + t4)
        )
    ) / (9.0 * uτ15)

    src[5] = (
        3.0*uτ2*(
            uτ*πQr
            - 2*ur*uτ*τM*r3*πQperp + uτ*r4*πQperp
            + uτ*PiQ
            - 2*ur*uτ*τM*r3*PiQ + uτ*r4*PiQ
            + 2*uτ2*τM*(πQr + πQperp - PiQ)*t3
            - uτ*πQr*t4 - uτ*πQperp*t4 + uτ*PiQ*t4
        ) - ηM*νr*(
            2*dtur*ur^2*(-2 + r4 + t4)
            + 2*drur*ur*uτ*(-2 + r4 + t4)
            + 5*dtur*(1 + r4 + t4)
        )
    ) / (9.0 * uτ15)

    return nothing
end

function build_IS2_system!(
    At::Matrix{Float64}, Ax::Matrix{Float64}, src::Vector{Float64},
    U::NTuple{5,Float64},
    τ::Float64, r::Float64,
    ur::Float64, T::Float64, dtT::Float64, drT::Float64, drur::Float64, dtur::Float64,
    n::Float64, dn_dα::Float64, dn_dT::Float64,
    κ::Float64, τn::Float64, τM::Float64, ηM::Float64, cM::Float64,
    dmn_eps::Float64 = 1e-6,
)
    α_val, νr, πQr, πQperp, PiQ = U

    r_safe = max(abs(r), 1e-6)
    uτ2 = 1.0 + ur^2
    uτ = sqrt(uτ2)
    uτ15 = uτ * uτ2
    r4 = r_safe^4
    r5 = r_safe^5
    t4 = τ^4

    _build_IS2_system_legacy_basis!(
        At, Ax, src,
        (α_val, νr, -πQr, -πQperp / r4, -PiQ),
        τ, r_safe,
        ur, T, dtT, drT, drur, dtur,
        n, dn_dα, dn_dT,
        κ, τn, τM, ηM, cM,
        dmn_eps,
    )

    ax14 = Ax[1,4]; ax24 = Ax[2,4]; ax34 = Ax[3,4]; ax44 = Ax[4,4]; ax54 = Ax[5,4]

    @inbounds for row in 1:5
        At[row,3] = -At[row,3]
        Ax[row,3] = -Ax[row,3]
        At[row,4] = -At[row,4] / r4
        Ax[row,4] = -Ax[row,4] / r4
        At[row,5] = -At[row,5]
        Ax[row,5] = -Ax[row,5]
    end

    # NOTE: the legacy At/Ax[3:5,2] (ν_r-derivative coupling into the second-moment
    # rows) and the legacy src[3:5] are already the faithful Fluidum values — column 2
    # (ν_r) is NOT transformed by the diagonal change of variables M=diag(1,1,-1,-1/r⁴,-1),
    # so they are kept as-is.  The only physical-basis correction is the ∂_r(−1/r⁴)
    # Jacobian source term below (Ax_legacy[:,4]·4πQperp/r⁵).  The previously hand-rolled
    # ur²-coefficient replacement and two-stage source subtraction were spurious and are
    # removed (they did not match HQ_const_BG_2nd_moment.jl).

    radial_basis_term = 4.0 * πQperp / r5
    src[1] -= ax14 * radial_basis_term
    src[2] -= ax24 * radial_basis_term
    src[3] -= ax34 * radial_basis_term
    src[4] -= ax44 * radial_basis_term
    src[5] -= ax54 * radial_basis_term

    return nothing
end