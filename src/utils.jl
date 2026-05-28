# =========================
# src/utils.jl   (STEP 2)
# =========================

@inline safe_sqrt(x) = x <= 0 ? 0.0 : sqrt(x)
@inline clamp(x, a, b) = x < a ? a : (x > b ? b : x)
@inline isfinite3(a, b, c) = isfinite(a) & isfinite(b) & isfinite(c)

#@inline safe_exp(x::Float64) = exp(clamp(x, LOG_MIN, LOG_MAX))

using SpecialFunctions

# --- slope limiter (MC) + minmod ---
@inline function mc_limiter(a, b)
    if sign(a) != sign(b)
        return 0.0
    end
    s  = sign(a)
    aa = abs(a)
    bb = abs(b)
    return s * min(2aa, min(2bb, 0.5 * (aa + bb)))
end

@inline minmod(a, b) = (sign(a) == sign(b)) ? sign(a) * min(abs(a), abs(b)) : 0.0

# --- smooth taper helper ---
@inline function smoothstep01_5(x)
    x <= 0 ? 0.0 : (x >= 1 ? 1.0 : x^3 * (10 - 15x + 6x^2))
end

# --- robust 3x3 linear solver (Gaussian elimination with pivoting + scale-aware pivot tol) ---
@inline function solve3x3_gauss!(δ::Vector{Float64}, J::Matrix{Float64}, F::Vector{Float64}, A::Matrix{Float64})
    @inbounds begin
        # augmented matrix A = [J | -F]
        A[1,1]=J[1,1]; A[1,2]=J[1,2]; A[1,3]=J[1,3]; A[1,4]=-F[1]
        A[2,1]=J[2,1]; A[2,2]=J[2,2]; A[2,3]=J[2,3]; A[2,4]=-F[2]
        A[3,1]=J[3,1]; A[3,2]=J[3,2]; A[3,3]=J[3,3]; A[3,4]=-F[3]

        # scale-aware pivot tolerance: reject pivots tiny compared to matrix scale
        smax = 0.0
        for r in 1:3, c in 1:3
            smax = max(smax, abs(A[r,c]))
        end
        # if J is basically zero, bail
        if !(isfinite(smax)) || smax == 0.0
            return false
        end
        pivtol = 1e-15 * smax   # conservative; adjust to 1e-12 if needed

        # --- pivot in column 1 ---
        p1 = 1
        a11 = abs(A[1,1]); a21 = abs(A[2,1]); a31 = abs(A[3,1])
        if a21 > a11; p1 = 2; a11 = a21; end
        if a31 > a11; p1 = 3; a11 = a31; end
        if a11 < pivtol
            return false
        end
        if p1 != 1
            for k in 1:4
                A[1,k], A[p1,k] = A[p1,k], A[1,k]
            end
        end

        # eliminate rows 2,3
        for r in 2:3
            m = A[r,1] / A[1,1]
            A[r,1] = 0.0
            A[r,2] -= m*A[1,2]
            A[r,3] -= m*A[1,3]
            A[r,4] -= m*A[1,4]
        end

        # --- pivot in column 2 (rows 2..3) ---
        p2 = (abs(A[2,2]) >= abs(A[3,2])) ? 2 : 3
        if abs(A[p2,2]) < pivtol
            return false
        end
        if p2 != 2
            for k in 2:4
                A[2,k], A[p2,k] = A[p2,k], A[2,k]
            end
        end

        # eliminate row 3
        m = A[3,2] / A[2,2]
        A[3,2] = 0.0
        A[3,3] -= m*A[2,3]
        A[3,4] -= m*A[2,4]

        if abs(A[3,3]) < pivtol
            return false
        end

        # back-substitution
        δ3 = A[3,4] / A[3,3]
        δ2 = (A[2,4] - A[2,3]*δ3) / A[2,2]
        δ1 = (A[1,4] - A[1,2]*δ2 - A[1,3]*δ3) / A[1,1]

        δ[1]=δ1; δ[2]=δ2; δ[3]=δ3
    end

    return all(isfinite, δ)
end


@inline function _check_transport_flags!(;
    advect_nur::Bool, relax_advect_nur::Bool,
    advect_Pi::Bool,  relax_advect_Pi::Bool,
    advect_pi::Bool,  relax_advect_pi::Bool
)
    if advect_nur && relax_advect_nur
        throw(ArgumentError("Pick ONE for nur transport: advect_nur OR relax_advect_nur (not both)."))
    end
    if advect_Pi && relax_advect_Pi
        throw(ArgumentError("Pick ONE for Pi transport: advect_Pi OR relax_advect_Pi (not both)."))
    end
    if advect_pi && relax_advect_pi
        throw(ArgumentError("Pick ONE for pi transport: advect_pi OR relax_advect_pi (not both)."))
    end
    return nothing
end
