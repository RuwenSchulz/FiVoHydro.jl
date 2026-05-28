# ------------------------------------------------------------
# Reconstruction in (logT, φ, y)
# ------------------------------------------------------------
function reconstruct_muscl_prims!(UL, UR, σ, yT, φ, y, grid; limiter=mc_limiter)
    Ntot = length(yT)
    ng = grid.nghost
    fill!(σ, 0.0)

    @inbounds for i in (ng+2):(Ntot-ng-1)
        σ[1,i] = limiter(yT[i]-yT[i-1], yT[i+1]-yT[i])
        σ[2,i] = limiter(φ[i]-φ[i-1],   φ[i+1]-φ[i])
        σ[3,i] = limiter(y[i]-y[i-1],   y[i+1]-y[i])
    end

    @inbounds for i in 1:(Ntot-1)
        if i <= (ng+1) || i >= (Ntot-ng-1)
            UL[1,i] = yT[i];   UL[2,i] = φ[i];   UL[3,i] = y[i]
            UR[1,i] = yT[i+1]; UR[2,i] = φ[i+1]; UR[3,i] = y[i+1]
        else
            UL[1,i] = yT[i]   + 0.5*σ[1,i]
            UL[2,i] = φ[i]    + 0.5*σ[2,i]
            UL[3,i] = y[i]    + 0.5*σ[3,i]

            UR[1,i] = yT[i+1] - 0.5*σ[1,i+1]
            UR[2,i] = φ[i+1]  - 0.5*σ[2,i+1]
            UR[3,i] = y[i+1]  - 0.5*σ[3,i+1]

            # Monotonicity-preserving clamp: interface extrapolations must stay within
            # local neighbor bounds, otherwise MUSCL can introduce wiggles that remain
            # "admissible" but are physically/visually wrong.
            yTminL = min(yT[i-1], yT[i], yT[i+1]); yTmaxL = max(yT[i-1], yT[i], yT[i+1])
            φminL  = min(φ[i-1],  φ[i],  φ[i+1]);  φmaxL  = max(φ[i-1],  φ[i],  φ[i+1])
            yminL  = min(y[i-1],  y[i],  y[i+1]);  ymaxL  = max(y[i-1],  y[i],  y[i+1])

            yTminR = min(yT[i], yT[i+1], yT[i+2]); yTmaxR = max(yT[i], yT[i+1], yT[i+2])
            φminR  = min(φ[i],  φ[i+1],  φ[i+2]);  φmaxR  = max(φ[i],  φ[i+1],  φ[i+2])
            yminR  = min(y[i],  y[i+1],  y[i+2]);  ymaxR  = max(y[i],  y[i+1],  y[i+2])

            UL[1,i] = clamp(UL[1,i], yTminL, yTmaxL)
            UL[2,i] = clamp(UL[2,i], φminL,  φmaxL)
            UL[3,i] = clamp(UL[3,i], yminL,  ymaxL)

            UR[1,i] = clamp(UR[1,i], yTminR, yTmaxR)
            UR[2,i] = clamp(UR[2,i], φminR,  φmaxR)
            UR[3,i] = clamp(UR[3,i], yminR,  ymaxR)
        end
    end
    return nothing
end
