#module HMC_MDstep_module

using Random
using LinearAlgebra

import ..AbstractGaugefields_module:
    AbstractGaugefields,
    gauss_distribution!,
    substitute_U!,
    exptU!,
    mul!,
    Traceless_antihermitian_add!
import ..GaugeAction_module: GaugeAction, evaluate_GaugeAction, calc_dSdUμ!
import ..HMC_module: calc_action, U_update!, P_update!, Flux_update!
import ..Abstractsmearing_module: calc_smearedU, back_prop
import ..Temporalfields_module: Temporalfields, get_temp, unused!

function MDstep_core!(
    gauge_action::GaugeAction,
    U::Array{T,1},
    p, MDsteps, Dim,
    Uold::Array{T,1},
    temps::Temporalfields;
    displayon=true,
    τ = 1.0,
) where {T<:AbstractGaugefields}
    tau = τ
    Δτ = tau / MDsteps
    gauss_distribution!(p)
    Sold = calc_action(gauge_action, U, p)
    substitute_U!(Uold, U)

    for itrj = 1:MDsteps
        U_update!(U, p, 0.5*tau, Δτ, Dim, gauge_action, temps)

        P_update!(U, p, 1.0*tau, Δτ, Dim, gauge_action, temps)

        U_update!(U, p, 0.5*tau, Δτ, Dim, gauge_action, temps)
    end
    Snew = calc_action(gauge_action, U, p)
    if displayon
        println("Sold = $Sold, Snew = $Snew")
        println("Snew - Sold = $(Snew-Sold)")
    end
    ratio = min(1, exp(-Snew + Sold))
    if rand() > ratio
        substitute_U!(U, Uold)
        return false
    else
        return true
    end
end
function MDstep_core!(
    gauge_action::GaugeAction,
    U::Array{T,1},
    p, MDsteps, Dim,
    Uold::Array{T,1};
    displayon=true,
    τ = 1.0,
) where {T<:AbstractGaugefields}
    tau = τ
    Δτ = tau / MDsteps
    gauss_distribution!(p)
    Sold = calc_action(gauge_action, U, p)
    substitute_U!(Uold, U)

    for itrj = 1:MDsteps
        U_update!(U, p, 0.5*tau, Δτ, Dim, gauge_action)

        P_update!(U, p, 1.0*tau, Δτ, Dim, gauge_action)

        U_update!(U, p, 0.5*tau, Δτ, Dim, gauge_action)
    end
    Snew = calc_action(gauge_action, U, p)
    if displayon
        println("Sold = $Sold, Snew = $Snew")
        println("Snew - Sold = $(Snew-Sold)")
    end
    ratio = min(1, exp(-Snew + Sold))
    if rand() > ratio
        substitute_U!(U, Uold)
        return false
    else
        return true
    end
end

function MDstep_core!(
    gauge_action::GaugeAction,
    U::Array{T,1},
    B::Array{T,2},
    p, MDsteps, Dim,
    Uold::Array{T,1},
    temps::Temporalfields;
    displayon=true,
    τ = 1.0,
) where {T<:AbstractGaugefields}
    tau = τ
    Δτ = tau / MDsteps
    gauss_distribution!(p)
    Sold = calc_action(gauge_action, U, B, p)
    substitute_U!(Uold, U)

    for itrj = 1:MDsteps
        U_update!(U,    p, 0.5*tau, Δτ, Dim, gauge_action, temps)

        P_update!(U, B, p, 1.0*tau, Δτ, Dim, gauge_action, temps)

        U_update!(U,    p, 0.5*tau, Δτ, Dim, gauge_action, temps)
    end
    Snew = calc_action(gauge_action, U, B, p)
    if displayon
        println("Sold = $Sold, Snew = $Snew")
        println("Snew - Sold = $(Snew-Sold)")
    end
    ratio = min(1, exp(-Snew + Sold))
    if rand() > ratio
        substitute_U!(U, Uold)
        return false
    else
        return true
    end
end
function MDstep_core!(
    gauge_action::GaugeAction,
    U::Array{T,1},
    B::Array{T,2},
    p, MDsteps, Dim,
    Uold::Array{T,1};
    displayon=true,
    τ = 1.0,
) where {T<:AbstractGaugefields}
    tau = τ
    Δτ = tau / MDsteps
    gauss_distribution!(p)
    Sold = calc_action(gauge_action, U, B, p)
    substitute_U!(Uold, U)

    for itrj = 1:MDsteps
        U_update!(U,    p, 0.5*tau, Δτ, Dim, gauge_action)

        P_update!(U, B, p, 1.0*tau, Δτ, Dim, gauge_action)

        U_update!(U,    p, 0.5*tau, Δτ, Dim, gauge_action)
    end
    Snew = calc_action(gauge_action, U, B, p)
    if displayon
        println("Sold = $Sold, Snew = $Snew")
        println("Snew - Sold = $(Snew-Sold)")
    end
    ratio = min(1, exp(-Snew + Sold))
    if rand() > ratio
        substitute_U!(U, Uold)
        return false
    else
        return true
    end
end


# Halfway-updating HMC
function MDstep_dynB!(
    gauge_action::GaugeAction,
    U::Array{T,1},
    B::Array{T,2},
    flux,
    p,
    MDsteps, # MDsteps should be an even integer
    Dim,
    Uold::Array{T,1},
    Bold::Array{T,2},
    flux_old,
    temps::Temporalfields;
    displayon=true,
    τ = 1.0,
    flux_cond="randall",
    γ = 1.0,
) where {T<:AbstractGaugefields}
    tau = τ
    Δτ = tau / MDsteps
    gauss_distribution!(p)

    Sold = calc_action(gauge_action,U,B,p)

    substitute_U!(Uold,U)
    substitute_U!(Bold,B)
    flux_old[:] = flux[:]

    for itrj=1:MDsteps
        U_update!(U,  p,0.5*tau,Δτ,Dim,gauge_action,temps)

        P_update!(U,B,p,1.0*tau,Δτ,Dim,gauge_action,temps)

        U_update!(U,  p,0.5*tau,Δτ,Dim,gauge_action,temps)

        if itrj == Int(MDsteps/2)
            Flux_update!(B,flux,condition=flux_cond,γ=γ)
        end
    end

    Snew = calc_action(gauge_action,U,B,p)
    if displayon
        println("Sold = $Sold, Snew = $Snew")
        println("Snew - Sold = $(Snew-Sold)")
    end
    ratio = min(1,exp(-Snew+Sold))
    if rand() > ratio
        println("rejected! flux = ", flux_old)
        substitute_U!(U,Uold)
        substitute_U!(B,Bold)
        flux[:] = flux_old[:]
        return false
    else
        println("accepted! flux_old = ", flux_old, " -> flux_new = ", flux)
        return true
    end
end
function MDstep_dynB!(
    gauge_action::GaugeAction,
    U::Array{T,1},
    B::Array{T,2},
    flux,
    p,
    MDsteps, # MDsteps should be an even integer
    Dim,
    Uold::Array{T,1},
    Bold::Array{T,2},
    flux_old;
    displayon=true,
    τ = 1.0,
    flux_cond="randall",
    γ = 1.0,
) where {T<:AbstractGaugefields}
    tau = τ
    Δτ = tau / MDsteps
    gauss_distribution!(p)

    Sold = calc_action(gauge_action,U,B,p)

    substitute_U!(Uold,U)
    substitute_U!(Bold,B)
    flux_old[:] = flux[:]

    for itrj=1:MDsteps
        U_update!(U,  p,0.5*tau,Δτ,Dim,gauge_action)

        P_update!(U,B,p,1.0*tau,Δτ,Dim,gauge_action)

        U_update!(U,  p,0.5*tau,Δτ,Dim,gauge_action)

        if itrj == Int(MDsteps/2)
            Flux_update!(B,flux,condition=flux_cond,γ=γ)
        end
    end

    Snew = calc_action(gauge_action,U,B,p)
    if displayon
        println("Sold = $Sold, Snew = $Snew")
        println("Snew - Sold = $(Snew-Sold)")
    end
    ratio = min(1,exp(-Snew+Sold))
    if rand() > ratio
        println("rejected! flux = ", flux_old)
        substitute_U!(U,Uold)
        substitute_U!(B,Bold)
        flux[:] = flux_old[:]
        return false
    else
        println("accepted! flux_old = ", flux_old, " -> flux_new = ", flux)
        return true
    end
end
## Halfway-updating HMC with random B fields
function MDstep_dynB!(
    gauge_action::GaugeAction,
    U::Array{T,1},
    B::Array{T,2},
    flux,
    p,
    MDsteps, # MDsteps should be an even integer
    Dim,
    Uold::Array{T,1},
    Bold::Array{T,2},
    Btemp::Array{T,2},
    flux_old,
    temps::Temporalfields,
    numtransf;
    displayon=true,
    τ = 1.0,
) where {T<:AbstractGaugefields}
    tau = τ
    Δτ = tau / MDsteps
    gauss_distribution!(p)

    Sold = calc_action(gauge_action,U,B,p)

    substitute_U!(Uold,U)
    substitute_U!(Bold,B)
    flux_old[:] = flux[:]

    for itrj=1:MDsteps
        U_update!(U,  p,0.5*tau,Δτ,Dim,gauge_action,temps)

        P_update!(U,B,p,1.0*tau,Δτ,Dim,gauge_action,temps)

        U_update!(U,  p,0.5*tau,Δτ,Dim,gauge_action,temps)

        if itrj == Int(MDsteps/2)
            Flux_update!(B,Btemp,flux,temps,numtransf=numtransf)
        end
    end

    Snew = calc_action(gauge_action,U,B,p)
    if displayon
        println("Sold = $Sold, Snew = $Snew")
        println("Snew - Sold = $(Snew-Sold)")
    end
    ratio = min(1,exp(-Snew+Sold))
    if rand() > ratio
        println("rejected! flux = ", flux_old)
        substitute_U!(U,Uold)
        substitute_U!(B,Bold)
        flux[:] = flux_old[:]
        return false
    else
        println("accepted! flux_old = ", flux_old, " -> flux_new = ", flux)
        return true
    end
end
function MDstep_dynB!(
    gauge_action::GaugeAction,
    U::Array{T,1},
    B::Array{T,2},
    flux,
    p,
    MDsteps, # MDsteps should be an even integer
    Dim,
    Uold::Array{T,1},
    Bold::Array{T,2},
    Btemp::Array{T,2},
    flux_old,
    numtransf;
    displayon=true,
    τ = 1.0,
) where {T<:AbstractGaugefields}
    tau = τ
    Δτ = tau / MDsteps
    gauss_distribution!(p)

    Sold = calc_action(gauge_action,U,B,p)

    substitute_U!(Uold,U)
    substitute_U!(Bold,B)
    flux_old[:] = flux[:]

    for itrj=1:MDsteps
        U_update!(U,  p,0.5*tau,Δτ,Dim,gauge_action)

        P_update!(U,B,p,1.0*tau,Δτ,Dim,gauge_action)

        U_update!(U,  p,0.5*tau,Δτ,Dim,gauge_action)

        if itrj == Int(MDsteps/2)
            Flux_update!(B,Btemp,flux,gauge_action,numtransf=numtransf)
        end
    end

    Snew = calc_action(gauge_action,U,B,p)
    if displayon
        println("Sold = $Sold, Snew = $Snew")
        println("Snew - Sold = $(Snew-Sold)")
    end
    ratio = min(1,exp(-Snew+Sold))
    if rand() > ratio
        println("rejected! flux = ", flux_old)
        substitute_U!(U,Uold)
        substitute_U!(B,Bold)
        flux[:] = flux_old[:]
        return false
    else
        println("accepted! flux_old = ", flux_old, " -> flux_new = ", flux)
        return true
    end
end

# Double-tesing HMC
function MDstep_dynB!(
    gauge_action::GaugeAction,
    U::Array{T,1},
    B::Array{T,2},
    flux,
    p,
    MDsteps,
    num_HMC,
    Dim,
    Uold1::Array{T,1},
    Uold2::Array{T,1},
    Bold::Array{T,2},
    flux_old,
    temps::Temporalfields;
    displayon=true
) where {T<:AbstractGaugefields}
    p0 = initialize_TA_Gaugefields(U)
    Sold = calc_action(gauge_action,U,B,p0)

    substitute_U!(Uold1,U)
    substitute_U!(Bold, B)
    flux_old[:] = flux[:]

    Flux_update!(B,flux)

    for ihmc=1:num_HMC
        MDstep_core!(gauge_action,U,B,p,MDsteps,Dim,Uold2,temps,displayon=false)
    end

    Snew = calc_action(gauge_action,U,B,p0)
    if displayon
        println("Sold = $Sold, Snew = $Snew")
        println("Snew - Sold = $(Snew-Sold)")
    end    
    ratio = min(1,exp(-Snew+Sold))
    if rand() > ratio
        println("rejected! flux = ", flux_old)
        substitute_U!(U,Uold1)
        substitute_U!(B,Bold)
        flux[:] = flux_old[:]
        return false
    else
        println("accepted! flux_old = ", flux_old, " -> flux_new = ", flux)
        return true
    end
end
function MDstep_dynB!(
    gauge_action::GaugeAction,
    U::Array{T,1},
    B::Array{T,2},
    flux,
    p,
    MDsteps,
    num_HMC,
    Dim,
    Uold1::Array{T,1},
    Uold2::Array{T,1},
    Bold::Array{T,2},
    flux_old;
    displayon=true
) where {T<:AbstractGaugefields}
    p0 = initialize_TA_Gaugefields(U)
    Sold = calc_action(gauge_action,U,B,p0)

    substitute_U!(Uold1,U)
    substitute_U!(Bold, B)
    flux_old[:] = flux[:]

    Flux_update!(B,flux)

    for ihmc=1:num_HMC
        MDstep_core!(gauge_action,U,B,p,MDsteps,Dim,Uold2,displayon=false)
    end

    Snew = calc_action(gauge_action,U,B,p0)
    if displayon
        println("Sold = $Sold, Snew = $Snew")
        println("Snew - Sold = $(Snew-Sold)")
    end
    ratio = min(1,exp(-Snew+Sold))
    if rand() > ratio
        println("rejected! flux = ", flux_old)
        substitute_U!(U,Uold1)
        substitute_U!(B,Bold)
        flux[:] = flux_old[:]
        return false
    else
        println("accepted! flux_old = ", flux_old, " -> flux_new = ", flux)
        return true
    end
end

function MDstep_stout!(
    gauge_action::GaugeAction,
    U::Array{T,1},
    p, MDsteps, Dim,
    Uold::Array{T,1},
    nn, dSdU,
    temps::Temporalfields;
    displayon=true
) where {T<:AbstractGaugefields}
    Δτ = 1.0 / MDsteps
    gauss_distribution!(p)

    Uout, Uout_multi, _ = calc_smearedU(U, nn)
    Sold = calc_action(gauge_action, Uout, p)

    substitute_U!(Uold, U)

    for itrj = 1:MDsteps
        U_update!(U, p, 0.5, Δτ, Dim, gauge_action, temps)

        P_update!(U, p, 1.0, Δτ, Dim, gauge_action, dSdU, nn, temps)

        U_update!(U, p, 0.5, Δτ, Dim, gauge_action, temps)
    end

    Uout, Uout_multi, _ = calc_smearedU(U, nn)
    Snew = calc_action(gauge_action, Uout, p)

    if displayon
        println("Sold = $Sold, Snew = $Snew")
        println("Snew - Sold = $(Snew-Sold)")
    end

    accept = exp(Sold - Snew) >= rand()

    if accept != true #rand() > ratio
        substitute_U!(U, Uold)
        return false
    else
        return true
    end

end
function MDstep_stout!(
    gauge_action::GaugeAction,
    U::Array{T,1},
    p, MDsteps, Dim,
    Uold::Array{T,1},
    nn, dSdU;
    displayon=true
) where {T<:AbstractGaugefields}
    Δτ = 1.0 / MDsteps
    gauss_distribution!(p)

    Uout, Uout_multi, _ = calc_smearedU(U, nn)
    Sold = calc_action(gauge_action, Uout, p)

    substitute_U!(Uold, U)

    for itrj = 1:MDsteps
        U_update!(U, p, 0.5, Δτ, Dim, gauge_action)

        P_update!(U, p, 1.0, Δτ, Dim, gauge_action, dSdU, nn)

        U_update!(U, p, 0.5, Δτ, Dim, gauge_action)
    end

    Uout, Uout_multi, _ = calc_smearedU(U, nn)
    Snew = calc_action(gauge_action, Uout, p)

    if displayon
        println("Sold = $Sold, Snew = $Snew")
        println("Snew - Sold = $(Snew-Sold)")
    end

    accept = exp(Sold - Snew) >= rand()

    if accept != true #rand() > ratio
        substitute_U!(U, Uold)
        return false
    else
        return true
    end

end

#end
