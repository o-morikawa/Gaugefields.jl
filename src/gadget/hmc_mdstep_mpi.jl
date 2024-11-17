#module HMC_MDstep_MPI_module

using Random
using LinearAlgebra

import ..AbstractGaugefields_module:
    gauss_distribution!,
    substitute_U!,
    exptU!,
    mul!,
    Traceless_antihermitian_add!,
    get_myrank
import ..GaugeAction_module: evaluate_GaugeAction, calc_dSdUμ!
import ..HMC_module: calc_action, U_update!, P_update!, Flux_update!
import ..Abstractsmearing_module: calc_smearedU, back_prop
import ..Temporalfields_module: Temporalfields, get_temp, unused!

function MDstep_core_mpi!(gauge_action, U, p, MDsteps, Dim, Uold, temps; displayon=true)
    Δτ = 1.0 / MDsteps
    gauss_distribution!(p)
    Sold = calc_action(gauge_action, U, p)
    substitute_U!(Uold, U)

    for itrj = 1:MDsteps
        U_update!(U, p, 0.5, Δτ, Dim, gauge_action, temps)

        P_update!(U, p, 1.0, Δτ, Dim, gauge_action, temps)

        U_update!(U, p, 0.5, Δτ, Dim, gauge_action, temps)
    end
    Snew = calc_action(gauge_action, U, p)
    if get_myrank(U)==0 && displayon
        println("Sold = $Sold, Snew = $Snew")
        println("Snew - Sold = $(Snew-Sold)")
    end
    ratio = min(1, exp(-Snew + Sold))
    r = rand()
    r = MPI.bcast(r, 0, MPI.COMM_WORLD)
    if r > ratio
        substitute_U!(U, Uold)
        return false
    else
        return true
    end
end
function MDstep_core_mpi!(gauge_action, U, B, p, MDsteps, Dim, Uold, temps; displayon=true)
    Δτ = 1.0 / MDsteps
    gauss_distribution!(p)
    Sold = calc_action(gauge_action, U, B, p)
    substitute_U!(Uold, U)

    for itrj = 1:MDsteps
        U_update!(U,    p, 0.5, Δτ, Dim, gauge_action, temps)

        P_update!(U, B, p, 1.0, Δτ, Dim, gauge_action, temps)

        U_update!(U,    p, 0.5, Δτ, Dim, gauge_action, temps)
    end
    Snew = calc_action(gauge_action, U, B, p)
    if get_myrank(U)==0 && displayon
        println("Sold = $Sold, Snew = $Snew")
        println("Snew - Sold = $(Snew-Sold)")
    end
    ratio = min(1, exp(-Snew + Sold))
    r = rand()
    r = MPI.bcast(r, 0, MPI.COMM_WORLD)
    if r > ratio
        substitute_U!(U, Uold)
        return false
    else
        return true
    end
end



function MDstep_dynB_mpi!(
    gauge_action,
    U,
    B,
    flux,
    p,
    MDsteps, # MDsteps should be an even integer
    Dim,
    Uold,
    Bold,
    flux_old,
    temps
) # Halfway-updating HMC
    Δτ = 1.0/MDsteps
    gauss_distribution!(p)

    Sold = calc_action(gauge_action,U,B,p)

    substitute_U!(Uold,U)
    substitute_U!(Bold,B)
    flux_old[:] = flux[:]

    for itrj=1:MDsteps
        U_update!(U,  p,0.5,Δτ,Dim,gauge_action,temps)

        P_update!(U,B,p,1.0,Δτ,Dim,gauge_action,temps)

        U_update!(U,  p,0.5,Δτ,Dim,gauge_action,temps)

        if itrj == Int(MDsteps/2)
            Flux_update!(B,flux)
        end
    end

    Snew = calc_action(gauge_action,U,B,p)
    ratio = min(1,exp(-Snew+Sold))
    r = rand()
    r = MPI.bcast(r, 0, MPI.COMM_WORLD)
    if rand() > r
        if get_myrank(U)==0
            println("rejected! flux = ", flux_old)
        end
        substitute_U!(U,Uold)
        substitute_U!(B,Bold)
        flux[:] = flux_old[:]
        return false
    else
        if get_myrank(U)==0
            println("accepted! flux_old = ", flux_old, " -> flux_new = ", flux)
        end
        return true
    end
end

function MDstep_dynB_mpi!(
    gauge_action,
    U,
    B,
    flux,
    p,
    MDsteps,
    num_HMC,
    Dim,
    Uold1,
    Uold2,
    Bold,
    flux_old,
    temps
) # Double-tesing HMC
    p0 = initialize_TA_Gaugefields(U)
    Sold = calc_action(gauge_action,U,B,p0)

    substitute_U!(Uold1,U)
    substitute_U!(Bold, B)
    flux_old[:] = flux[:]

    Flux_update!(B,flux)

    for ihmc=1:num_HMC
        MDstep!(gauge_action,U,B,p,MDsteps,Dim,Uold2,temps)
    end

    Snew = calc_action(gauge_action,U,B,p0)
    ratio = min(1,exp(-Snew+Sold))
    r = rand()
    r = MPI.bcast(r, 0, MPI.COMM_WORLD)
    if rand() > r
        if get_myrank(U)==0
            println("rejected! flux = ", flux_old)
        end
        substitute_U!(U,Uold1)
        substitute_U!(B,Bold)
        flux[:] = flux_old[:]
        return false
    else
        if get_myrank(U)==0
            println("accepted! flux_old = ", flux_old, " -> flux_new = ", flux)
        end
        return true
    end
end

function MDstep_stout_mpi!(gauge_action, U, p, MDsteps, Dim, Uold, nn, dSdU, temps;
                           displayon=true)
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

    if get_myrank(U)==0 && displayon
        println("Sold = $Sold, Snew = $Snew")
        println("Snew - Sold = $(Snew-Sold)")
    end

    r = rand()
    r = MPI.bcast(r, 0, MPI.COMM_WORLD)
    accept = exp(Sold - Snew) >= r

    if accept != true
        substitute_U!(U, Uold)
        return false
    else
        return true
    end

end


#end
