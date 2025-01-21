#module HMC_MDstep_MPI_module

using Random
using LinearAlgebra

import ..AbstractGaugefields_module:
    Initialize_Bfields,
    B_TfluxGauges,
    thooftFlux_4D_B_at_bndry_nowing_mpi,
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

function MDstep_core_mpi!(
    gauge_action::GaugeAction,
    U::Array{T,1},
    p, MDsteps, Dim,
    Uold::Array{T,1},
    temps::Temporalfields;
    displayon=true
) where {T<:AbstractGaugefields}
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
function MDstep_core_mpi!(
    gauge_action::GaugeAction,
    U::Array{T,1},
    p, MDsteps, Dim,
    Uold::Array{T,1};
    displayon=true
) where {T<:AbstractGaugefields}
    Δτ = 1.0 / MDsteps
    gauss_distribution!(p)
    Sold = calc_action(gauge_action, U, p)
    substitute_U!(Uold, U)

    for itrj = 1:MDsteps
        U_update!(U, p, 0.5, Δτ, Dim, gauge_action)

        P_update!(U, p, 1.0, Δτ, Dim, gauge_action)

        U_update!(U, p, 0.5, Δτ, Dim, gauge_action)
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

function MDstep_core_mpi!(
    gauge_action::GaugeAction,
    U::Array{T,1},
    B::Array{T,2},
    p, MDsteps, Dim,
    Uold::Array{T,1},
    temps::Temporalfields;
    displayon=true
) where {T<:AbstractGaugefields}
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
function MDstep_core_mpi!(
    gauge_action::GaugeAction,
    U::Array{T,1},
    B::Array{T,2},
    p, MDsteps, Dim,
    Uold::Array{T,1};
    displayon=true
) where {T<:AbstractGaugefields}
    Δτ = 1.0 / MDsteps
    gauss_distribution!(p)
    Sold = calc_action(gauge_action, U, B, p)
    substitute_U!(Uold, U)

    for itrj = 1:MDsteps
        U_update!(U,    p, 0.5, Δτ, Dim, gauge_action)

        P_update!(U, B, p, 1.0, Δτ, Dim, gauge_action)

        U_update!(U,    p, 0.5, Δτ, Dim, gauge_action)
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
    temps::Temporalfields,
    PEs
) where {T<:AbstractGaugefields} # Halfway-updating HMC
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
            Flux_update_mpi!(B,flux,gauge_action,PEs)
        end
    end

    Snew = calc_action(gauge_action,U,B,p)
    ratio = min(1,exp(-Snew+Sold))
    r = rand()
    r = MPI.bcast(r, 0, MPI.COMM_WORLD)
    if r > ratio
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
    PEs
) where {T<:AbstractGaugefields} # Halfway-updating HMC
    Δτ = 1.0/MDsteps
    gauss_distribution!(p)

    Sold = calc_action(gauge_action,U,B,p)

    substitute_U!(Uold,U)
    substitute_U!(Bold,B)
    flux_old[:] = flux[:]

    for itrj=1:MDsteps
        U_update!(U,  p,0.5,Δτ,Dim,gauge_action)
        if get_myrank(U)==0
            println("Trj: $itrj, Flux: $flux")
            println(typeof(B))
        end
        println(get_myrank(U), B[3,4].U[1,1, PEs[1],PEs[2],PEs[3],:])
        P_update!(U,B,p,1.0,Δτ,Dim,gauge_action)

        U_update!(U,  p,0.5,Δτ,Dim,gauge_action)

        if itrj == Int(MDsteps/2)
            Flux_update_mpi!(B,flux,gauge_action,PEs)
        end
    end

    Snew = calc_action(gauge_action,U,B,p)
    ratio = min(1,exp(-Snew+Sold))
    r = rand()
    r = MPI.bcast(r, 0, MPI.COMM_WORLD)
    if r > ratio
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
    temps::Temporalfields,
    PEs
) where {T<:AbstractGaugefields} # Double-tesing HMC
    p0 = initialize_TA_Gaugefields(U)
    Sold = calc_action(gauge_action,U,B,p0)

    substitute_U!(Uold1,U)
    substitute_U!(Bold, B)
    flux_old[:] = flux[:]

    Flux_update_mpi!(B,flux,gauge_action,PEs)

    for ihmc=1:num_HMC
        MDstep!(gauge_action,U,B,p,MDsteps,Dim,Uold2,temps)
    end

    Snew = calc_action(gauge_action,U,B,p0)
    ratio = min(1,exp(-Snew+Sold))
    r = rand()
    r = MPI.bcast(r, 0, MPI.COMM_WORLD)
    if r > ratio
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
function MDstep_dynB_mpi!(
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
    PEs
) where {T<:AbstractGaugefields} # Double-tesing HMC
    p0 = initialize_TA_Gaugefields(U)
    Sold = calc_action(gauge_action,U,B,p0)

    substitute_U!(Uold1,U)
    substitute_U!(Bold, B)
    flux_old[:] = flux[:]

    Flux_update_mpi!(B,flux,gauge_action,PEs)

    for ihmc=1:num_HMC
        MDstep!(gauge_action,U,B,p,MDsteps,Dim,Uold2)
    end

    Snew = calc_action(gauge_action,U,B,p0)
    ratio = min(1,exp(-Snew+Sold))
    r = rand()
    r = MPI.bcast(r, 0, MPI.COMM_WORLD)
    if r > ratio
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

function MDstep_stout_mpi!(
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
function MDstep_stout_mpi!(
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


function Flux_update_mpi!(
    B::Array{T,2},
    flux,
    gauge_action,
    PEs,
) where {T<:AbstractGaugefields}
    Dim = 4
    
    NC  = B[1,2].NC
    NDW = B[1,2].NDW
    NX  = B[1,2].NX
    NY  = B[1,2].NY
    NZ  = B[1,2].NZ
    NT  = B[1,2].NT

    if !(NDW==0)
        error("not implemented for wing...")
    end

    temps, it_temps = get_temp(gauge_action._temp_U,2)

    i = rand(1:6)
    i = MPI.bcast(i, 0, MPI.COMM_WORLD)
    r = rand(-1:1)
    r = MPI.bcast(r, 0, MPI.COMM_WORLD)
    if get_myrank(B[1,2])==0
        println("i=$i,  r=$r")
    end
    flux[i] += r
    flux[i] %= NC
    flux[i] += (flux[i] < 0) ? NC : 0
#    flux[:] = rand(0:NC-1,6)
#    flux[:] = MPI.bcast(flux[:], 0, MPI.COMM_WORLD)


    #B = Initialize_Bfields(NC,flux,NDW,NX,NY,NZ,NT,condition = "tflux",mpi=true,PEs = PEs,mpiinit = true)

    fluxnum = i
    temps[1] = thooftFlux_4D_B_at_bndry_nowing_mpi(
        NC,
        flux[fluxnum],
        fluxnum,
        NX,NY,NZ,NT,
        PEs,
        overallminus=false,
    )
    temps[2] = thooftFlux_4D_B_at_bndry_nowing_mpi(
        NC,
        flux[fluxnum],
        fluxnum,
        NX,NY,NZ,NT,
        PEs,
        overallminus=true,
    )
    if fluxnum == 1
        substitute_U!(B[1,2], temps[1])
        substitute_U!(B[2,1], temps[2])
    elseif fluxnum == 2
        substitute_U!(B[1,3], temps[1])
        substitute_U!(B[3,1], temps[2])
    elseif fluxnum == 3
        substitute_U!(B[1,4], temps[1])
        substitute_U!(B[4,1], temps[2])
    elseif fluxnum == 4
        substitute_U!(B[2,3], temps[1])
        substitute_U!(B[3,2], temps[2])
    elseif fluxnum == 5
        substitute_U!(B[2,4], temps[1])
        substitute_U!(B[4,2], temps[2])
    elseif fluxnum == 6
        substitute_U!(B[3,4], temps[1])
        substitute_U!(B[4,3], temps[2])
    end

    unused!(gauge_action._temp_U,it_temps)
end


#end
