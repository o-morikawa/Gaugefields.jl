module HMC_core_module

using Requires

import ..AbstractGaugefields_module: AbstractGaugefields
import ..GaugeAction_module: GaugeAction
import ..HMC_module
import ..Abstractsmearing_module
import ..Temporalfields_module: Temporalfields

include("./hmc_mdstep.jl")

function __init__()
    @require MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195" begin
        include("./hmc_mdstep_mpi.jl")
    end
end

function MDstep!(
    gauge_action::GaugeAction,
    U::Array{T,1},
    p, MDsteps, Dim,
    Uold::Array{T,1},
    temps::Temporalfields;
    displayon=false, mpi=false
) where {T<:AbstractGaugefields}
    if mpi
        MDstep_core_mpi!(gauge_action, U, p, MDsteps, Dim, Uold, temps;
                         displayon=displayon)
    else
        MDstep_core!(gauge_action, U, p, MDsteps, Dim, Uold, temps;
                     displayon=displayon)
    end
end
function MDstep!(
    gauge_action::GaugeAction,
    U::Array{T,1},
    p, MDsteps, Dim,
    Uold::Array{T,1};
    displayon=false, mpi=false
) where {T<:AbstractGaugefields}
    if mpi
        MDstep_core_mpi!(gauge_action, U, p, MDsteps, Dim, Uold;
                         displayon=displayon)
    else
        MDstep_core!(gauge_action, U, p, MDsteps, Dim, Uold;
                     displayon=displayon)
    end
end

function MDstep!(
    gauge_action::GaugeAction,
    U::Array{T,1},
    B::Array{T,2},
    p, MDsteps, Dim,
    Uold::Array{T,1},
    temps::Temporalfields;
    displayon=false, mpi=false
) where {T<:AbstractGaugefields}
    if mpi
        MDstep_core_mpi!(gauge_action, U, B, p, MDsteps, Dim, Uold, temps;
                         displayon=displayon)
    else
        MDstep_core!(gauge_action, U, B, p, MDsteps, Dim, Uold, temps;
                     displayon=displayon)
    end
end
function MDstep!(
    gauge_action::GaugeAction,
    U::Array{T,1},
    B::Array{T,2},
    p, MDsteps, Dim,
    Uold::Array{T,1};
    displayon=false, mpi=false
) where {T<:AbstractGaugefields}
    if mpi
        MDstep_core_mpi!(gauge_action, U, B, p, MDsteps, Dim, Uold;
                         displayon=displayon)
    else
        MDstep_core!(gauge_action, U, B, p, MDsteps, Dim, Uold;
                     displayon=displayon)
    end
end

# Halfway-updating HMC
function MDstep!(
    gauge_action::GaugeAction,
    U::Array{T,1},
    B::Array{T,2},
    flux, p, MDsteps, Dim,
    Uold::Array{T,1},
    Bold::Array{T,2},
    flux_old,
    temps::Temporalfields;
    mpi=false,
    PEs=nothing
) where {T<:AbstractGaugefields}
    if mpi
        MDstep_dynB_mpi!(gauge_action,U,B,flux,p,MDsteps,Dim,Uold,Bold,flux_old,temps,PEs)
    else
        MDstep_dynB!(gauge_action,U,B,flux,p,MDsteps,Dim,Uold,Bold,flux_old,temps)
    end
end
function MDstep!(
    gauge_action::GaugeAction,
    U::Array{T,1},
    B::Array{T,2},
    flux, p, MDsteps, Dim,
    Uold::Array{T,1},
    Bold::Array{T,2},
    flux_old;
    mpi=false,
    PEs=nothing
) where {T<:AbstractGaugefields}
    if mpi
        MDstep_dynB_mpi!(gauge_action,U,B,flux,p,MDsteps,Dim,Uold,Bold,flux_old,PEs)
    else
        MDstep_dynB!(gauge_action,U,B,flux,p,MDsteps,Dim,Uold,Bold,flux_old)
    end
end
# Double-tesing HMC
function MDstep!(
    gauge_action::GaugeAction,
    U::Array{T,1},
    B::Array{T,2},
    flux, p, MDsteps, num_HMC, Dim,
    Uold1::Array{T,1},
    Uold2::Array{T,1},
    Bold::Array{T,2},
    flux_old,
    temps::Temporalfields;
    PEs=nothing
) where {T<:AbstractGaugefields}
    if mpi
        MDstep_dynB_mpi!(gauge_action,U,B,flux,p,MDsteps,num_HMC,Dim,
                         Uold1,Uold2,Bold,flux_old,temps,PEs)
    else
        MDstep_dynB!(gauge_action,U,B,flux,p,MDsteps,num_HMC,Dim,
                     Uold1,Uold2,Bold,flux_old,temps)
    end
end
function MDstep!(
    gauge_action::GaugeAction,
    U::Array{T,1},
    B::Array{T,2},
    flux, p, MDsteps, num_HMC, Dim,
    Uold1::Array{T,1},
    Uold2::Array{T,1},
    Bold::Array{T,2},
    flux_old;
    PEs=nothing
) where {T<:AbstractGaugefields}
    if mpi
        MDstep_dynB_mpi!(gauge_action,U,B,flux,p,MDsteps,num_HMC,Dim,
                         Uold1,Uold2,Bold,flux_old,PEs)
    else
        MDstep_dynB!(gauge_action,U,B,flux,p,MDsteps,num_HMC,Dim,
                     Uold1,Uold2,Bold,flux_old)
    end
end

function MDstep!(
    gauge_action::GaugeAction,
    U::Array{T,1},
    p, MDsteps, Dim,
    Uold::Array{T,1},
    nn, dSdU,
    temps::Temporalfields;
    displayon=false, mpi=false
) where {T<:AbstractGaugefields}
    if mpi
        MDstep_stout_mpi!(gauge_action, U, p, MDsteps, Dim, Uold, nn, dSdU, temps;
                          displayon=displayon)
    else
        MDstep_stout!(gauge_action, U, p, MDsteps, Dim, Uold, nn, dSdU, temps;
                      displayon=displayon)
    end
end
function MDstep!(
    gauge_action::GaugeAction,
    U::Array{T,1},
    p, MDsteps, Dim,
    Uold::Array{T,1},
    nn, dSdU;
    displayon=false, mpi=false
) where {T<:AbstractGaugefields}
    if mpi
        MDstep_stout_mpi!(gauge_action, U, p, MDsteps, Dim, Uold, nn, dSdU;
                          displayon=displayon)
    else
        MDstep_stout!(gauge_action, U, p, MDsteps, Dim, Uold, nn, dSdU;
                      displayon=displayon)
    end
end



end