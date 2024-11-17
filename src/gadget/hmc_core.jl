module HMC_core_module

using Requires

import ..AbstractGaugefields_module
import ..GaugeAction_module
import ..HMC_module
import ..Abstractsmearing_module
import ..Temporalfields_module

include("./hmc_mdstep.jl")

function __init__()
    @require MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195" begin
        include("./hmc_mdstep_mpi.jl")
    end
end

function MDstep!(gauge_action, U, p, MDsteps, Dim, Uold, temps;
                 displayon=false, mpi=false)
    if mpi
        MDstep_core_mpi!(gauge_action, U, p, MDsteps, Dim, Uold, temps;
                         displayon=displayon)
    else
        MDstep_core!(gauge_action, U, p, MDsteps, Dim, Uold, temps;
                     displayon=displayon)
    end
end
function MDstep!(gauge_action, U, B, p, MDsteps, Dim, Uold, temps;
                 displayon=false, mpi=false)
    if mpi
        MDstep_core_mpi!(gauge_action, U, B, p, MDsteps, Dim, Uold, temps;
                         displayon=displayon)
    else
        MDstep_core!(gauge_action, U, B, p, MDsteps, Dim, Uold, temps;
                     displayon=displayon)
    end
end

# Halfway-updating HMC
function MDstep!(gauge_action,U,B,flux,p,MDsteps,Dim,Uold,Bold,flux_old,temps;
                 mpi=false)
    if mpi
        MDstep_dynB_mpi!(gauge_action,U,B,flux,p,MDsteps,Dim,Uold,Bold,flux_old,temps)
    else
        MDstep_dynB!(gauge_action,U,B,flux,p,MDsteps,Dim,Uold,Bold,flux_old,temps)
    end
end
# Double-tesing HMC
function MDstep!(gauge_action,U,B,flux,p,MDsteps,num_HMC,Dim,
                      Uold1,Uold2,Bold,flux_old,temps)
    if mpi
        MDstep_dynB_mpi!(gauge_action,U,B,flux,p,MDsteps,num_HMC,Dim,
                         Uold1,Uold2,Bold,flux_old,temps)
    else
        MDstep_dynB!(gauge_action,U,B,flux,p,MDsteps,num_HMC,Dim,
                     Uold1,Uold2,Bold,flux_old,temps)
    end
end

function MDstep!(gauge_action, U, p, MDsteps, Dim, Uold, nn, dSdU, temps;
                 displayon=false, mpi=false)
    if mpi
        MDstep_stout_mpi!(gauge_action, U, p, MDsteps, Dim, Uold, nn, dSdU, temps;
                          displayon=displayon)
    else
        MDstep_stout!(gauge_action, U, p, MDsteps, Dim, Uold, nn, dSdU, temps;
                      displayon=displayon)
    end
end



end
