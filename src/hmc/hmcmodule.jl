module HMC_module

using Random
using LinearAlgebra

import ..AbstractGaugefields_module:
    AbstractGaugefields,
    Initialize_Bfields,
    gaugetransf_4D_Bfields!,
    gauss_distribution!,
    substitute_U!,
    exptU!,
    mul!,
    Traceless_antihermitian_add!
import ..GaugeAction_module:
    GaugeAction,
    get_temporary_gaugefields,
    evaluate_GaugeAction,
    calc_dSdUμ!
import ..Abstractsmearing_module: calc_smearedU, back_prop
import ..Temporalfields_module: Temporalfields, get_temp, unused!

function calc_action(
    gauge_action::GaugeAction,
    U::Array{T,1},
    p
) where {T<:AbstractGaugefields}
    NC = U[1].NC
    Sg = -evaluate_GaugeAction(gauge_action, U) / NC #evaluate_Gauge_action(gauge_action,U) = tr(evaluate_Gaugeaction_untraced(gauge_action,U))
    Sp = p * p / 2
    S = Sp + Sg
    return real(S)
end
function calc_action(
    gauge_action::GaugeAction,
    U::Array{T,1},
    B::Array{T,2},
    p
) where {T<:AbstractGaugefields}
    NC = U[1].NC
    Sg = -evaluate_GaugeAction(gauge_action, U, B) / NC
    Sp = p * p / 2
    S = Sp + Sg
    return real(S)
end

function U_update!(
    U::Array{T,1},
    p,
    ϵ,
    Δτ,
    Dim,
    gauge_action::GaugeAction,
    temps::Temporalfields
) where {T<:AbstractGaugefields}
    #temps = get_temporary_gaugefields(gauge_action)
    temp1, it_temp1 = get_temp(temps)#[1]
    temp2, it_temp2 = get_temp(temps)#temps[2]
    expU, it_expU = get_temp(temps)#[3]
    W, it_W = get_temp(temps)#[4]

    for μ = 1:Dim
        exptU!(expU, ϵ * Δτ, p[μ], [temp1, temp2])
        mul!(W, expU, U[μ])
        substitute_U!(U[μ], W)
    end
    unused!(temps, it_temp1)
    unused!(temps, it_temp2)
    unused!(temps, it_expU)
    unused!(temps, it_W)
end
function U_update!(
    U::Array{T,1},
    p,
    ϵ,
    Δτ,
    Dim,
    gauge_action::GaugeAction,
) where {T<:AbstractGaugefields}
    temps = get_temporary_gaugefields(gauge_action)
    temp1, it_temp1 = get_temp(temps)#[1]
    temp2, it_temp2 = get_temp(temps)#temps[2]
    expU, it_expU = get_temp(temps)#[3]
    W, it_W = get_temp(temps)#[4]

    for μ = 1:Dim
        exptU!(expU, ϵ * Δτ, p[μ], [temp1, temp2])
        mul!(W, expU, U[μ])
        substitute_U!(U[μ], W)
    end
    unused!(temps, it_temp1)
    unused!(temps, it_temp2)
    unused!(temps, it_expU)
    unused!(temps, it_W)
end


function P_update!(
    U::Array{T,1},
    p,
    ϵ,
    Δτ,
    Dim,
    gauge_action::GaugeAction,
    temps::Temporalfields
) where {T<:AbstractGaugefields}  # p -> p +factor*U*dSdUμ
    NC = U[1].NC
    #temps = get_temporary_gaugefields(gauge_action)
    temp1, it_temp1 = get_temp(temps)
    dSdUμ, it_dSdUμ = get_temp(temps)#[end]
    factor = -ϵ * Δτ / (NC)

    for μ = 1:Dim
        calc_dSdUμ!(dSdUμ, gauge_action, μ, U)
        mul!(temp1, U[μ], dSdUμ) # U*dSdUμ
        Traceless_antihermitian_add!(p[μ], factor, temp1)
    end
    unused!(temps, it_dSdUμ)
    unused!(temps, it_temp1)
end
function P_update!(
    U::Array{T,1},
    p,
    ϵ,
    Δτ,
    Dim,
    gauge_action::GaugeAction
) where {T<:AbstractGaugefields} # p -> p +factor*U*dSdUμ
    NC = U[1].NC
    temps = get_temporary_gaugefields(gauge_action)
    temp1, it_temp1 = get_temp(temps)
    dSdUμ, it_dSdUμ = get_temp(temps)
    factor = -ϵ * Δτ / (NC)

    for μ = 1:Dim
        calc_dSdUμ!(dSdUμ, gauge_action, μ, U)
        mul!(temp1, U[μ], dSdUμ) # U*dSdUμ
        Traceless_antihermitian_add!(p[μ], factor, temp1)
    end
    unused!(temps, it_dSdUμ)
    unused!(temps, it_temp1)
end

function P_update!(
    U::Array{T,1},
    B::Array{T,2},
    p,
    ϵ,
    Δτ,
    Dim,
    gauge_action::GaugeAction,
    temps::Temporalfields
) where {T<:AbstractGaugefields}
    NC = U[1].NC
    temp1, it_temp1 = get_temp(temps)
    dSdUμ, it_dSdUμ = get_temp(temps)
    factor = -ϵ * Δτ / (NC)

    for μ = 1:Dim
        calc_dSdUμ!(dSdUμ, gauge_action, μ, U, B)
        mul!(temp1, U[μ], dSdUμ) # U*dSdUμ
        Traceless_antihermitian_add!(p[μ], factor, temp1)
    end
    unused!(temps, it_dSdUμ)
    unused!(temps, it_temp1)
end
function P_update!(
    U::Array{T,1},
    B::Array{T,2},
    p,
    ϵ,
    Δτ,
    Dim,
    gauge_action::GaugeAction
) where {T<:AbstractGaugefields}
    NC = U[1].NC
    temps = get_temporary_gaugefields(gauge_action)
    temp1, it_temp1 = get_temp(temps)
    dSdUμ, it_dSdUμ = get_temp(temps)
    factor = -ϵ * Δτ / (NC)
    for μ = 1:Dim
        calc_dSdUμ!(dSdUμ, gauge_action, μ, U, B)
        mul!(temp1, U[μ], dSdUμ) # U*dSdUμ
        Traceless_antihermitian_add!(p[μ], factor, temp1)
    end
    unused!(temps, it_dSdUμ)
    unused!(temps, it_temp1)
end

function P_update!(
    U::Array{T,1},
    p,
    ϵ,
    Δτ,
    Dim,
    gauge_action::GaugeAction,
    dSdU,
    nn,
    temps::Temporalfields
) where {T<:AbstractGaugefields}
    NC = U[1].NC
    factor = -ϵ * Δτ / (NC)
    temp1, it_temp1 = get_temp(temps)

    Uout, Uout_multi, _ = calc_smearedU(U, nn)

    for μ = 1:Dim
        calc_dSdUμ!(dSdU[μ], gauge_action, μ, Uout)
    end

    dSdUbare = back_prop(dSdU, nn, Uout_multi, U)

    for μ = 1:Dim
        mul!(temp1, U[μ], dSdUbare[μ]) # U*dSdUμ
        Traceless_antihermitian_add!(p[μ], factor, temp1)
    end
    unused!(temps, it_temp1)
end
function P_update!(
    U::Array{T,1},
    p,
    ϵ,
    Δτ,
    Dim,
    gauge_action::GaugeAction,
    dSdU,
    nn
) where {T<:AbstractGaugefields}
    NC = U[1].NC
    factor = -ϵ * Δτ / (NC)
    temp1, it_temp1 = get_temp(temps)

    Uout, Uout_multi, _ = calc_smearedU(U, nn)

    for μ = 1:Dim
        calc_dSdUμ!(dSdU[μ], gauge_action, μ, Uout)
    end

    dSdUbare = back_prop(dSdU, nn, Uout_multi, U)

    for μ = 1:Dim
        mul!(temp1, U[μ], dSdUbare[μ]) # U*dSdUμ
        Traceless_antihermitian_add!(p[μ], factor, temp1)
    end
    unused!(temps, it_temp1)
end

function Flux_update!(
    B::Array{T,2},
    flux;
    condition="randall",
    γ=1.0,
) where {T<:AbstractGaugefields}

    NC  = B[1,2].NC
    NDW = B[1,2].NDW
    NX  = B[1,2].NX
    NY  = B[1,2].NY
    NZ  = B[1,2].NZ
    NT  = B[1,2].NT

    if condition=="randall"
        flux[:] = rand(0:NC-1,6)
    elseif condition=="normall"
        for i = 1:6
            flux[i] = Flux_normal_update(flux[i], NC, γ)
        end
    elseif condition=="randone"
        i = rand(1:6)
        flux[i] += rand(-1:1)
        flux[i] %= NC
        flux[i] += (flux[i] < 0) ? NC : 0
    elseif condition=="normone"
        i = rand(1:6)
        flux[i] = Flux_normal_update(flux[i], NC, γ)
    elseif condition=="zeros"
        flux[:] = zeros(Int,6)
    elseif condition=="temporal"
        flux[3] = rand(0:NC-1)
        flux[5] = rand(0:NC-1)
        flux[6] = rand(0:NC-1)
    elseif condition=="norm_temporal"
        flux[3] = Flux_normal_update(flux[3], NC, γ)
        flux[5] = Flux_normal_update(flux[5], NC, γ)
        flux[6] = Flux_normal_update(flux[6], NC, γ)
    elseif condition=="12"
        flux[1] = rand(0:NC-1)
    elseif condition=="13"
        flux[2] = rand(0:NC-1)
    elseif condition=="14"
        flux[3] = rand(0:NC-1)
    elseif condition=="23"
        flux[4] = rand(0:NC-1)
    elseif condition=="24"
        flux[5] = rand(0:NC-1)
    elseif condition=="34"
        flux[6] = rand(0:NC-1)
    elseif condition=="12_34"
        flux[1] = rand(0:NC-1)
        flux[6] = rand(0:NC-1)
    elseif condition=="13_24"
        flux[2] = rand(0:NC-1)
        flux[5] = rand(0:NC-1)
    elseif condition=="14_23"
        flux[3] = rand(0:NC-1)
        flux[4] = rand(0:NC-1)
    elseif condition=="norm_12"
        flux[1] = Flux_normal_update(flux[1], NC, γ)
    elseif condition=="norm_13"
        flux[2] = Flux_normal_update(flux[2], NC, γ)
    elseif condition=="norm_14"
        flux[3] = Flux_normal_update(flux[3], NC, γ)
    elseif condition=="norm_23"
        flux[4] = Flux_normal_update(flux[4], NC, γ)
    elseif condition=="norm_24"
        flux[5] = Flux_normal_update(flux[5], NC, γ)
    elseif condition=="norm_34"
        flux[6] = Flux_normal_update(flux[6], NC, γ)
    elseif condition=="norm_12_34"
        flux[1] = Flux_normal_update(flux[1], NC, γ)
        flux[6] = Flux_normal_update(flux[1], NC, γ)
    elseif condition=="norm_13_24"
        flux[2] = Flux_normal_update(flux[2], NC, γ)
        flux[5] = Flux_normal_update(flux[1], NC, γ)
    elseif condition=="norm_14_23"
        flux[3] = Flux_normal_update(flux[1], NC, γ)
        flux[4] = Flux_normal_update(flux[1], NC, γ)
    end

    B = Initialize_Bfields(NC,flux,NDW,NX,NY,NZ,NT,condition = "tflux")

end

function sample_categorical(prob)
    cdf = cumsum(prob)
    r = rand()
    for (j, p) in enumerate(cdf)
        if r < p
            return j - 1
        end
    end
    return length(probabilities) - 1
end
function Flux_normal_update(z_i, NC, γ)
    z_val = 0:(NC-1)
    
    prob = exp.(-γ .* (z_val .- z_i) .^ 2)
    prob /= sum(prob)
    
    z_j = sample_categorical(prob)
    return z_j
end

function Flux_update!(
    B::Array{T,2},
    Btemp::Array{T,2},
    flux,
    temps::Temporalfields;
    verbose_level = 2,
    randomnumber = "Random",
    numtransf = 0,
) where {T<:AbstractGaugefields}

    NC  = B[1,2].NC
    NDW = B[1,2].NDW
    NX  = B[1,2].NX
    NY  = B[1,2].NY
    NZ  = B[1,2].NZ
    NT  = B[1,2].NT

#    i = rand(1:6)
#    flux[i] += rand(-1:1)
#    flux[i] %= NC
#    flux[i] += (flux[i] < 0) ? NC : 0
    flux[:] = rand(0:NC-1,6)
    Btemp = Initialize_Bfields(NC,flux,NDW,NX,NY,NZ,NT,condition = "tflux")

    gaugetransf_4D_Bfields!(B,Btemp,temps,verbose_level=verbose_level,randomnumber=randomnumber,numtransf=numtransf)
end
function Flux_update!(
    B::Array{T,2},
    Btemp::Array{T,2},
    flux,
    gauge_action::GaugeAction;
    verbose_level = 2,
    randomnumber = "Random",
    numtransf = 0,
) where {T<:AbstractGaugefields}

    NC  = B[1,2].NC
    NDW = B[1,2].NDW
    NX  = B[1,2].NX
    NY  = B[1,2].NY
    NZ  = B[1,2].NZ
    NT  = B[1,2].NT

#    i = rand(1:6)
#    flux[i] += rand(-1:1)
#    flux[i] %= NC
#    flux[i] += (flux[i] < 0) ? NC : 0
    flux[:] = rand(0:NC-1,6)
    Btemp = Initialize_Bfields(NC,flux,NDW,NX,NY,NZ,NT,condition = "tflux")

    temps = get_temporary_gaugefields(gauge_action)
    gaugetransf_4D_Bfields!(B,Btemp,temps,verbose_level=verbose_level,randomnumber=randomnumber,numtransf=numtransf)
end

function set_comb(
    U::Array{T,1},
    Dim
) where {T<:AbstractGaugefields}
    if Dim == 4
        comb = 6 #4*3/2
    elseif Dim == 3
        comb = 3
    elseif Dim == 2
        comb = 1
    else
        error("dimension $Dim is not supported")
    end
    
    factor = 1 / (comb * U[1].NV * U[1].NC)

    return comb, factor
end

end
