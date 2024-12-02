abstract type A_Gaugefields_3D{NC} <: TA_Gaugefields{NC,3} end

include("./TA_gaugefields_3D_serial.jl")


function A_Gaugefields(NC, NX, NY, NT)
    return A_Gaugefields_3D_serial(NC, NX, NY, NT)
end

function clear_U!(U::Array{T,1}) where {T<:A_Gaugefields_3D}
    for μ = 1:3
        clear_U!(U[μ])
    end
end
