struct A_Gaugefields_3D_serial{NC,NumofBasis} <: A_Gaugefields_3D{NC}
    a::Array{Float64,5}
    NX::Int64
    NY::Int64
    NT::Int64
    NC::Int64
    NumofBasis::Int64
    generators::Union{Nothing,Generator}

    function A_Gaugefields_3D_serial(NC, NX, NY, NT)
        NumofBasis = NC^2
        if NC <= 3
            generators = nothing
        else
            generators = Generator(NC)
        end

        return new{NC,NumofBasis}(
            zeros(Float64, NumofBasis, NX, NY, NT),
            NX,
            NY,
            NT,
            NC,
            NumofBasis,
            generators,
        )
    end
end

function Base.setindex!(x::T, v, i...) where {T<:A_Gaugefields_3D_serial}
    @inbounds x.a[i...] = v
end

function Base.getindex(x::T, i...) where {T<:A_Gaugefields_3D_serial}
    @inbounds return x.a[i...]
end


function Base.similar(u::A_Gaugefields_3D_serial{NC,NumofBasis}) where {NC,NumofBasis}
    return A_Gaugefields_3D_serial(NC, u.NX, u.NY, u.NT)
    #error("similar! is not implemented in type $(typeof(U)) ")
end




function gauss_distribution!(
    p::A_Gaugefields_3D_serial{NC,NumofBasis};
    σ = 1.0,
) where {NC,NumofBasis}
    d = Normal(0.0, σ)
    NT = p.NT
    NY = p.NY
    NX = p.NX
    #NumofBasis = Uμ.NumofBasis
    pwork = rand(d, NX * NY * NT * NumofBasis)
    icount = 0
    @inbounds for it = 1:NT
        for iy = 1:NY
            for ix = 1:NX
                for k = 1:NumofBasis
                    icount += 1
                    p[k, ix, iy, it] = pwork[icount]
                end
            end
        end
    end
end

function substitute_U!(
    Uμ::A_Gaugefields_3D_serial{NC,NumofBasis},
    pwork,
) where {NC,NumofBasis}
    NT = Uμ.NT
    NY = Uμ.NY
    NX = Uμ.NX
    #NumofBasis = Uμ.NumofBasis
    icount = 0
    for it = 1:NT
        for iy = 1:NY
            for ix = 1:NX
                for k = 1:NumofBasis
                    icount += 1
                    Uμ[k, ix, iy, it] = pwork[icount]
                end
            end
        end
    end
end



function Base.:*(
    x::A_Gaugefields_3D_serial{NC,NumofBasis},
    y::A_Gaugefields_3D_serial{NC,NumofBasis},
) where {NC,NumofBasis}
    NT = x.NT
    NY = x.NY
    NX = x.NX
    #NumofBasis = Uμ.NumofBasis
    s = 0.0
    for it = 1:NT
        for iy = 1:NY
            for ix = 1:NX
                for k = 1:NumofBasis
                    s += x[k, ix, iy, it] * y[k, ix, iy, it]
                end
            end
        end
    end

    return s
end

function clear_U!(Uμ::A_Gaugefields_3D_serial{NC,NumofBasis}) where {NC,NumofBasis}
    NT = Uμ.NT
    NY = Uμ.NY
    NX = Uμ.NX
    #NumofBasis = Uμ.NumofBasis
    for it = 1:NT
        for iy = 1:NY
            for ix = 1:NX
                for k = 1:NumofBasis
                    Uμ[k, ix, iy, it] = 0
                end
            end
        end
    end
end

function add_U!(
    c::A_Gaugefields_3D_serial{NC,NumofBasis},
    α::N,
    a::A_Gaugefields_3D_serial{NC,NumofBasis},
) where {NC,N<:Number,NumofBasis}
    NT = c.NT
    NY = c.NY
    NX = c.NX
    #NumofBasis = c.NumofBasis
    for it = 1:NT
        for iy = 1:NY
            for ix = 1:NX
                for k = 1:NumofBasis
                    c[k, ix, iy, it] =
                        c[k, ix, iy, it] + α * a[k, ix, iy, it]
                end
            end
        end
    end
    #error("add_U! is not implemented in type $(typeof(c)) ")
end


function exptU!(
    uout::T,
    t::N,
    u::A_Gaugefields_3D_serial{1,NumofBasis},
    temps::Array{T,1},
) where {N<:Number,T<:Gaugefields_3D,NumofBasis} #uout = exp(t*u)
    #g = u.generators
    NT = u.NT
    NY = u.NY
    NX = u.NX

    @inbounds for it = 1:NT
        for iy=1:NY
            for ix = 1:NX
                uout[1, 1, ix,iy,it] = exp(t * im * u[1, ix, iy,it])
            end
        end
    end
    #error("exptU! is not implemented in type $(typeof(u)) ")
end



function exptU!(
    uout::T,
    t::N,
    u::A_Gaugefields_3D_serial{NC,NumofBasis},
    temps::Array{T,1},
) where {N<:Number,T<:Gaugefields_3D,NC,NumofBasis} #uout = exp(t*u)
    ###@assert NC != 3 && NC != 2 "This function is for NC != 2,3"
    g = u.generators
    NT = u.NT
    NY = u.NY
    NX = u.NX

    u0 = zeros(ComplexF64, NC, NC)
    a = zeros(Float64, length(g))
    @inbounds for it = 1:NT
        for iy = 1:NY
            for ix = 1:NX
                for k = 1:length(a)
                    a[k] = u[k, ix, iy, it]
                end
                
                lie2matrix!(u0, g, a)
                uout[:, :, ix, iy, it] = exp(t * (im / 2) * u0)
                
            end
        end
    end
    #error("exptU! is not implemented in type $(typeof(u)) ")
end
