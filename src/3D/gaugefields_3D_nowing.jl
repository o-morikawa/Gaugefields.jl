using Random
using LinearAlgebra

function random_unitary(rng, N; scale=1)
    A = scale * rand(rng, ComplexF64, N, N) +
        scale * im * rand(rng, ComplexF64, N, N)
    Q, R = qr(A)
    Q *= Diagonal(sign.(diag(R)))
    return Q
end

"""
`Gaugefields_3D_nowing{NC} <: Gaugefields_3D{NC}``

U(N) Gauge fields in three dimensional lattice. 
"""
struct Gaugefields_3D_nowing{NC} <: Gaugefields_3D{NC}
    U::Array{ComplexF64,5}
    NX::Int64
    NY::Int64
    NT::Int64
    NDW::Int64
    NV::Int64
    NC::Int64
    mpi::Bool
    verbose_print::Verbose_print
    Ushifted::Array{ComplexF64,5}

    function Gaugefields_3D_nowing(
        NC::T,
        NX::T,
        NY::T,
        NT::T;
        verbose_level = 2,
    ) where {T<:Integer}
        NV = NX * NY * NT
        NDW = 0
        U = zeros(ComplexF64, NC, NC, NX + 2NDW, NY + 2NDW, NT + 2NDW)
        Ushifted = zero(U)
        mpi = false
        verbose_print = Verbose_print(verbose_level)
        return new{NC}(U, NX, NY, NT, NDW, NV, NC, mpi, verbose_print, Ushifted)
    end
end

function write_to_numpyarray(U::T, filename) where {T<:Gaugefields_3D_nowing}
    data = Dict{String,Any}()
    data["U"] = U.U
    data["NX"] = U.NX
    data["NY"] = U.NY
    data["NT"] = U.NT
    data["NV"] = U.NV
    data["NDW"] = U.NDW
    data["NC"] = U.NC

    npzwrite(filename, data)
end




@inline function get_latticeindex(i, NX, NY, NT)
    ix = (i - 1) % NX + 1
    ii = div(i - ix, NX)
    iy = ii % NY + 1
    ii = div(ii - (iy - 1), NY)
    it = ii % NT + 1
    return ix, iy, it
end


function Base.setindex!(x::Gaugefields_3D_nowing, v, i1, i2, i3, i4, i5)
    @inbounds x.U[i1, i2, i3, i4, i5] = v
end

@inline function Base.getindex(x::Gaugefields_3D_nowing, i1, i2, i3, i4, i5)
    @inbounds return x.U[i1, i2, i3, i4, i5]
end

function Base.setindex!(x::Gaugefields_3D_nowing, v, i1, i2, ii)
    ix, iy, it = get_latticeindex(ii, x.NX, x.NY, x.NT)
    @inbounds x.U[i1, i2, ix, iy, it] = v
end

@inline function Base.getindex(x::Gaugefields_3D_nowing, i1, i2, ii)
    ix, iy, it = get_latticeindex(ii, x.NX, x.NY, x.NT)
    @inbounds return x.U[i1, i2, ix, iy, it]
end





function substitute_U!(
    a::Array{T1,1},
    b::Array{T2,1},
) where {T1<:Gaugefields_3D_nowing,T2<:Gaugefields_3D_nowing}
    for μ = 1:3
        substitute_U!(a[μ], b[μ])
    end
end
function substitute_U!(
    a::Array{T1,2},
    b::Array{T2,2},
) where {T1<:Gaugefields_3D_nowing,T2<:Gaugefields_3D_nowing}
    for μ = 1:3
        for ν = 1:3
            if μ == ν
                continue
            end
            substitute_U!(a[μ,ν], b[μ,ν])
        end
    end
end

function substitute_U!(
    a::Array{T1,1},
    b::Array{T2,1},
    iseven,
) where {T1<:Gaugefields_3D_nowing,T2<:Gaugefields_3D_nowing}
    for μ = 1:3
        substitute_U!(a[μ], b[μ], iseven)
    end
end
function substitute_U!(
    a::Array{T1,2},
    b::Array{T2,2},
    iseven,
) where {T1<:Gaugefields_3D_nowing,T2<:Gaugefields_3D_nowing}
    for μ = 1:3
        for ν = 1:3
            if μ == ν
                continue
            end
            substitute_U!(a[μ,ν], b[μ,ν], iseven)
        end
    end
end

function Base.similar(U::T) where {T<:Gaugefields_3D_nowing}
    Uout = Gaugefields_3D_nowing(
        U.NC,
        U.NX,
        U.NY,
        U.NT,
        verbose_level = U.verbose_print.level,
    )
    return Uout
end

function Base.similar(U::Array{T,1}) where {T<:Gaugefields_3D_nowing}
    Uout = Array{T,1}(undef, 3)
    for μ = 1:3
        Uout[μ] = similar(U[μ])
    end
    return Uout
end
function Base.similar(U::Array{T,2}) where {T<:Gaugefields_3D_nowing}
    Uout = Array{T,2}(undef, 3, 3)
    for μ = 1:3
        for ν = 1:3
            if μ == ν
                continue
            end
            Uout[μ,ν] = similar(U[μ,ν])
        end
    end
    return Uout
end

function substitute_U!(a::T, b::T) where {T<:Gaugefields_3D_nowing}
    for i = 1:length(a.U)
        a.U[i] = b.U[i]
    end
    return
end

function substitute_U!(a::Gaugefields_3D_nowing{NC}, b::T2) where {NC,T2<:Abstractfields}
    NT = a.NT
    NY = a.NY
    NX = a.NX
    for it = 1:NT
        for iy = 1:NY
            for ix = 1:NX
                for k2 = 1:NC
                    for k1 = 1:NC
                        @inbounds a[k1, k2, ix, iy, it] = b[k1, k2, ix, iy, it]
                    end
                end
            end
        end
    end
    set_wing_U!(a)

end


function substitute_U!(
    a::Gaugefields_3D_nowing{NC},
    b::T2,
    iseven::Bool,
) where {NC,T2<:Abstractfields}
    NT = a.NT
    NY = a.NY
    NX = a.NX
    for it = 1:NT
        for iy = 1:NY
            for ix = 1:NX
                evenodd = ifelse((ix + iy + it) % 2 == 0, true, false)
                if evenodd == iseven
                    for k2 = 1:NC
                        for k1 = 1:NC
                            @inbounds a[k1, k2, ix, iy, it] =
                                b[k1, k2, ix, iy, it]
                        end
                    end
                end
            end
        end
    end
    set_wing_U!(a)

end

function m3complv!(a::T) where {T<:Gaugefields_3D_nowing}
    aa = zeros(Float64, 18)
    NX = a.NX
    NY = a.NY
    NT = a.NT

    @inbounds for it = 1:NT
        for iy = 1:NY
            @simd for ix = 1:NX

                aa[1] = real(a[1, 1, ix, iy, it])
                aa[2] = imag(a[1, 1, ix, iy, it])
                aa[3] = real(a[1, 2, ix, iy, it])
                aa[4] = imag(a[1, 2, ix, iy, it])
                aa[5] = real(a[1, 3, ix, iy, it])
                aa[6] = imag(a[1, 3, ix, iy, it])
                aa[7] = real(a[2, 1, ix, iy, it])
                aa[8] = imag(a[2, 1, ix, iy, it])
                aa[9] = real(a[2, 2, ix, iy, it])
                aa[10] = imag(a[2, 2, ix, iy, it])
                aa[11] = real(a[2, 3, ix, iy, it])
                aa[12] = imag(a[2, 3, ix, iy, it])

                aa[13] =
                    aa[3] * aa[11] - aa[4] * aa[12] - aa[5] * aa[9] + aa[6] * aa[10]
                aa[14] =
                    aa[5] * aa[10] + aa[6] * aa[9] - aa[3] * aa[12] - aa[4] * aa[11]
                aa[15] = aa[5] * aa[7] - aa[6] * aa[8] - aa[1] * aa[11] + aa[2] * aa[12]
                aa[16] = aa[1] * aa[12] + aa[2] * aa[11] - aa[5] * aa[8] - aa[6] * aa[7]
                aa[17] = aa[1] * aa[9] - aa[2] * aa[10] - aa[3] * aa[7] + aa[4] * aa[8]
                aa[18] = aa[3] * aa[8] + aa[4] * aa[7] - aa[1] * aa[10] - aa[2] * aa[9]

                a[3, 1, ix, iy, it] = aa[13] + im * aa[14]
                a[3, 2, ix, iy, it] = aa[15] + im * aa[16]
                a[3, 3, ix, iy, it] = aa[17] + im * aa[18]

                #println(a[:,:,ix,iy,iz,it]'*a[:,:,ix,iy,iz,it] )
            end
        end
    end
end

function det_unitary(U::Gaugefields_3D_nowing)
    NT = U.NT
    NY = U.NY
    NX = U.NX
    NV = U.NV

    d = zeros(ComplexF64,NX,NY,NT)
    for it = 1:NT
        for iy = 1:NY
            for ix = 1:NX
                d[ix,iy,it] = det(U[:,:,ix,iy,it])
            end
        end
    end
    return d
end

function randomGaugefields_3D_nowing(
    NC,
    NX,
    NY,
    NT;
    verbose_level = 2,
    randomnumber = "Random",
    scale = 1,
)
    U = Gaugefields_3D_nowing(NC, NX, NY, NT, verbose_level = verbose_level)
    if randomnumber == "Random"
        rng = MersenneTwister()
    elseif randomnumber == "Reproducible"
        rng = StableRNG(123)
    else
        error(
            "randomnumber should be \"Random\" or \"Reproducible\". Now randomnumber = $randomnumber",
        )
    end

    for it = 1:NT
        for iy = 1:NY
            @simd for ix = 1:NX
                U[:, :, ix, iy, it] = random_unitary(rng, NC, scale=scale)
            end
        end
    end
    set_wing_U!(U)
    return U
end

function RandomGauges_3D(NC, NX, NY, NT;
                         verbose_level = 2, randomnumber = "Random", randscale=1)
    return randomGaugefields_3D_nowing(
        NC,
        NX,
        NY,
        NT,
        verbose_level = verbose_level,
        randomnumber = randomnumber,
        scale = randscale,
    )
end

function IdentityGauges_3D(NC, NX, NY, NT; verbose_level = 2)
    return identityGaugefields_3D_nowing(NC, NX, NY, NT, verbose_level = verbose_level)
end

function identityGaugefields_3D_nowing(NC, NX, NY, NT; verbose_level = 2)
    U = Gaugefields_3D_nowing(NC, NX, NY, NT, verbose_level = verbose_level)

    for it = 1:NT
        for iy = 1:NY
            for ix = 1:NX
                @simd for ic = 1:NC
                    U[ic, ic, ix, iy, it] = 1
                end
            end
        end
    end
    set_wing_U!(U)
    return U
end


struct Shifted_Gaugefields_3D_nowing{NC} <: Shifted_Gaugefields{NC,3}
    parent::Gaugefields_3D_nowing{NC}
    shift::NTuple{3,Int8}
    NX::Int64
    NY::Int64
    NT::Int64

    function Shifted_Gaugefields_3D_nowing(U::Gaugefields_3D_nowing{NC}, shift) where {NC}
        shifted_U!(U, shift)
        return new{NC}(U, shift, U.NX, U.NY, U.NT)
    end
end

function shifted_U!(U::Gaugefields_3D_nowing{NC}, shift) where {NC}
    NT = U.NT
    NY = U.NY
    NX = U.NX
    for it = 1:NT
        it_shifted = it + shift[3]
        it_shifted += ifelse(it_shifted > NT, -NT, 0)
        it_shifted += ifelse(it_shifted < 1, NT, 0)
        for iy = 1:NY
            iy_shifted = iy + shift[2]
            iy_shifted += ifelse(iy_shifted > NY, -NY, 0)
            iy_shifted += ifelse(iy_shifted < 1, NY, 0)
            for ix = 1:NX
                ix_shifted = ix + shift[1]
                ix_shifted += ifelse(ix_shifted > NX, -NX, 0)
                ix_shifted += ifelse(ix_shifted < 1, NX, 0)
                for k2 = 1:NC
                    for k1 = 1:NC
                        U.Ushifted[k1, k2, ix, iy, it] =
                            U[k1, k2, ix_shifted, iy_shifted, it_shifted]
                    end
                end
            end
        end
    end
end

#lattice shift
function shift_U(U::Gaugefields_3D_nowing{NC}, ν::T) where {T<:Integer,NC}
    if ν == 1
        shift = (1, 0, 0)
    elseif ν == 2
        shift = (0, 1, 0)
    elseif ν == 3
        shift = (0, 0, 1)
    elseif ν == -1
        shift = (-1, 0, 0)
    elseif ν == -2
        shift = (0, -1, 0)
    elseif ν == -3
        shift = (0, 0, -1)
    end

    return Shifted_Gaugefields_3D_nowing(U, shift)
end

function shift_U(
    U::TU,
    shift::NTuple{Dim,T},
) where {Dim,T<:Integer,TU<:Gaugefields_3D_nowing}
    return Shifted_Gaugefields_3D_nowing(U, shift)
end


@inline function Base.getindex(
    U::Shifted_Gaugefields_3D_nowing{NC},
    i1,
    i2,
    i3,
    i4,
    i5,
) where {NC}
    @inbounds return U.parent.Ushifted[i1, i2, i3, i4, i5]
end

function Base.getindex(
    u::Staggered_Gaugefields{T,μ},
    i1,
    i2,
    i3,
    i4,
    i5,
) where {T<:Gaugefields_3D_nowing,μ}
    NT = u.parent.NT
    NY = u.parent.NY
    NX = u.parent.NX

    t = i5 - 1
    t += ifelse(t < 0, NT, 0)
    t += ifelse(t ≥ NT, -NT, 0)
    #boundary_factor_t = ifelse(t == NT -1,BoundaryCondition[3],1)
    y = i4 - 1
    y += ifelse(y < 0, NY, 0)
    y += ifelse(y ≥ NY, -NY, 0)
    #boundary_factor_y = ifelse(y == NY -1,BoundaryCondition[2],1)
    x = i3 - 1
    x += ifelse(x < 0, NX, 0)
    x += ifelse(x ≥ NX, -NX, 0)
    #boundary_factor_x = ifelse(x == NX -1,BoundaryCondition[1],1)
    if μ == 1
        η = 1
    elseif μ == 2
        #η = (-1.0)^(x)
        η = ifelse(x % 2 == 0, 1, -1)
    elseif μ == 3
        #η = (-1.0)^(x+y)
        η = ifelse((x + y) % 2 == 0, 1, -1)
    else
        error("η should be positive but η = $η")
    end

    @inbounds return η * u.parent[i1, i2, i3, i4, i5]
end

function Base.getindex(
    u::Staggered_Gaugefields{Shifted_Gaugefields_3D_nowing{NC},μ},
    i1,
    i2,
    i3,
    i4,
    i5,
) where {μ,NC}
    NT = u.parent.NT
    NY = u.parent.NY
    NX = u.parent.NX

    t = i5 - 1 + u.parent.shift[3]
    t += ifelse(t < 0, NT, 0)
    t += ifelse(t ≥ NT, -NT, 0)
    #boundary_factor_t = ifelse(t == NT -1,BoundaryCondition[3],1)
    y = i4 - 1 + u.parent.shift[2]
    y += ifelse(y < 0, NY, 0)
    y += ifelse(y ≥ NY, -NY, 0)
    #boundary_factor_y = ifelse(y == NY -1,BoundaryCondition[2],1)
    x = i3 - 1 + u.parent.shift[1]
    x += ifelse(x < 0, NX, 0)
    x += ifelse(x ≥ NX, -NX, 0)
    #boundary_factor_x = ifelse(x == NX -1,BoundaryCondition[1],1)
    if μ == 1
        η = 1
    elseif μ == 2
        #η = (-1.0)^(x)
        η = ifelse(x % 2 == 0, 1, -1)
    elseif μ == 3
        #η = (-1.0)^(x+y)
        η = ifelse((x + y) % 2 == 0, 1, -1)
    else
        error("η should be positive but η = $η")
    end

    @inbounds return η * u.parent[i1, i2, i3, i4, i5]
end

function map_U!(
    U::Gaugefields_3D_nowing{NC},
    f!::Function,
    V::Gaugefields_3D_nowing{NC},
    iseven::Bool,
) where {NC}
    NT = U.NT
    NY = U.NY
    NX = U.NX
    A = zeros(ComplexF64, NC, NC)
    B = zeros(ComplexF64, NC, NC)
    for it = 1:NT
        for iy = 1:NY
            for ix = 1:NX
                evenodd = ifelse((ix + iy + it) % 2 == 0, true, false)
                if evenodd == iseven
                    for k2 = 1:NC
                        for k1 = 1:NC
                            A[k1, k2] = V[k1, k2, ix, iy, it]
                            B[k1, k2] = U[k1, k2, ix, iy, it]
                        end
                    end
                    f!(B, A)
                    for k2 = 1:NC
                        for k1 = 1:NC
                            U[k1, k2, ix, iy, it] = B[k1, k2]
                        end
                    end
                end
            end
        end
    end
    set_wing_U!(U)
end

function map_U!(
    U::Gaugefields_3D_nowing{NC},
    f!::Function,
    V::Gaugefields_3D_nowing{NC},
) where {NC}
    NT = U.NT
    NY = U.NY
    NX = U.NX
    A = zeros(ComplexF64, NC, NC)
    B = zeros(ComplexF64, NC, NC)
    for it = 1:NT
        for iy = 1:NY
            for ix = 1:NX

                for k2 = 1:NC
                    for k1 = 1:NC
                        A[k1, k2] = V[k1, k2, ix, iy, it]
                        B[k1, k2] = U[k1, k2, ix, iy, it]
                    end
                end
                f!(B, A)
                for k2 = 1:NC
                    for k1 = 1:NC
                        U[k1, k2, ix, iy, it] = B[k1, k2]
                    end
                end

            end
        end
    end
    #set_wing_U!(U)
end



function map_U_sequential!(U::Gaugefields_3D_nowing{NC}, f!::Function, Uin) where {NC}
    NT = U.NT
    NY = U.NY
    NX = U.NX
    #A = zeros(ComplexF64,NC,NC)
    B = zeros(ComplexF64, NC, NC)
    for it = 1:NT
        for iy = 1:NY
            for ix = 1:NX

                for k2 = 1:NC
                    for k1 = 1:NC
                        B[k1, k2] = U[k1, k2, ix, iy, it]
                    end
                end
                f!(B, Uin, ix, iy, it)

                for k2 = 1:NC
                    for k1 = 1:NC
                        U[k1, k2, ix, iy, it] = B[k1, k2]
                    end
                end

            end
        end
    end
    #set_wing_U!(U)
end



function clear_U!(Uμ::Gaugefields_3D_nowing{NC}) where {NC}
    NT = Uμ.NT
    NY = Uμ.NY
    NX = Uμ.NX
    for it = 1:NT
        for iy = 1:NY
            for ix = 1:NX
                for k2 = 1:NC
                    for k1 = 1:NC
                        @inbounds Uμ[k1, k2, ix, iy, it] = 0
                    end
                end
            end
        end
    end
    set_wing_U!(Uμ)

end

function clear_U!(Uμ::Gaugefields_3D_nowing{NC}, iseven::Bool) where {NC}
    NT = Uμ.NT
    NY = Uμ.NY
    NX = Uμ.NX
    for it = 1:NT
        for iy = 1:NY
            for ix = 1:NX
                evenodd = ifelse((ix + iy + it) % 2 == 0, true, false)
                if evenodd == iseven
                    for k2 = 1:NC
                        for k1 = 1:NC
                            @inbounds Uμ[k1, k2, ix, iy, it] = 0
                        end
                    end
                end
            end
        end
    end
    set_wing_U!(Uμ)

end


function unit_U!(Uμ::Gaugefields_3D_nowing{NC}) where {NC}
    NT = Uμ.NT
    NY = Uμ.NY
    NX = Uμ.NX
    for it = 1:NT
        for iy = 1:NY
            for ix = 1:NX

                for k2 = 1:NC
                    for k1 = 1:NC
                        @inbounds Uμ[k1, k2, ix, iy, it] = ifelse(k1 == k2, 1, 0)
                    end
                end
            end
        end
    end
    set_wing_U!(Uμ)

end

"""
M = (U*δ_prev) star (dexp(Q)/dQ)
Λ = TA(M)
"""
function construct_Λmatrix_forSTOUT!(
    Λ,
    δ_current::Gaugefields_3D_nowing{NC},
    Q,
    u::Gaugefields_3D_nowing{NC},
) where {NC}
    NT = u.NT
    NY = u.NY
    NX = u.NX
    Qn = zeros(ComplexF64, NC, NC)
    Un = zero(Qn)
    Mn = zero(Qn)
    Λn = zero(Qn)
    δn_current = zero(Qn)
    temp1 = similar(Qn)
    temp2 = similar(Qn)
    temp3 = similar(Qn)

    for it = 1:NT
        for iy = 1:NY
            for ix = 1:NX

                for jc = 1:NC
                    for ic = 1:NC
                        Un[ic, jc] = u[ic, jc, ix, iy, it]
                        Qn[ic, jc] = Q[ic, jc, ix, iy, it]#*im
                        #if (ix,iy,iz,it) == (1,1,1,1)
                        #    println("Qij $ic $jc ",Qn[ic,jc],"\t ",Q[ic,jc,ix,iy,iz,it])
                        #end
                        δn_current[ic, jc] = δ_current[ic, jc, ix, iy, it]
                    end
                end

                #=
                if (ix,iy,iz,it) == (1,1,1,1)
                    println("Qn = ",Qn[1:3,1:3])
                end
                =#

                calc_Mmatrix!(Mn, δn_current, Qn, Un, u, [temp1, temp2, temp3])
                #=
                if (ix,iy,iz,it) == (1,1,1,1)
                    println(" Un ",  Un[1,1])
                    println("δn_current ", δn_current[1,1])
                    #println("Qn[1,1] = ",Qn[1,1])
                    println("M[1,1] = ",Mn[1,1])
                    #println("Qn = ",Qn[1:3,1:3])
                end
                =#
                calc_Λmatrix!(Λn, Mn, NC)

                
                for jc = 1:NC
                    for ic = 1:NC
                        Λ[ic, jc, ix, iy, it] = Λn[ic, jc]
                    end
                end

            end
        end
    end
    set_wing_U!(Λ)
end


function set_wing_U!(u::Array{Gaugefields_3D_nowing{NC},1}) where {NC} #do nothing
    return
end


function set_wing_U!(u::Gaugefields_3D_nowing{NC}) where {NC} # do nothing
    return
end

function exptU!(
    uout::T,
    t::N,
    u::Gaugefields_3D_nowing{NC},
    temps::Array{T,1},
) where {N<:Number,T<:Gaugefields_3D_nowing,NC} #uout = exp(t*u)
    ###@assert NC != 3 && NC != 2 "This function is for NC != 2,3"


    NT = u.NT
    NY = u.NY
    NX = u.NX
    V0 = zeros(ComplexF64, NC, NC)
    V1 = zeros(ComplexF64, NC, NC)
    for it = 1:NT
        for iy = 1:NY
            for ix = 1:NX
                for k2 = 1:NC
                    for k1 = 1:NC
                        @inbounds V0[k1, k2] = im * t * u[k1, k2, ix, iy, it]
                    end
                end
                V1[:, :] = exp(V0)
                for k2 = 1:NC
                    for k1 = 1:NC
                        @inbounds uout[k1, k2, ix, iy, it] = V1[k1, k2]
                    end
                end

            end
        end
    end
    #error("exptU! is not implemented in type $(typeof(u)) ")
end



"""
-----------------------------------------------------c
     !!!!!   vin and vout should be different vectors

     Projectin of the etraceless antiermite part 
     vout = x/2 - Tr(x)/6
     wher   x = vin - Conjg(vin)      
-----------------------------------------------------c
    """
function Traceless_antihermitian!(
    vout::Gaugefields_3D_nowing{NC},
    vin::Gaugefields_3D_nowing{NC};
    factor=1
) where {NC}
    #NC = vout.NC
    fac1N = 1 / NC
    nv = vin.NV

    NX = vin.NX
    NY = vin.NY
    NT = vin.NT

    for it = 1:NT
        for iy = 1:NY
            @simd for ix = 1:NX
                tri = 0.0
                @simd for k = 1:NC
                    tri += imag(vin[k, k, ix, iy, it])
                end
                tri *= fac1N
                @simd for k = 1:NC
                    vout[k, k, ix, iy, it] =
                        (imag(vin[k, k, ix, iy, it]) - tri) * im * factor
                end
            end
        end
    end

    for it = 1:NT
        for iy = 1:NY
            @simd for ix = 1:NX
                for k1 = 1:NC
                    @simd for k2 = k1+1:NC
                        vv =
                            0.5 * (
                                vin[k1, k2, ix, iy, it] -
                                    conj(vin[k2, k1, ix, iy, it])
                            )
                        vout[k1, k2, ix, iy, it] = vv*factor
                        vout[k2, k1, ix, iy, it] = -conj(vv)*factor
                    end
                end
            end
        end
    end


end

function Antihermitian!(
    vout::Gaugefields_3D_nowing{NC},
    vin::Gaugefields_3D_nowing{NC};factor = 1
) where {NC} #vout = factor*(vin - vin^+)

    #NC = vout.NC
    fac1N = 1 / NC
    nv = vin.NV

    NX = vin.NX
    NY = vin.NY
    NT = vin.NT



    for it = 1:NT
        for iy = 1:NY
            @simd for ix = 1:NX
                for k1 = 1:NC
                    @simd for k2 = k1:NC
                        vv =vin[k1, k2, ix, iy, it] -
                            conj(vin[k2, k1, ix, iy, it])
                        vout[k1, k2, ix, iy, it] = vv*factor
                        if k1 != k2
                            vout[k2, k1, ix, iy, it] = -conj(vv)*factor
                        end
                    end
                end
            end
        end
    end


end

function LinearAlgebra.tr(
    a::Gaugefields_3D_nowing{NC},
    b::Gaugefields_3D_nowing{NC},
) where {NC}
    NX = a.NX
    NY = a.NY
    NT = a.NT

    s = 0
    for it = 1:NT
        for iy = 1:NY
            for ix = 1:NX
                for k = 1:NC
                    for k2 = 1:NC
                        s += a[k, k2, ix, iy, it] * b[k2, k, ix, iy, it]
                    end
                end
            end
        end
    end
    #println(3*NT*NZ*NY*NX*NC)
    return s
end


function LinearAlgebra.tr(a::Gaugefields_3D_nowing{NC}) where {NC}
    NX = a.NX
    NY = a.NY
    NT = a.NT

    s = 0
    for it = 1:NT
        for iy = 1:NY
            for ix = 1:NX
                @simd for k = 1:NC
                    s += a[k, k, ix, iy, it]
                    #println(a[k,k,ix,iy,iz,it])
                end
            end
        end
    end
    #println(3*NT*NZ*NY*NX*NC)
    return s
end

function partial_tr(a::Gaugefields_3D_nowing{NC}, μ) where {NC}
    NX = a.NX
    NY = a.NY
    NT = a.NT

    if μ == 1
        s = 0
        ix = 1
        for it = 1:NT
            for iy = 1:NY
                #for ix=1:NX
                @simd for k = 1:NC
                    s += a[k, k, ix, iy, it]
                    #println(a[k,k,ix,iy,iz,it])
                end

                #end
            end
        end
    elseif μ == 2
        s = 0
        iy = 1
        for it = 1:NT
            #for iy=1:NY
            for ix = 1:NX
                @simd for k = 1:NC
                    s += a[k, k, ix, iy, it]
                    #println(a[k,k,ix,iy,iz,it])
                end
            end
            #end
        end
    elseif μ == 3
        s = 0
        it = 1
        #for iz=1:NT
        for iy = 1:NY
            for ix = 1:NX
                @simd for k = 1:NC
                    s += a[k, k, ix, iy, it]
                    #println(a[k,k,ix,iy,iz,it])
                end
            end
        end
        #end
    end



    #println(3*NT*NZ*NY*NX*NC)
    return s
end

function add_U!(c::Gaugefields_3D_nowing{NC}, a::T1) where {NC,T1<:Abstractfields}

    NT = c.NT
    NY = c.NY
    NX = c.NX
    for it = 1:NT
        for iy = 1:NY
            for ix = 1:NX
                for k2 = 1:NC
                    @inbounds @simd for k1 = 1:NC
                        c[k1, k2, ix, iy, it] += a[k1, k2, ix, iy, it]
                    end
                end
            end
        end
    end
end

function add_U!(
    c::Gaugefields_3D_nowing{NC},
    a::T1,
    iseven::Bool,
) where {NC,T1<:Abstractfields}
    NT = c.NT
    NY = c.NY
    NX = c.NX
    for it = 1:NT
        for iy = 1:NY
            for ix = 1:NX
                evenodd = ifelse((ix + iy + it) % 2 == 0, true, false)
                if evenodd == iseven
                    for k2 = 1:NC
                        @inbounds @simd for k1 = 1:NC
                            c[k1, k2, ix, iy, it] += a[k1, k2, ix, iy, it]
                        end
                    end
                end
            end
        end
    end
end


function add_U!(
    c::Gaugefields_3D_nowing{NC},
    α::N,
    a::T1,
) where {NC,T1<:Abstractfields,N<:Number}
    NT = c.NT
    NY = c.NY
    NX = c.NX
    @inbounds for it = 1:NT
        for iy = 1:NY
            for ix = 1:NX
                for k2 = 1:NC
                    @inbounds @simd for k1 = 1:NC
                        c[k1, k2, ix, iy, it] += α * a[k1, k2, ix, iy, it]
                    end
                end
            end
        end
    end
    set_wing_U!(c)
end




function LinearAlgebra.mul!(
    c::Gaugefields_3D_nowing{NC},
    a::T1,
    b::T2,
) where {NC,T1<:Abstractfields,T2<:Abstractfields}
    ###@assert NC != 2 && NC != 3 "This function is for NC != 2,3"
    NT = c.NT
    NY = c.NY
    NX = c.NX
    @inbounds for it = 1:NT
        for iy = 1:NY
            for ix = 1:NX
                for k2 = 1:NC
                    for k1 = 1:NC
                        c[k1, k2, ix, iy, it] = 0

                        @inbounds @simd for k3 = 1:NC
                            c[k1, k2, ix, iy, it] +=
                                a[k1, k3, ix, iy, it] * b[k3, k2, ix, iy, it]
                        end
                    end
                end
            end
        end
    end
    set_wing_U!(c)
end

function LinearAlgebra.mul!(
    c::Gaugefields_3D_nowing{NC},
    a::T1,
    b::T2,
    iseven::Bool,
) where {NC,T1<:Abstractfields,T2<:Abstractfields}
    ###@assert NC != 2 && NC != 3 "This function is for NC != 2,3"
    NT = c.NT
    NY = c.NY
    NX = c.NX
    @inbounds for it = 1:NT
        for iy = 1:NY
            for ix = 1:NX
                evenodd = ifelse((ix + iy + it) % 2 == 0, true, false)
                if evenodd == iseven
                    for k2 = 1:NC
                        for k1 = 1:NC
                            c[k1, k2, ix, iy, it] = 0
                            @inbounds @simd for k3 = 1:NC
                                c[k1, k2, ix, iy, it] +=
                                    a[k1, k3, ix, iy, it] *
                                    b[k3, k2, ix, iy, it]
                            end
                        end
                    end
                end
            end
        end
    end
    set_wing_U!(c)
end

function LinearAlgebra.mul!(
    c::Gaugefields_3D_nowing{NC},
    a::T1,
    b::T2,
) where {NC,T1<:Number,T2<:Abstractfields}
    NT = c.NT
    NY = c.NY
    NX = c.NX
    for it = 1:NT
        for iy = 1:NY
            for ix = 1:NX
                for k2 = 1:NC
                    @inbounds @simd for k1 = 1:NC
                        c[k1, k2, ix, iy, it] = a * b[k1, k2, ix, iy, it]
                    end
                end
            end
        end
    end
    set_wing_U!(c)
end

function LinearAlgebra.mul!(
    c::Gaugefields_3D_nowing{NC},
    a::T1,
    b::T2,
    α::Ta,
    β::Tb,
) where {NC,T1<:Abstractfields,T2<:Abstractfields,Ta<:Number,Tb<:Number}
    ###@assert NC != 2 && NC != 3 "This function is for NC != 2,3"
    NT = c.NT
    NY = c.NY
    NX = c.NX
    for it = 1:NT
        for iy = 1:NY
            for ix = 1:NX
                for k2 = 1:NC
                    for k1 = 1:NC
                        c[k1, k2, ix, iy, it] = β * c[k1, k2, ix, iy, it]
                        @inbounds @simd for k3 = 1:NC
                            c[k1, k2, ix, iy, it] +=
                                α *
                                a[k1, k3, ix, iy, it] *
                                b[k3, k2, ix, iy, it]
                        end
                    end
                end
            end
        end
    end
    set_wing_U!(c)
end

function mul_skiplastindex!(
    c::Gaugefields_3D_nowing{NC},
    a::T1,
    b::T2,
) where {NC,T1<:Abstractfields,T2<:Abstractfields}
    #@assert NC != 2 && NC != 3 "This function is for NC != 2,3"
    NT = c.NT
    NY = c.NY
    NX = c.NX
    it = 1
    for iy = 1:NY
        for ix = 1:NX
            for k2 = 1:NC
                for k1 = 1:NC
                    c[k1, k2, ix, iy, it] = 0

                    @simd for k3 = 1:NC
                        c[k1, k2, ix, iy, it] +=
                                a[k1, k3, ix, iy, it] * b[k3, k2, ix, iy, it]
                    end
                end
            end
        end
    end
    #end
end


#=
function gramschmidt!(v)
    n = size(v)[1]
    for i=1:n
        @simd for j=1:i-1
            v[:,i] = v[:,i] - v[:,j]'*v[:,i]*v[:,j]
        end
        v[:,i] = v[:,i]/norm(v[:,i])
    end
end
=#


function normalize_U!(u::Gaugefields_3D_nowing{NC}) where {NC}
    NX = u.NX
    NY = u.NY
    NT = u.NT

    for it = 1:NT
        for iy = 1:NY
            @simd for ix = 1:NX
                A = u[:, :, ix, iy, it]
                gramschmidt!(A)
                u[:, :, ix, iy, it] = A[:, :]
            end
        end
    end

end


"""
    b = (lambda_k/2)*a
    lambda_k : SUN matrices. k=1, ...
"""
function lambda_k_mul!(
    b::Gaugefields_3D_nowing{NC},
    a::Gaugefields_3D_nowing{NC},
    k,generator
) where NC
    NX = a.NX
    NY = a.NY
    NT = a.NT
    #NV = a.NV
    #NC = generator.NC
    matrix = generator.generator[k]
    for it = 1:NT
        for iy = 1:NY
            @inbounds @simd for ix = 1:NX
                for k2=1:NC
                    for k1=1:NC
                        b[k1,k2,ix,iy,it] = 0
                        @simd for l=1:NC
                            b[k1,k2,ix,iy,it] += matrix[k1,l]*a[l,k2,ix,iy,it]/2
                        end
                    end
                end
            end
        end
    end


    return
end



function calculate_gdg!(
    c::Gaugefields_3D_nowing{NC},
    a::Gaugefields_3D_nowing{NC},
    ν::Integer,
    temp;
    cc=false,
) where NC
    NX = a.NX
    NY = a.NY
    NT = a.NT

    b = temp
    for it = 1:NT
        for iy = 1:NY
            for ix = 1:NX
                for k1 = 1:NC
                    @inbounds @simd for k2 = 1:NC
                        b[k1,k2,ix,iy,it] = a[k1,k2,ix,iy,it]
                    end
                end
            end
        end
    end

    clear_U!(c)
    #eye = Matrix(LinearAlgebra.I,NC,NC)
    for it = 1:NT
        t = it
        t_f = it
        t_b = it
        if ν==3
            t_f += 1
            t_b += - 1
        elseif ν==-3
            t_f += - 1
            t_b += 1
        end
        if t_f == (NT+1)
            t_f = 1
        elseif t_b == 0
            t_b = NT
        elseif t_b == (NT+1)
            t_b = 1
        elseif t_f == 0
            t_f = NT
        end
        for iy = 1:NY
            y = iy
            y_f = iy
            y_b = iy
            if ν==2
                y_f += 1
                y_b += - 1
            elseif ν==-2
                y_f += - 1
                y_b += 1
            end
            if y_f == (NY+1)
                y_f = 1
            elseif y_b == 0
                y_b = NY
            elseif y_b == (NY+1)
                y_b = 1
            elseif y_f == 0
                y_f = NY
            end
            @inbounds @simd for ix = 1:NX
                x = ix
                x_f = ix
                x_b = ix
                if ν==1
                    x_f += 1
                    x_b += - 1
                elseif ν==-1
                    x_f += - 1
                    x_b += 1
                end
                if x_f == (NX+1)
                    x_f = 1
                elseif x_b == 0
                    x_b = NX
                elseif x_b == (NX+1)
                    x_b = 1
                elseif x_f == 0
                    x_f = NX
                end
                if !cc
                    c[:,:,x,y,t] =
                        0.5 * b[:,:,x,y,t]' * b[:,:,x_f,y_f,t_f] -
                        0.5 * b[:,:,x,y,t]' * b[:,:,x_b,y_b,t_b]
                else
                    c[:,:,x,y,t] =
                        0.5 * b[:,:,x_f,y_f,t_f]' * b[:,:,x,y,t] -
                        0.5 * b[:,:,x_b,y_b,t_b]' * b[:,:,x,y,t]
                end
            end
        end
    end
    return c
end
function calculate_gdg!(
    c::Gaugefields_3D_nowing{NC},
    a::Gaugefields_3D_nowing{NC},
    ν::Integer,
    η,
    temp;
    cc=false,
) where NC
    NX = a.NX
    NY = a.NY
    NT = a.NT

    b = temp
    for it = 1:NT
        for iy = 1:NY
            for ix = 1:NX
                for k1 = 1:NC
                    @inbounds @simd for k2 = 1:NC
                        b[k1,k2,ix,iy,it] = a[k1,k2,ix,iy,it]
                    end
                end
            end
        end
    end

    clear_U!(c)
    #eye = Matrix(LinearAlgebra.I,NC,NC)
    for it = 1:NT
        t = it
        t_f = it
        t_f2 = it
        t_b = it
        t_b2 = it
        if ν==3
            t_f += 1
            t_f2 += 2
            t_b += - 1
            t_b2 += - 2
        elseif ν==-3
            t_f += - 1
            t_f2 += - 2
            t_b += 1
            t_b2 += 2
        end
        if t_f == (NT+1)
            t_f = 1
        elseif t_b == 0
            t_b = NT
        elseif t_b == (NT+1)
            t_b = 1
        elseif t_f == 0
            t_f = NT
        end
        if t_f2 == (NT+1)
            t_f2 = 1
        elseif t_b2 == 0
            t_b2 = NT
        elseif t_b2 == (NT+1)
            t_b2 = 1
        elseif t_f2 == 0
            t_f2 = NT
        end
        if t_f2 == (NT+2)
            t_f2 = 2
        elseif t_b2 == -1
            t_b2 = NT-1
        elseif t_b2 == (NT+2)
            t_b2 = 2
        elseif t_f2 == -1
            t_f2 = NT-1
        end
        for iy = 1:NY
            y = iy
            y_f = iy
            y_f2 = iy
            y_b = iy
            y_b2 = iy
            if ν==2
                y_f += 1
                y_f2 += 2
                y_b += - 1
                y_b2 += - 2
            elseif ν==-2
                y_f += - 1
                y_f2 += - 2
                y_b += 1
                y_b2 += 2
            end
            if y_f == (NY+1)
                y_f = 1
            elseif y_b == 0
                y_b = NY
            elseif y_b == (NY+1)
                y_b = 1
            elseif y_f == 0
                y_f = NY
            end
            if y_f2 == (NY+1)
                y_f2 = 1
            elseif y_b2 == 0
                y_b2 = NY
            elseif y_b2 == (NY+1)
                y_b2 = 1
            elseif y_f2 == 0
                y_f2 = NY
            end
            if y_f2 == (NY+2)
                y_f2 = 2
            elseif y_b2 == -1
                y_b2 = NY-1
            elseif y_b2 == (NY+2)
                y_b2 = 2
            elseif y_f2 == -1
                y_f2 = NY-1
            end
            @inbounds @simd for ix = 1:NX
                x = ix
                x_f = ix
                x_f2 = ix
                x_b = ix
                x_b2 = ix
                if ν==1
                    x_f += 1
                    x_f2 += 2
                    x_b += - 1
                    x_b2 += - 2
                elseif ν==-1
                    x_f += - 1
                    x_f2 += - 2
                    x_b += 1
                    x_b2 += 2
                end
                if x_f == (NX+1)
                    x_f = 1
                elseif x_b == 0
                    x_b = NX
                elseif x_b == (NX+1)
                    x_b = 1
                elseif x_f == 0
                    x_f = NX
                end
                if x_f2 == (NX+1)
                    x_f2 = 1
                elseif x_b2 == 0
                    x_b2 = NX
                elseif x_b2 == (NX+1)
                    x_b2 = 1
                elseif x_f2 == 0
                    x_f2 = NX
                end
                if x_f2 == (NX+2)
                    x_f2 = 2
                elseif x_b2 == -1
                    x_b2 = NX-1
                elseif x_b2 == (NX+2)
                    x_b2 = 2
                elseif x_f2 == -1
                    x_f2 = NX-1
                end
                if !cc
                    c[:,:,x,y,t] =
                        0.5 * b[:,:,x,y,t]' *
                        (b[:,:,x_f,y_f,t_f] - b[:,:,x_b,y_b,t_b]
                         - (η/6) * (b[:,:,x_f2,y_f2,t_f2]
                                      - 2b[:,:,x_f,y_f,t_f]
                                      + 2b[:,:,x_b,y_b,t_b]
                                      - b[:,:,x_b2,y_b2,t_b2]
                                      )
                         )
                else
                    c[:,:,x,y,t] =
                        (b[:,:,x_f,y_f,t_f] - b[:,:,x_b,y_b,t_b]
                         - (η/6) * (b[:,:,x_f2,y_f2,t_f2]
                                      - 2b[:,:,x_f,y_f,t_f]
                                      + 2b[:,:,x_b,y_b,t_b]
                                      - b[:,:,x_b2,y_b2,t_b2]
                                      )
                         )' *
                             0.5 * b[:,:,x,y,t]
                end
            end
        end
    end
    return c
end

function calculate_gdg_conj!(
    c::Gaugefields_3D_nowing{NC},
    a::Gaugefields_3D_nowing{NC},
    ν::Integer,
    temps
) where NC
    b1 = temps[1]
    calculate_gdg!(b1,a,ν,temps[2],cc=false)

    clear_U!(c)
    NX = a.NX
    NY = a.NY
    NT = a.NT
    for it = 1:NT
        for iy = 1:NY
            @inbounds @simd for ix = 1:NX
                c[:,:,ix,iy,it] = b1[:,:,ix,iy,it] - b1[:,:,ix,iy,it]'
            end
        end
    end
    return c
end
function calculate_gdg_conj!(
    c::Gaugefields_3D_nowing{NC},
    a::Gaugefields_3D_nowing{NC},
    ν::Integer,
    η,
    temps
) where NC
    b1 = temps[1]
    calculate_gdg!(b1,a,ν,η,temps[2],cc=false)

    clear_U!(c)
    NX = a.NX
    NY = a.NY
    NT = a.NT
    for it = 1:NT
        for iy = 1:NY
            @inbounds @simd for ix = 1:NX
                c[:,:,ix,iy,it] = b1[:,:,ix,iy,it] - b1[:,:,ix,iy,it]'
            end
        end
    end
    return c
end

function calculate_gdg_action(
    a::Gaugefields_3D_nowing{NC},
    ν::Integer,
    temps
) where NC
    NX = a.NX
    NY = a.NY
    NT = a.NT

    b = temps[1]
    calculate_gdg_conj!(b,a,ν,[temps[2], temps[3]])

    c = 0.0 + 0.0im
    for it = 1:NT
        for iy = 1:NY
            for ix = 1:NX
                for k = 1:NC
                    @inbounds @simd for k3 = 1:NC
                        c += b[k,k3,ix,iy,it]*b[k3,k,ix,iy,it]
                    end
                end
            end
        end
    end
    return c
end
function calculate_gdg_action(
    a::Gaugefields_3D_nowing{NC},
    ν::Integer,
    η,
    temps
) where NC
    NX = a.NX
    NY = a.NY
    NT = a.NT

    b = temps[1]
    calculate_gdg_conj!(b,a,ν,η,[temps[2], temps[3]])

    c = 0.0 + 0.0im
    for it = 1:NT
        for iy = 1:NY
            for ix = 1:NX
                for k = 1:NC
                    @inbounds @simd for k3 = 1:NC
                        c += b[k,k3,ix,iy,it]*b[k3,k,ix,iy,it]
                    end
                end
            end
        end
    end
    return c
end

function calculate_gdg_actiondensity(
    a::Gaugefields_3D_nowing{NC},
    ix::Integer,
    iy::Integer,
    it::Integer,
    ν::Integer,
    temps
) where NC
    NX = a.NX
    NY = a.NY
    NT = a.NT

    b = temps[1]
    calculate_gdg_conj!(b,a,ν,[temps[2], temps[3]])

    c = 0.0 + 0.0im
    for k = 1:NC
        @inbounds @simd for k3 = 1:NC
            c += b[k,k3,ix,iy,it]*b[k3,k,ix,iy,it]
        end
    end
    return c
end
function calculate_gdg_actiondensity(
    a::Gaugefields_3D_nowing{NC},
    ix::Integer,
    iy::Integer,
    it::Integer,
    ν::Integer,
    η,
    temps
) where NC
    NX = a.NX
    NY = a.NY
    NT = a.NT

    b = temps[1]
    calculate_gdg_conj!(b,a,ν,η,[temps[2], temps[3]])

    c = 0.0 + 0.0im
    for k = 1:NC
        @inbounds @simd for k3 = 1:NC
            c += b[k,k3,ix,iy,it]*b[k3,k,ix,iy,it]
        end
    end
    return c
end

function calculate_gdg_wind(
    a::Gaugefields_3D_nowing{NC},
    temp,
    temps
) where NC
    NX = a.NX
    NY = a.NY
    NT = a.NT
    w = 0.0

    b = temp
    b1 = temps[1]
    b2 = temps[2]
    b3 = temps[3]

    calculate_gdg_conj!(b1,a,1,[temps[4],temps[5]])
    calculate_gdg_conj!(b2,a,2,[temps[4],temps[5]])
    calculate_gdg_conj!(b3,a,3,[temps[4],temps[5]])

    clear_U!(b)
    for it = 1:NT
        for iy = 1:NY
            for ix = 1:NX
                b[:,:,ix,iy,it] =
                    b1[:,:,ix,iy,it] *
                    (b2[:,:,ix,iy,it]*b3[:,:,ix,iy,it] -
                     b3[:,:,ix,iy,it]*b2[:,:,ix,iy,it])
            end
        end
    end
    for it = 1:NT
        for iy = 1:NY
            for ix = 1:NX
                for k = 1:NC
                    w += b[k,k,ix,iy,it]
                end
            end
        end
    end
    return 3 * real(w)
end
function calculate_gdg_wind(
    a::Gaugefields_3D_nowing{NC},
    η,
    temp,
    temps
) where NC
    NX = a.NX
    NY = a.NY
    NT = a.NT
    w = 0.0

    b = temp
    b1 = temps[1]
    b2 = temps[2]
    b3 = temps[3]

    calculate_gdg_conj!(b1,a,1,η,[temps[4],temps[5]])
    calculate_gdg_conj!(b2,a,2,η,[temps[4],temps[5]])
    calculate_gdg_conj!(b3,a,3,η,[temps[4],temps[5]])

    clear_U!(b)
    for it = 1:NT
        for iy = 1:NY
            for ix = 1:NX
                b[:,:,ix,iy,it] =
                    b1[:,:,ix,iy,it] *
                    (b2[:,:,ix,iy,it]*b3[:,:,ix,iy,it] -
                     b3[:,:,ix,iy,it]*b2[:,:,ix,iy,it])
            end
        end
    end
    for it = 1:NT
        for iy = 1:NY
            for ix = 1:NX
                for k = 1:NC
                    w += b[k,k,ix,iy,it]
                end
            end
        end
    end
    return 3 * real(w)
end

function calculate_g_gdg_gdg_g!(
    c::Gaugefields_3D_nowing{NC},
    a::Gaugefields_3D_nowing{NC},
    ν::Integer,
    temps;
    cc=false,
) where NC
    NX = a.NX
    NY = a.NY
    NT = a.NT

    b = temps[1]

    calculate_gdg!(b,a,ν,temps[2],cc=cc)

    clear_U!(c)
    for it = 1:NT
        for iy = 1:NY
            @inbounds @simd for ix = 1:NX
                c[:,:,ix,iy,it] =
                    a[:,:,ix,iy,it] * b[:,:,ix,iy,it] *
                    b[:,:,ix,iy,it] * a[:,:,ix,iy,it]'
            end
        end
    end
    for it = 1:NT
        t = it
        t_f = it
        t_b = it
        if ν==3
            t_f += 1
            t_b += - 1
        elseif ν==-3
            t_f += - 1
            t_b += 1
        end
        if t_f == (NT+1)
            t_f = 1
        elseif t_b == 0
            t_b = NT
        elseif t_b == (NT+1)
            t_b = 1
        elseif t_f == 0
            t_f = NT
        end
        for iy = 1:NY
            y = iy
            y_f = iy
            y_b = iy
            if ν==2
                y_f += 1
                y_b += - 1
            elseif ν==-2
                y_f += - 1
                y_b += 1
            end
            if y_f == (NY+1)
                y_f = 1
            elseif y_b == 0
                y_b = NY
            elseif y_b == (NY+1)
                y_b = 1
            elseif y_f == 0
                y_f = NY
            end
            @inbounds @simd for ix = 1:NX
                x = ix
                x_f = ix
                x_b = ix
                if ν==1
                    x_f += 1
                    x_b += - 1
                elseif ν==-1
                    x_f += - 1
                    x_b += 1
                end
                if x_f == (NX+1)
                    x_f = 1
                elseif x_b == 0
                    x_b = NX
                elseif x_b == (NX+1)
                    x_b = 1
                elseif x_f == 0
                    x_f = NX
                end
                c[:,:,x,y,t] +=
                    0.5 * a[:,:,x,y,t] * b[:,:,x_f,y_f,t_f] * a[:,:,x_f,y_f,t_f]' -
                    0.5 * a[:,:,x,y,t] * b[:,:,x_b,y_b,t_b] * a[:,:,x_b,y_b,t_b]'
            end
        end
    end
    for it = 1:NT
        t = it
        t_f = it
        t_b = it
        if ν==3
            t_f += 2
            t_b += - 2
        elseif ν==-3
            t_f += - 2
            t_b += 2
        end
        if t_f == (NT+1)
            t_f = 1
        elseif t_b == 0
            t_b = NT
        elseif t_b == (NT+1)
            t_b = 1
        elseif t_f == 0
            t_f = NT
        end
        if t_f == (NT+2)
            t_f = 2
        elseif t_b == -1
            t_b = NT-1
        elseif t_b == (NT+2)
            t_b = 2
        elseif t_f == -1
            t_f = NT-1
        end
        for iy = 1:NY
            y = iy
            y_f = iy
            y_b = iy
            if ν==2
                y_f += 2
                y_b += - 2
            elseif ν==-2
                y_f += - 2
                y_b += 2
            end
            if y_f == (NY+1)
                y_f = 1
            elseif y_b == 0
                y_b = NY
            elseif y_b == (NY+1)
                y_b = 1
            elseif y_f == 0
                y_f = NY
            end
            if y_f == (NY+2)
                y_f = 2
            elseif y_b == -1
                y_b = NY-1
            elseif y_b == (NY+2)
                y_b = 2
            elseif y_f == -1
                y_f = NY-1
            end
            @inbounds @simd for ix = 1:NX
                x = ix
                x_f = ix
                x_b = ix
                if ν==1
                    x_f += 2
                    x_b += - 2
                elseif ν==-1
                    x_f += - 2
                    x_b += 2
                end
                if x_f == (NX+1)
                    x_f = 1
                elseif x_b == 0
                    x_b = NX
                elseif x_b == (NX+1)
                    x_b = 1
                elseif x_f == 0
                    x_f = NX
                end
                if x_f == (NX+2)
                    x_f = 2
                elseif x_b == -1
                    x_b = NX-1
                elseif x_b == (NX+2)
                    x_b = 2
                elseif x_f == -1
                    x_f = NX-1
                end
                c[:,:,x,y,t] +=
                    - 0.25 * a[:,:,x,y,t] * a[:,:,x_f,y_f,t_f]' -
                    0.25 * a[:,:,x,y,t] * a[:,:,x_b,y_b,t_b]'
            end
        end
    end
    return
end
function calculate_g_gdg_gdg_g!(
    c::Gaugefields_3D_nowing{NC},
    a::Gaugefields_3D_nowing{NC},
    ν::Integer,
    η,
    temps;
    cc=false,
) where NC
    NX = a.NX
    NY = a.NY
    NT = a.NT

    b = temps[1]

    calculate_gdg!(b,a,ν,η,temps[2],cc=cc)

    clear_U!(c)
    for it = 1:NT
        for iy = 1:NY
            @inbounds @simd for ix = 1:NX
                c[:,:,ix,iy,it] =
                    a[:,:,ix,iy,it] * b[:,:,ix,iy,it] *
                    (b[:,:,ix,iy,it]-b[:,:,ix,iy,it]') * a[:,:,ix,iy,it]'
            end
        end
    end
    for it = 1:NT
        t = it
        t_f = it
        t_b = it
        if ν==3
            t_f += 1
            t_b += - 1
        elseif ν==-3
            t_f += - 1
            t_b += 1
        end
        if t_f == (NT+1)
            t_f = 1
        elseif t_b == 0
            t_b = NT
        elseif t_b == (NT+1)
            t_b = 1
        elseif t_f == 0
            t_f = NT
        end
        for iy = 1:NY
            y = iy
            y_f = iy
            y_b = iy
            if ν==2
                y_f += 1
                y_b += - 1
            elseif ν==-2
                y_f += - 1
                y_b += 1
            end
            if y_f == (NY+1)
                y_f = 1
            elseif y_b == 0
                y_b = NY
            elseif y_b == (NY+1)
                y_b = 1
            elseif y_f == 0
                y_f = NY
            end
            @inbounds @simd for ix = 1:NX
                x = ix
                x_f = ix
                x_b = ix
                if ν==1
                    x_f += 1
                    x_b += - 1
                elseif ν==-1
                    x_f += - 1
                    x_b += 1
                end
                if x_f == (NX+1)
                    x_f = 1
                elseif x_b == 0
                    x_b = NX
                elseif x_b == (NX+1)
                    x_b = 1
                elseif x_f == 0
                    x_f = NX
                end
                c[:,:,x,y,t] +=
                    0.5 * (1+η/3) * a[:,:,x,y,t] *
                    (b[:,:,x_f,y_f,t_f]-b[:,:,x_f,y_f,t_f]') * a[:,:,x_f,y_f,t_f]' -
                    0.5 * (1+η/3) * a[:,:,x,y,t] *
                    (b[:,:,x_b,y_b,t_b]-b[:,:,x_b,y_b,t_b]') * a[:,:,x_b,y_b,t_b]'
            end
        end
    end
    for it = 1:NT
        t = it
        t_f = it
        t_b = it
        if ν==3
            t_f += 2
            t_b += - 2
        elseif ν==-3
            t_f += - 2
            t_b += 2
        end
        if t_f == (NT+1)
            t_f = 1
        elseif t_b == 0
            t_b = NT
        elseif t_b == (NT+1)
            t_b = 1
        elseif t_f == 0
            t_f = NT
        end
        if t_f == (NT+2)
            t_f = 2
        elseif t_b == -1
            t_b = NT-1
        elseif t_b == (NT+2)
            t_b = 2
        elseif t_f == -1
            t_f = NT-1
        end
        for iy = 1:NY
            y = iy
            y_f = iy
            y_b = iy
            if ν==2
                y_f += 2
                y_b += - 2
            elseif ν==-2
                y_f += - 2
                y_b += 2
            end
            if y_f == (NY+1)
                y_f = 1
            elseif y_b == 0
                y_b = NY
            elseif y_b == (NY+1)
                y_b = 1
            elseif y_f == 0
                y_f = NY
            end
            if y_f == (NY+2)
                y_f = 2
            elseif y_b == -1
                y_b = NY-1
            elseif y_b == (NY+2)
                y_b = 2
            elseif y_f == -1
                y_f = NY-1
            end
            @inbounds @simd for ix = 1:NX
                x = ix
                x_f = ix
                x_b = ix
                if ν==1
                    x_f += 2
                    x_b += - 2
                elseif ν==-1
                    x_f += - 2
                    x_b += 2
                end
                if x_f == (NX+1)
                    x_f = 1
                elseif x_b == 0
                    x_b = NX
                elseif x_b == (NX+1)
                    x_b = 1
                elseif x_f == 0
                    x_f = NX
                end
                if x_f == (NX+2)
                    x_f = 2
                elseif x_b == -1
                    x_b = NX-1
                elseif x_b == (NX+2)
                    x_b = 2
                elseif x_f == -1
                    x_f = NX-1
                end
                c[:,:,x,y,t] +=
                    - 0.5 * (η/6) * a[:,:,x,y,t] *
                    (b[:,:,x_f,y_f,t_f]-b[:,:,x_f,y_f,t_f]') * a[:,:,x_f,y_f,t_f]' +
                    0.5 * (η/6) * a[:,:,x,y,t] *
                    (b[:,:,x_b,y_b,t_b]-b[:,:,x_b,y_b,t_b]') * a[:,:,x_b,y_b,t_b]'
            end
        end
    end
    return
end

function test_map_U2_theta(x,y,t,m)
    theta = [0.0 0.0 0.0 0.0]

    theta[1] = m + (cos(x)-1) + (cos(y)-1) + (cos(t)-1)
    theta[2] = sin(x)
    theta[3] = sin(y)
    theta[4] = sin(t)
    
    return theta / norm(theta)
end

function test_map_U2_g(x,y,t,m)
    eye = [1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im]
    sigma1 = [0.0+0.0im 1.0+0.0im; 1.0+0.0im 0.0+0.0im]
    sigma2 = [0.0+0.0im 0.0-1.0im; 0.0+1.0im 0.0+0.0im]
    sigma3 = [1.0+0.0im 0.0+0.0im; 0.0+0.0im (-1.0)+0.0im]

    theta = test_map_U2_theta(x,y,t,m)

    return theta[1]*eye + im*theta[2]*sigma1 + im*theta[3]*sigma2 + im*theta[4]*sigma3
end

function TestmapGauges_3D(NC, m, NX, NY, NT; verbose_level = 2)
    return test_map_U2Gaugefields_3D_nowing(NC, NX, NY, NT, m, verbose_level = verbose_level)
end

function test_map_U2Gaugefields_3D_nowing(NC, NX, NY, NT, m; verbose_level = 2)
    @assert NC==2 "NC should be 2."
    U = Gaugefields_3D_nowing(NC, NX, NY, NT, verbose_level = verbose_level)

    for it = 1:NT
        for iy = 1:NY
            @inbounds @simd for ix = 1:NX
                x = - pi + (ix-1) * (2pi) / NX
                y = - pi + (iy-1) * (2pi) / NY
                t = - pi + (it-1) * (2pi) / NT
                U[:,:, ix, iy, it] = test_map_U2_g(x,y,t,m)
            end
        end
    end
    set_wing_U!
    return U
end

function test_map_Random_U2_theta(x,y,t,m,rng,eps)
    theta = [0.0 0.0 0.0 0.0]
    zeta  = rand(rng, Float64, 4) - 0.5ones(4)

    theta[1] = m + (cos(x)-1) + (cos(y)-1) + (cos(t)-1) + eps*zeta[1]
    theta[2] = sin(x) + eps*zeta[2]
    theta[3] = sin(y) + eps*zeta[3]
    theta[4] = sin(t) + eps*zeta[4]
    
    return theta / norm(theta)
end

function test_map_Random_U2_g(x,y,t,m,rng,eps)
    eye = [1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im]
    sigma1 = [0.0+0.0im 1.0+0.0im; 1.0+0.0im 0.0+0.0im]
    sigma2 = [0.0+0.0im 0.0-1.0im; 0.0+1.0im 0.0+0.0im]
    sigma3 = [1.0+0.0im 0.0+0.0im; 0.0+0.0im (-1.0)+0.0im]

    theta = test_map_Random_U2_theta(x,y,t,m,rng,eps)

    return theta[1]*eye + im*theta[2]*sigma1 + im*theta[3]*sigma2 + im*theta[4]*sigma3
end

function TestmapRandomGauges_3D(NC, m, NX, NY, NT; verbose_level = 2, randomnumber = "Random", reps = 0.1)
    return test_map_Random_U2Gaugefields_3D_nowing(NC, NX, NY, NT, m, verbose_level = verbose_level, randomnumber = randomnumber, reps = reps)
end

function test_map_Random_U2Gaugefields_3D_nowing(NC, NX, NY, NT, m; verbose_level = 2, randomnumber = "Random", reps = 0.1)
    @assert NC==2 "NC should be 2."
    U = Gaugefields_3D_nowing(NC, NX, NY, NT, verbose_level = verbose_level)
    if randomnumber == "Random"
        rng = MersenneTwister()
    elseif randomnumber == "Reproducible"
        rng = StableRNG(123)
    else
        error(
            "randomnumber should be \"Random\" or \"Reproducible\". Now randomnumber = $randomnumber",
        )
    end

    for it = 1:NT
        for iy = 1:NY
            @inbounds @simd for ix = 1:NX
                x = - pi + (ix-1) * (2pi) / NX
                y = - pi + (iy-1) * (2pi) / NY
                t = - pi + (it-1) * (2pi) / NT
                U[:,:, ix, iy, it] = test_map_Random_U2_g(x,y,t,m,rng,reps)
            end
        end
    end
    set_wing_U!
    return U
end

function test_map_U2n_theta(x,y,t,m,n,rng)
    theta = zeros(Float64, 4, n, n)

    r = rand(rng, Float64, n, n)
    s = rand(rng, Float64, n, n)
    theta[1,:,:] = ( m - 3 + cos(x) + cos(y) + cos(t) ) * r + ( m - 3 + cos(2x) + cos(2y) + cos(2t) ) * s
    
    r = rand(rng, Float64, n, n)
    s = rand(rng, Float64, n, n)
    theta[2,:,:] = sin(x) * r + sin(2x) * s
    r = rand(rng, Float64, n, n)
    s = rand(rng, Float64, n, n)
    theta[3,:,:] = sin(y) * r + sin(2y) * s
    r = rand(rng, Float64, n, n)
    s = rand(rng, Float64, n, n)
    theta[4,:,:] = sin(t) * r + sin(2t) * s
    
    return theta
end

function test_map_U2n_g(x,y,t,m,n,rng)
    eye = [1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im]
    sigma1 = [0.0+0.0im 1.0+0.0im; 1.0+0.0im 0.0+0.0im]
    sigma2 = [0.0+0.0im 0.0-1.0im; 0.0+1.0im 0.0+0.0im]
    sigma3 = [1.0+0.0im 0.0+0.0im; 0.0+0.0im (-1.0)+0.0im]

    theta = test_map_U2n_theta(x,y,t,m,n,rng)

    q = zeros(ComplexF64, 2*n, 2*n)
    for i = 1:2*n
        for j = 1:2*n
            i_R = (i-1) % n + 1
            j_R = (j-1) % n + 1
            i_P = div(i-1, n) + 1
            j_P = div(j-1, n) + 1
            q[i,j] = theta[1,i_R,j_R]*eye[i_P,j_P] +
                im*theta[2,i_R,j_R]*sigma1[i_P,j_P] +
                im*theta[3,i_R,j_R]*sigma2[i_P,j_P] +
                im*theta[4,i_R,j_R]*sigma3[i_P,j_P]
        end
    end
    F = svd(q)
    return F.U * F.Vt
end

function TestmapGauges_3D_U2n(NC, m, n, NX, NY, NT; verbose_level = 2, randomnumber = "Random")
    return test_map_U2nGaugefields_3D_nowing(NC, NX, NY, NT, m, n, verbose_level = verbose_level, randomnumber = randomnumber)
end

function test_map_U2nGaugefields_3D_nowing(NC, NX, NY, NT, m, n; verbose_level = 2, randomnumber = "Random")
    @assert NC==2 "NC should be 2."
    U = Gaugefields_3D_nowing(NC*n, NX, NY, NT, verbose_level = verbose_level)
    if randomnumber == "Random"
        rng = MersenneTwister()
    elseif randomnumber == "Reproducible"
        rng = StableRNG(123)
    else
        error(
            "randomnumber should be \"Random\" or \"Reproducible\". Now randomnumber = $randomnumber",
        )
    end

    for it = 1:NT
        for iy = 1:NY
            @inbounds @simd for ix = 1:NX
                x = - pi + (ix-1) * (2pi) / NX
                y = - pi + (iy-1) * (2pi) / NY
                t = - pi + (it-1) * (2pi) / NT
                U[:,:, ix, iy, it] = test_map_U2n_g(x,y,t,m,n,rng)
            end
        end
    end
    set_wing_U!
    return U
end



function random_matrix_r(n,randmat)
    if n==1
        r0 = [0.876608492574193]
        r1 = [0.8400934725278035]
        r2 = [0.003912847405306286]
        r3 = [0.5219920025343154]
    elseif n==2
        r0 = [0.876608492574193 0.5219642502018771; 0.08622342695413243 0.3779129543048858]
        r1 = [0.8400934725278035 0.005292539995173762; 0.20381352056978863 0.9537947095189847]
        r2 = [0.003912847405306286 0.6481849103764872; 0.32747967430726743 0.42958562996150196]
        r3 = [0.5219920025343154 0.09973719932020075; 0.9800628057498066 0.4856043179437335]
    elseif n==3
        r0 = [0.876608492574193 0.5219642502018771 0.08622342695413243; 0.3779129543048858 0.011644572969246703 0.9272661633421619; 0.5437567669510681 0.47933167047242464 0.24534922778070478]
        r1 = [0.8400934725278035 0.005292539995173762 0.20381352056978863; 0.9537947095189847 0.41508206440865214 0.06390418562058486; 0.22123954438621207 0.29232260658369924 0.7070486600861496]
        r2 = [0.003912847405306286 0.6481849103764872 0.32747967430726743; 0.42958562996150196 0.44356528910750637 0.1523878038992892; 0.6707317883583468 0.6664470284756423 0.9739672993869939]
        r3 = [0.5219920025343154 0.09973719932020075 0.9800628057498066; 0.4856043179437335 0.2395294527440972 0.8179046610218443; 0.7923016581440165 0.27830885490902335 0.6121518504536243]
    elseif n==4
        r0 = [0.876608492574193 0.5219642502018771 0.08622342695413243 0.3779129543048858; 0.011644572969246703 0.9272661633421619 0.5437567669510681 0.47933167047242464; 0.24534922778070478 0.7598960015615879 0.9849929975443044 0.21704512156333866; 0.4590171869250881 0.884729168704953 0.5838542859901488 0.2639731675000303]
        r1 = [0.8400934725278035 0.005292539995173762 0.20381352056978863 0.9537947095189847; 0.41508206440865214 0.06390418562058486 0.22123954438621207 0.29232260658369924; 0.7070486600861496 0.7446005126738047 0.7520709207435481 0.9650767773527269; 0.5955762345784386 0.357155999761672 0.09745849794680561 0.5096183728199861]
        r2 = [0.003912847405306286 0.6481849103764872 0.32747967430726743 0.42958562996150196; 0.44356528910750637 0.1523878038992892 0.6707317883583468 0.6664470284756423; 0.9739672993869939 0.2887470155421499 0.8503720615519674 0.7490030016787093; 0.7420527205495799 0.3554331249694711 0.7489232428075163 0.916096175750742]
        r3 = [0.5219920025343154 0.09973719932020075 0.9800628057498066 0.4856043179437335; 0.2395294527440972 0.8179046610218443 0.7923016581440165 0.27830885490902335; 0.6121518504536243 0.9875247884241343 0.8331961863493131 0.04447342634563767; 0.08795814325876505 0.25058597394645354 0.3849220301540264 0.6181497500484561]
    end

    if randmat=="all"
        return r0, r1, r2, r3
    end
    r = zeros(Float64, 4, n, n)
    for i = 1:4
        if randmat[i]=='0'
            r[i,:,:] += r0
        elseif randmat[i]=='1'
            r[i,:,:] += r1
        elseif randmat[i]=='2'
            r[i,:,:] += r2
        elseif randmat[i]=='3'
            r[i,:,:] += r3
        end
    end
    return r[1,:,:], r[2,:,:], r[3,:,:], r[4,:,:]
end

function random_matrix_s(n,randmat)
    if n==1
        r0 = [0.5619536046671718]
        r1 = [0.1470991416512073]
        r2 = [0.27832888711501824]
        r3 = [0.051238502478856196]
    elseif n==2
        r0 = [0.5619536046671718 0.030508304060367752; 0.4164838731963958 0.9574107786119304]
        r1 = [0.1470991416512073 0.3511691375514294; 0.03385647214299836 0.33241260057484645]
        r2 = [0.27832888711501824 0.5040670417441588; 0.31173813322508503 0.6617628514667968]
        r3 = [0.051238502478856196 0.29787803608015273; 0.42172374852711547 0.08237998583569683]
    elseif n==3
        r0 = [0.5619536046671718 0.030508304060367752 0.4164838731963958; 0.9574107786119304 0.9142817338556259 0.9740782644674564; 0.24813852615858556 0.31804225418531873 0.9460003473615142]
        r1 = [0.1470991416512073 0.3511691375514294 0.03385647214299836; 0.33241260057484645 0.9720175627103507 0.9680227484586859; 0.38047233748776654 0.9665772953247509 0.8649037288619748]
        r2 = [0.27832888711501824 0.5040670417441588 0.31173813322508503; 0.6617628514667968 0.4419535717259877 0.06380382547441488; 0.3076699760916697 0.9157378960865203 0.7388326027583885]
        r3 = [0.051238502478856196 0.29787803608015273 0.42172374852711547; 0.08237998583569683 0.32092443266221493 0.670261436046625; 0.0735133633533871 0.28419671665120005 0.7243070233647702]
    elseif n==4
        r0 = [0.5619536046671718 0.030508304060367752 0.4164838731963958 0.9574107786119304; 0.9142817338556259 0.9740782644674564 0.24813852615858556 0.31804225418531873; 0.9460003473615142 0.41456288196777313 0.3741834018772541 0.6430147250746139; 0.5069513138049517 0.30412265338932065 0.6348017834131661 0.2728514221540965]
        r1 = [0.1470991416512073 0.3511691375514294 0.03385647214299836 0.33241260057484645; 0.9720175627103507 0.9680227484586859 0.38047233748776654 0.9665772953247509; 0.8649037288619748 0.03387641745688175 0.15285327197958165 0.8792485819486102; 0.947237182251994 0.2925446566269678 0.42523830174427313 0.698323741141224]
        r2 = [0.27832888711501824 0.5040670417441588 0.31173813322508503 0.6617628514667968; 0.4419535717259877 0.06380382547441488 0.3076699760916697 0.9157378960865203; 0.7388326027583885 0.8241952684729963 0.002525456197198128 0.8194857191074327; 0.10643543921829024 0.6173802137393589 0.8866707281650839 0.04858806127944981]
        r3 = [0.051238502478856196 0.29787803608015273 0.42172374852711547 0.08237998583569683; 0.32092443266221493 0.670261436046625 0.0735133633533871 0.28419671665120005; 0.7243070233647702 0.33873433411608933 0.7381987002113375 0.5330474061296473; 0.9569587466043488 0.6872238158358952 0.33421745032481653 0.9043199774699715]
    end

    if randmat=="all"
        return r0, r1, r2, r3
    end
    r = zeros(Float64, 4, n, n)
    for i = 1:4
        if randmat[i+5]=='0'
            r[i,:,:] += r0
        elseif randmat[i+5]=='1'
            r[i,:,:] += r1
        elseif randmat[i+5]=='2'
            r[i,:,:] += r2
        elseif randmat[i+5]=='3'
            r[i,:,:] += r3
        end
    end
    return r[1,:,:], r[2,:,:], r[3,:,:], r[4,:,:]
end

function test_map_U2nhs_theta(x,y,t,m,n,rng,randmat,eps)
    @assert n<5 "$n is not implemented for HS gauge."
    theta = zeros(Float64, 4, n, n)
    zeta  = rand(rng, Float64, 4, n, n) - 0.5ones(4, n, n)

    r0, r1, r2, r3 = random_matrix_r(n,randmat)
    s0, s1, s2, s3 = random_matrix_s(n,randmat)

    theta[1,:,:] = ( m - 3 + cos(x) + cos(y) + cos(t) ) * r0 +
        eps*zeta[1,:,:] +
        ( m - 3 + cos(2x) + cos(2y) + cos(2t) ) * s0
    theta[2,:,:] = sin(x) * r1 + eps*zeta[2,:,:] +
        sin(2x) * s1
    theta[3,:,:] = sin(y) * r2 + eps*zeta[3,:,:] +
        sin(2y) * s2
    theta[4,:,:] = sin(t) * r3 + eps*zeta[4,:,:] +
        sin(2t) * s3
    
    return theta
end

function test_map_U2nhs_g(x,y,t,m,n,rng,randmat,eps)
    eye = [1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im]
    sigma1 = [0.0+0.0im 1.0+0.0im; 1.0+0.0im 0.0+0.0im]
    sigma2 = [0.0+0.0im 0.0-1.0im; 0.0+1.0im 0.0+0.0im]
    sigma3 = [1.0+0.0im 0.0+0.0im; 0.0+0.0im (-1.0)+0.0im]

    theta = test_map_U2nhs_theta(x,y,t,m,n,rng,randmat,eps)

    q = zeros(ComplexF64, 2*n, 2*n)
    for i = 1:2*n
        for j = 1:2*n
            i_R = (i-1) % n + 1
            j_R = (j-1) % n + 1
            i_P = div(i-1, n) + 1
            j_P = div(j-1, n) + 1
            q[i,j] = theta[1,i_R,j_R]*eye[i_P,j_P] +
                im*theta[2,i_R,j_R]*sigma1[i_P,j_P] +
                im*theta[3,i_R,j_R]*sigma2[i_P,j_P] +
                im*theta[4,i_R,j_R]*sigma3[i_P,j_P]
        end
    end
    F = svd(q)
    return F.U * F.Vt
end

function TestmapGauges_3D_U2nhs(NC, m, n, NX, NY, NT; verbose_level = 2, randomnumber = "Random", randmat_cond="all", reps = 0.1)
    return test_map_U2nhsGaugefields_3D_nowing(NC, NX, NY, NT, m, n, verbose_level = verbose_level, randomnumber = randomnumber, randmat_cond = randmat_cond, reps = 0.1)
end

function test_map_U2nhsGaugefields_3D_nowing(NC, NX, NY, NT, m, n; verbose_level = 2, randomnumber = "Random", randmat_cond = "all", reps=0.1)
    @assert NC==2 "NC should be 2."
    U = Gaugefields_3D_nowing(NC*n, NX, NY, NT, verbose_level = verbose_level)
    if randomnumber == "Random"
        rng = MersenneTwister()
    elseif randomnumber == "Reproducible"
        rng = StableRNG(123)
    else
        error(
            "randomnumber should be \"Random\" or \"Reproducible\". Now randomnumber = $randomnumber",
        )
    end

    for it = 1:NT
        for iy = 1:NY
            @inbounds @simd for ix = 1:NX
                x = - pi + (ix-1) * (2pi) / NX
                y = - pi + (iy-1) * (2pi) / NY
                t = - pi + (it-1) * (2pi) / NT
                U[:,:, ix, iy, it] = test_map_U2nhs_g(x,y,t,m,n,rng,randmat_cond,reps)
            end
        end
    end
    set_wing_U!
    return U
end



function test_map_U8hs_theta(x,y,t,m)
    theta = zeros(Float64, 4, 4, 4)

    r = [0.876608 0.521964 0.0862234 0.377913; 0.0116446 0.927266 0.543757 0.479332; 0.245349 0.759896 0.984993 0.217045; 0.459017 0.884729 0.583854 0.263973]
    s = [0.561954 0.0305083 0.416484 0.957411; 0.914282 0.974078 0.248139 0.318042; 0.946 0.414563 0.374183 0.643015; 0.506951 0.304123 0.634802 0.272851]
    theta[1,:,:] = ( m - 3 + cos(x) + cos(y) + cos(t) ) * r + ( m - 3 + cos(2x) + cos(2y) + cos(2t) ) * s
    
    r = [0.840093 0.00529254 0.203814 0.953795; 0.415082 0.0639042 0.22124 0.292323; 0.707049 0.744601 0.752071 0.965077; 0.595576 0.357156 0.0974585 0.509618]
    s = [0.147099 0.351169 0.0338565 0.332413; 0.972018 0.968023 0.380472 0.966577; 0.864904 0.0338764 0.152853 0.879249; 0.947237 0.292545 0.425238 0.698324]
    theta[2,:,:] = sin(x) * r + sin(2x) * s
    r = [0.00391285 0.648185 0.32748 0.429586; 0.443565 0.152388 0.670732 0.666447; 0.973967 0.288747 0.850372 0.749003; 0.742053 0.355433 0.748923 0.916096]
    s = [0.278329 0.504067 0.311738 0.661763; 0.441954 0.0638038 0.30767 0.915738; 0.738833 0.824195 0.00252546 0.819486; 0.106435 0.61738 0.886671 0.0485881]
    theta[3,:,:] = sin(y) * r + sin(2y) * s
    r = [0.521992 0.0997372 0.980063 0.485604; 0.239529 0.817905 0.792302 0.278309; 0.612152 0.987525 0.833196 0.0444734; 0.0879581 0.250586 0.384922 0.61815]
    s = [0.0512385 0.297878 0.421724 0.08238; 0.320924 0.670261 0.0735134 0.284197; 0.724307 0.338734 0.738199 0.533047; 0.956959 0.687224 0.334217 0.90432]
    theta[4,:,:] = sin(t) * r + sin(2t) * s
    
    return theta
end

function test_map_U8hs_g(x,y,t,m)
    eye = [1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im]
    sigma1 = [0.0+0.0im 1.0+0.0im; 1.0+0.0im 0.0+0.0im]
    sigma2 = [0.0+0.0im 0.0-1.0im; 0.0+1.0im 0.0+0.0im]
    sigma3 = [1.0+0.0im 0.0+0.0im; 0.0+0.0im (-1.0)+0.0im]

    theta = test_map_U8hs_theta(x,y,t,m)

    q = zeros(ComplexF64, 8, 8)
    for i = 1:8
        for j = 1:8
            i_R = (i-1) % 4 + 1
            j_R = (j-1) % 4 + 1
            i_P = div(i-1, 4) + 1
            j_P = div(j-1, 4) + 1
            q[i,j] = theta[1,i_R,j_R]*eye[i_P,j_P] +
                im*theta[2,i_R,j_R]*sigma1[i_P,j_P] +
                im*theta[3,i_R,j_R]*sigma2[i_P,j_P] +
                im*theta[4,i_R,j_R]*sigma3[i_P,j_P]
        end
    end
    F = svd(q)
    return F.U * F.Vt
end

function TestmapGauges_3D_U8hs(NC, m, NX, NY, NT; verbose_level = 2)
    return test_map_U8hsGaugefields_3D_nowing(NC, NX, NY, NT, m, verbose_level = verbose_level)
end

function test_map_U8hsGaugefields_3D_nowing(NC, NX, NY, NT, m; verbose_level = 2)
    @assert NC==2 "NC should be 2."
    U = Gaugefields_3D_nowing(NC*4, NX, NY, NT, verbose_level = verbose_level)

    for it = 1:NT
        for iy = 1:NY
            @inbounds @simd for ix = 1:NX
                x = - pi + (ix-1) * (2pi) / NX
                y = - pi + (iy-1) * (2pi) / NY
                t = - pi + (it-1) * (2pi) / NT
                U[:,:, ix, iy, it] = test_map_U8hs_g(x,y,t,m)
            end
        end
    end
    set_wing_U!
    return U
end




function minusidentityGaugefields_3D_nowing(NC, NX, NY, NT; verbose_level = 2)
    U = Gaugefields_3D_nowing(NC, NX, NY, NT, verbose_level = verbose_level)

    for it = 1:NT
        for iy = 1:NY
            for ix = 1:NX
                @simd for ic = 1:NC
                    U[ic, ic, ix, iy, it] = -1
                end
            end
        end
    end
    set_wing_U!(U)
    return U
end


function thooftFlux_3D_B_at_bndry(
    NC,
    FLUX,
    FLUXNUM,
    NN...;
    overallminus = false,
    verbose_level = 2,
)
    dim = length(NN)
    if dim == 4
        if overallminus
            U = minusidentityGaugefields_3D_nowing(
                NC,
                NN[1],
                NN[2],
                NN[3],
                verbose_level = verbose_level,
            )
        else
            U = identityGaugefields_3D_nowing(
                NC,
                NN[1],
                NN[2],
                NN[3],
                verbose_level = verbose_level,
            )
        end
        
        v = exp(-im * (2pi/NC) * FLUX)
      if FLUXNUM==1
          for it = 1:NN[3]
              #for iy = 1:NN[2]
              #for ix = 1:NN[1]
              @simd for ic = 1:NC
                  U[ic,ic,NN[1],NN[2],it] *= v
              end
              #end
              #end
          end
      elseif FLUXNUM==2
          #for it = 1:NN[3]
          for iy = 1:NN[2]
              #for ix = 1:NN[1]
              @simd for ic = 1:NC
                  U[ic,ic,NN[1],iy,NN[3]] *= v
              end
              #end
          end
          #end
      elseif FLUXNUM==3
          #for it = 1:NN[3]
          #for iy = 1:NN[2]
          for ix = 1:NN[1]
              @simd for ic = 1:NC
                  U[ic,ic,ix,NN[2],NN[3]] *= v
              end
          end
          #end
          #end
      else
          error("NumofFlux is out")
      end
    end
    set_wing_U!(U)
    return U
end


# end
