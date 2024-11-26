using Random
using LinearAlgebra

function random_unitary(rng, N)
    A = rand(rng, ComplexF64, N, N) + im * rand(rng, ComplexF64, N, N)
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
                U[:, :, ix, iy, it] = random_unitary(rng, NC)
            end
        end
    end
    set_wing_U!(U)
    return U
end

function RandomGauges_3D(NC, NX, NY, NT; verbose_level = 2, randomnumber = "Random")
    return randomGaugefields_3D_nowing(
        NC,
        NX,
        NY,
        NT,
        verbose_level = verbose_level,
        randomnumber = randomnumber,
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
