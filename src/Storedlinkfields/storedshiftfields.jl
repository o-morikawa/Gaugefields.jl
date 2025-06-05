module Storedshiftfields_module

mutable struct Storedshiftfields{TG}
    _data::Vector{TG}
    _disp::Vector{(Tuple, Bool)}
    _flagusing::Vector{Bool}
    _indices::Vector{Int64}
    _numused::Vector{Int64}
    Nmax::Int64

    function Storedshiftfields(a::TG; num=1, Nmax=1000) where {TG}
        _data = Vector{TG}(undef, num)
        _disp = Vector{(NTuple{4,Int}, Bool)}(undef, num)
        _flagusing = zeros(Bool, num)
        _indices = zeros(Int64, num)
        _numused = zeros(Int64, num)
        similar_l = (ntuple(_->0, 4), false)
        for i = 1:num
            _data[i] = similar(a)
            _disp[i] = similar_l
        end
        return new{TG}(_data, _disp, _flagusing, _indices, _numused, Nmax)
    end

    function Storedshiftfields(_data::Vector{TG}, _disp::Vector{(NTuple{4,Int}, Bool)}, _flagusing, _indices, _numused, Nmax) where {TG}
        return new{TG}(_data, _disp, _flagusing, _indices, _numused, Nmax)
    end

end

function Storedshiftfields_fromvector(a::Vector{TG}, l::Vector{(NTuple{4,Int}, Bool)}; Nmax=1000) where {TG}
    num = length(a)
    if num != length(l)
        error("Lengths of TG and Disp vectors are mismatched.")
    end
    _flagusing = zeros(Bool, num)
    _indices = zeros(Int64, num)
    _numused = zeros(Int64, num)
    return Storedshiftfields(a, l, _flagusing, _indices, _numused, Nmax)
end
export Storedshiftfields_fromvector

Base.eltype(::Type{Storedshiftfields{TG}}) where {TG} = TG

Base.length(t::Storedshiftfields{TG}) where {TG} = length(t._data)

Base.size(t::Storedshiftfields{TG}) where {TG} = size(t._data)

function Base.firstindex(t::Storedshiftfields{TG}) where {TG}
    return 1
end

function Base.lastindex(t::Storedshiftfields{TG}) where {TG}
    return length(t._data)
end

function Base.getindex(t::Storedshiftfields{TG}, i::Int) where {TG}
    @assert i <= length(t._data) "The length of the storedlinkfields is shorter than the index $i."
    @assert i <= t.Nmax "The number of the storedlinkfields $i is larger than the maximum number $(Nmax). Change Nmax."
    if t._indices[i] == 0
        index = findfirst(x -> x == 0, t._flagusing)
        t._flagusing[index] = true
        t._indices[i] = index
    end

    return t._data[t._indices[i]], t._disp[t._indices[i]]
end

function Base.getindex(t::Storedshiftfields{TG}, I::Vararg{Int,N}) where {TG,N}
    data = TG[]
    disp = []
    for i in I
        data_tmp, disp_tmp = t[i]
        push!(data, data_tmp)
        push!(disp, disp_tmp)
    end
    return data, disp
end

function Base.getindex(t::Storedshiftfields{TG}, I::AbstractVector{T}) where {TG,T<:Integer}
    data = TG[]
    disp = []
    for i in I
        data_tmp, disp_tmp = t[i]
        push!(data, data_tmp)
        push!(disp, disp_tmp)
    end
    return data, disp
end

function Base.display(t::Storedshiftfields{TG}) where {TG}
    n = length(t._data)
    println("The storage size of fields: $n")
    numused = sum(t._flagusing)
    println("The total number of fields used: $numused")
    for i = 1:n
        if t._indices[i] != 0
            println("The address $(t._indices[i]) is used as the index $i")
        end
    end
    println("The flags: $(t._flagusing)")
    println("The indices: $(t._indices)")
    println("The num of using: $(t._numused)")
end

function is_stored_shiftfield(t::Storedshiftfields{TG}, l::(NTuple{4, Int}, Bool)) where {TG}
    if l in t._disp
        return true
    else
        return false
    end
end

function store_shiftfield!(t::Storedshiftfields{TG}, a::TG, l::(NTuple{4, Int}, Bool)) where {TG}
    n = length(t._data)
    i = findfirst(x -> x == 0, t._indices)
    if i == nothing
        error("All strage of $n fields are used.")
    end
    index = i
    if !is_stored_shiftfield(t,l)
        t._flagusing[index] = true
        t._indices[i] = index
        t._numused[index] += 1
        t._data[index] = deepcopy(a)
        t._disp[index] = deepcopy(l)
    end
end

function store_shiftfield!(t::Storedshiftfields{TG}, as::Vector{TG}, ls::Vector{(NTuple{4, Int}, Bool)}) where {TG}
    n = length(as)
    if n != length(ls)
        error("Lengths of TG and WL vectors are mismatched.")
    end
    for i = 1:n
        store_shiftfield!(t, as[i], ls[i])
    end
end

function get_stored_shiftfield(t::Storedshiftfields{TG}, l::(NTuple{4, Int}, Bool)) where {TG}
    i = findfirst(x -> x == l, t._disp)
    if i == nothing
        error("not matched shiftfields.")
    end
    index = t._indices[i]
    t._numused[index] += 1
    return t._data[index]
end

export Storedshiftfields, is_stored_shiftfield, store_shiftfield!, get_stored_shiftfield

end
