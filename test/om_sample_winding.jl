#using LatticeQCD

using Random
using Dates
using Gaugefields
using LinearAlgebra
using Wilsonloop

function UN_test_3D(NX,NY,NT,NC,β)

    Dim = 3

    #Random.seed!(123)
    t0 = Dates.DateTime(2024,1,1,16,10,7)
    t  = Dates.now()
    Random.seed!(Dates.value(t-t0))

    U = Initialize_3D_UN_Gaugefields(
        NC,NX,NY,NT,
        condition = "hot",
        randomnumber="Random"
    )
    println(typeof(U))

    temps = Temporalfields(U, num=9)
    println(typeof(temps))

    g = Gradientflow_3D(U, eps=0.001)
    flownumber = 1000
    for iflow = 1:flownumber
        flow!(U, g)
        if iflow%100==0
            W = winding_UN_3D(U,temps)
            println(W)
        end
    end

end


function main()
    β = 3.0
    NX = 16
    NY = 16
    NT = 16
    NC = 2
    @time UN_test_3D(NX,NY,NT,NC,β)
end
main()
