#using LatticeQCD

using Random
using Dates
using Gaugefields
using LinearAlgebra
using Wilsonloop

using Plots

function UN_test_3D(NX,NY,NT,NC,β)

    Dim = 3

    n = 3

    eps = 0.01
    flow_number = 1000
    step = 10
    
    w = zeros(Float64, n, Int(flow_number/step))

    for i = 1:n

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

        g = Gradientflow_3D(U, eps=eps)
        flownumber = flow_number
        j = 0
        for iflow = 1:flownumber
            flow!(U, g)
            if iflow%step==0
                j += 1
                W = winding_UN_3D(U,temps)
                w[i,j] = W
            end
        end
    end

    println(w)
    
    flow = (eps*step):(eps*step):(eps*flow_number)
    plt = plot(flow, w[1,:])
    for i = 2:n
        plot!(plt, flow, w[i,:])
    end
    savefig("winding.png")

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
