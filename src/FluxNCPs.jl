module FluxNCPs
using Flux
using Random
using StatsBase: sample

export Wiring, FullyConnected, NCP, AutoNCP, CfC

include("wirings/wiring.jl")
include("cfc/cfc_cell.jl")
include("cfc/wired_cfc_cell.jl")
include("cfc/cfc.jl")

end # module