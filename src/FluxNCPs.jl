module FluxNCPs
using Flux
using Random
using StatsBase: sample

export Wiring, FullyConnected, NCP

include("wirings/wiring.jl")
end # module