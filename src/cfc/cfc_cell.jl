function LeCun(x)
    return 1.7159 * Flux.tanh(0.666 .* x)
end

mutable struct CfCCell
    backbone_layers::Int
    backbone::Union{Nothing, Flux.Chain}
    sparsity_mask::AbstractArray{Float32}
    ff1::Flux.Dense
    ff2::Flux.Dense
    time_a::Flux.Dense
    time_b::Flux.Dense
end

function CfCCell(input_size, hidden_size; mode="default", backbone_activation = "lecun_tanh", backbone_units=128, backbone_layers = 1, backbone_dropout = 0.0, sparsity_mask = nothing)
    allowed_modes = ["default", "pure", "no_gate"]
    sparsity_mask = if sparsity_mask |> isnothing 
        nothing
    else
        data = abs.(sparsity_mask |> transpose)
        data
    end

    bba = if backbone_activation == "lecun_tanh"
        LeCun
    else
        throw(ErrorException("Unknown activation: $backbone_activation"))
    end

    backbone = nothing
    if backbone_layers > 0
        layer_list = [
            Flux.Dense((input_size + hidden_size) => backbone_units, bba; init=Flux.glorot_uniform)
        ]
        for i in 2:backbone_layers
            push!(layer_list, Flux.Dense(backbone_units => backbone_units, bba; init=Flux.glorot_uniform))
            if backbone_dropout > 0
                push!(layer_list, Flux.Dropout(backbone_dropout))
            end
        end
        backbone = Flux.Chain(layer_list...)
    end
    cat_shape = Int(floor( backbone_layers == 0 ? hidden_size + input_size : backbone_units))
    ff1 = Dense(cat_shape => hidden_size; init=Flux.glorot_uniform)
    @assert mode == "default"
    ff2 = Dense(cat_shape => hidden_size; init=Flux.glorot_uniform)
    time_a = Dense(cat_shape => hidden_size; init=Flux.glorot_uniform)
    time_b = Dense(cat_shape => hidden_size; init=Flux.glorot_uniform)
    CfCCell(backbone_layers, backbone, sparsity_mask, ff1, ff2, time_a, time_b)
end