function LeCun(x)
    return 1.7159f0 * Flux.tanh(0.666f0 .* x)
end

mutable struct CfCCell
    backbone_layers::Int
    backbone::Union{Nothing,Flux.Chain}
    sparsity_mask::AbstractArray
    ff1::Flux.Dense
    ff2::Flux.Dense
    time_a::Flux.Dense
    time_b::Flux.Dense
end

function get_property(cell::CfCCell, v::Symbol)
    if v == :state_size
        return cell.ff2.bias |> length
    else
        return getfield(cell, v)
    end

end

function CfCCell(input_size, hidden_size; mode="default", backbone_activation="lecun_tanh", backbone_units=128, backbone_layers=1, backbone_dropout=0.0, sparsity_mask=nothing)
    sparsity_mask = if sparsity_mask |> isnothing
        nothing
    else
        abs.(sparsity_mask |> transpose)
    end

    bba = if backbone_activation == "lecun_tanh"
        LeCun
    else
        throw(ErrorException("Unknown activation: $backbone_activation"))
    end

    backbone = nothing
    if backbone_layers > 0
        layer_list = [
            Flux.Dense((input_size + hidden_size) => backbone_units, bba)
        ]
        for i in 2:backbone_layers
            push!(layer_list, Flux.Dense(backbone_units => backbone_units, bba))
            if backbone_dropout > 0
                push!(layer_list, Flux.Dropout(backbone_dropout))
            end
        end
        backbone = Flux.Chain(layer_list...)
    end
    cat_shape = backbone_layers == 0 ? hidden_size + input_size : backbone_units
    ff1 = Dense(cat_shape => hidden_size; init=Flux.glorot_uniform)
    ff2 = Dense(cat_shape => hidden_size; init=Flux.glorot_uniform)
    time_a = Dense(cat_shape => hidden_size; init=Flux.glorot_uniform)
    time_b = Dense(cat_shape => hidden_size; init=Flux.glorot_uniform)
    CfCCell(backbone_layers, backbone, sparsity_mask, ff1, ff2, time_a, time_b)
end

function (cell::CfCCell)(h, x)

    x = vcat(x, h)
    if cell.backbone_layers > 0
        x = cell.backbone(x)
    end
    if !(isnothing(cell.sparsity_mask))

        # ff1 = F.linear(x, self.ff1.weight * self.sparsity_mask, self.ff1.bias)
        ff1 = muladd(cell.ff1.weight .* cell.sparsity_mask, x, cell.ff1.bias)
    else
        ff1 = cell.ff1(x)
    end
    if !(isnothing(cell.sparsity_mask))
        # ff2 = F.linear(x, self.ff2.weight * self.sparsity_mask, self.ff2.bias)
        ff2 = muladd(cell.ff2.weight .* cell.sparsity_mask, x, cell.ff2.bias)
    else
        ff2 = cell.ff2(x)
    end
    ff1 = Flux.tanh_fast(ff1)
    ff2 = Flux.tanh_fast(ff2)
    t_a = cell.time_a(x)
    t_b = cell.time_b(x)
    t_interp = Flux.sigmoid(-t_a .+ t_b)
    h_prime = ff1 .* (1.0f0 .- t_interp) .+ ff2 .* t_interp
    return h_prime
end

Flux.@functor CfCCell