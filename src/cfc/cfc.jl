
mutable struct LSTMCell
    input_size::Int
    hidden_size::Int
    input_map::Flux.Dense
    recurrent_map::Flux.Dense
end

function LSTMCell(input_size, hidden_size)
    input_map = Flux.Dense(input_size => 4 * hidden_size, bias=true)
    recurrent_map = Flux.Dense(hidden_size, 4*hidden_size, bias=false)
    t = LSTMCell(input_size, hidden_size, input_map, recurrent_map)
    init_weights!(t)
    return t
end

function init_weights!(lstm::LSTMCell)
    
end



mutable struct WiredCfC
    input_size::Int
    wiring::Wiring
    return_sequences::Bool
    batch_first::Bool
    mixed_memory::Bool
    lstm::Union{Nothing, LSTMCell}
    rnn_cell::Union{CfCCell, WiredCfCCell}
end

function WiredCfC(
    input_size::Int,
    wiring::Union{Wiring, NCP, FullyConnected};
    return_sequences = true,
    batch_first = true,
    mixed_memory = false,
    mode::String = "default",
)
    state_size = wiring.units
    rnn_cell = WiredCfCCell(input_size, wiring; mode=mode)

    lstm = mixed_memory ? LSTMCell(input_size, state_size) : nothing

    return WiredCfC(input_size, wiring, return_sequences, batch_first, mixed_memory, lstm, rnn_cell)

end

function Base.getproperty(cfc::WiredCfC, v::Symbol)
    if v == :output_size
        return cfc.wiring.output_dim
    else 
        return getfield(cfc, v)
    end
end