
mutable struct LSTMCell
    input_size::Int
    hidden_size::Int
    input_map::Flux.Dense
    recurrent_map::Flux.Dense
end

function LSTMCell(input_size, hidden_size)
    input_map = Flux.Dense(input_size => 4 * hidden_size, bias=true)
    recurrent_map = Flux.Dense(hidden_size, 4 * hidden_size, bias=false)
    t = LSTMCell(input_size, hidden_size, input_map, recurrent_map)
    init_weights!(t)
    return t
end

function init_weights!(lstm::LSTMCell)

end

function CfC(
    input_size::Int,
    wiring::NCP;
    return_sequences=true,
    mixed_memory=false,
    mode::String="default"
)
    w = WiredCfC(input_size, wiring; return_sequences, mixed_memory, mode)
    return Flux.Recur(w, w.state0)
end




mutable struct WiredCfC
    input_size::Int
    wiring::Union{Wiring,NCP,FullyConnected}
    return_sequences::Bool
    mixed_memory::Bool
    lstm::Union{Nothing,LSTMCell}
    rnn_cell::Union{CfCCell,WiredCfCCell}
    state0::AbstractArray
end

function WiredCfC(
    input_size::Int,
    wiring::Union{Wiring,NCP,FullyConnected};
    return_sequences=true,
    mixed_memory=false,
    mode::String="default"
)
    state_size = wiring.units
    rnn_cell = WiredCfCCell(input_size, wiring; mode=mode)

    lstm = mixed_memory ? LSTMCell(input_size, state_size) : nothing

    return WiredCfC(input_size, wiring, return_sequences, mixed_memory, lstm, rnn_cell, zeros(state_size, 1))
end

function Base.getproperty(cfc::WiredCfC, v::Symbol)
    if v == :output_size
        return cfc.wiring.output_dim
    elseif v == :state_size
        return cfc.wiring.units
    elseif v == :layer_sizes
        return cfc.rnn_cell.layer_sizes
    else
        return getfield(cfc, v)
    end
end

function (cfc::WiredCfC)(hx::AbstractArray, x::AbstractArray)
    # the dimensions of x are (input_size, batch_size, sequence_length)
    
    # assert that input size lines up with that of cfc


    h_state, c_state = cfc.mixed_memory ? hx : (hx, nothing)
    
    if cfc.mixed_memory
        h_state, c_state = cfc.lstm(x, (h_state, c_state))
    end

    return cfc.rnn_cell(h_state, x)

end


function CfC(in_out::Pair{Int, NCP}; kwargs...)
    CfC(in_out.first, in_out.second)
end


Flux.@functor WiredCfC