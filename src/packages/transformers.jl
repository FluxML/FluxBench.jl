using Transformers
using Transformers.Pretrain
using Transformers.BidirectionalEncoder

function bhfw(f, args...)
    CUDA.@sync f(args...)
end

function bhbw(f, args...)
    gs = CUDA.@sync gradient((x...)->sum(f(x...)), args...)
end

function transformer_add_trf(::Type{Transformer}, layern, batchn,
                             trf_group = addgroup!(SUITE, "Transformers"))
    hidden, head, intermediate = 512, 8, 2048
    encoder = Chain((Transformer(hidden, head, intermediate; act = gelu) for i = 1:layern)...)
    ip = randn(512, 100, batchn)

    trf_group["Transformers_Forward_Pass_nlayer_$(layern)_nbatch_$(batchn)"] = @benchmarkable(
        fw(model, gip), setup = (model = gpu($encoder); gip = gpu($ip)),
        teardown = (GC.gc(); CUDA.reclaim())
    )

    trf_group["Transformers_Backward_Pass_nlayer_$(layern)_nbatch_$(batchn)"] = @benchmarkable(
        bw(model, gip), setup = (model = gpu($encoder); gip = gpu($ip)),
        teardown = (GC.gc(); CUDA.reclaim())
    )

end

bert_fw(m, x) = m.transformers(m.embed(x))

function transformer_add_trf(::Type{Bert}, batchn,
                             trf_group = addgroup!(SUITE, "Transformers"))
    ENV["DATADEPS_ALWAYS_ACCEPT"] = true
    bert_model = pretrain"Bert-uncased_L-12_H-768_A-12:bert_model"
    vocab_size = size(bert_model.embed.embeddings.tok.embedding, 2)
    gmodel = gpu(bert_model)

    for seq_len in (8, 32, 128) #, 512) OOM
        token = rand(1:vocab_size, seq_len, batchn)
        segment = fill(1, seq_len, batchn)
        ip = (tok = token, segment = segment)

        trf_group["Bert-base-uncased_Forward_Pass_seq_len_$(seq_len)_nbatch_$(batchn)"] = @benchmarkable(
            bhfw(bert_fw, model, gip), setup = (model = $gmodel; gip = gpu($ip)),
            teardown = (GC.gc(); CUDA.reclaim())
        )

        trf_group["Bert-base-uncased_Backward_Pass_seq_len_$(seq_len)_nbatch_$(batchn)"] = @benchmarkable(
            bhbw(bert_fw, model, gip), setup = (model = $gmodel; gip = gpu($ip)),
            teardown = (GC.gc(); CUDA.reclaim())
        )
    end
end
