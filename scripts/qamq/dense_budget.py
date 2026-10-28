import re, json, sys
from pathlib import Path

REAL_BYTES = {'DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS.rco-allocation.txt': 138238893056}

# GGML block table from ISTA converter (elements per block, bytes per block)
GGML_BLOCK = {
 'F32':(1,4),'F16':(1,2),'BF16':(1,2),'Q8_0':(32,34),'Q5_K':(256,176),'Q6_K':(256,210),
 'Q4_K':(256,144),'Q3_K':(256,110),'Q2_K':(256,84),'IQ1_M':(256,56),'IQ1_S':(256,50),
 'IQ2_XXS':(256,66),'IQ2_XS':(256,74),'IQ2_S':(256,82),'IQ2_M':(256,98),'IQ3_XXS':(256,98),
 'IQ3_XS':(256,110),'IQ3_S':(256,110),'IQ4_NL':(32,18),'IQ4_XS':(256,136),'Q4_0':(32,18),
}
H, I, E, L = 5120, 2304, 384, 40
QL, QQ, KVH, HD, RD = 1280, 64, 1, 512, 64
IDX, IDH, IH = 128, 128, 32
def shapes():
    s = {'token_embd.weight':(H,129280), 'output.weight':(129280,H), 'output_norm.weight':(H,)}
    for l in range(L):
        p=f'blk.{l}.'
        s[p+'attn_q_a.weight']=(QL,H); s[p+'attn_q_b.weight']=(QQ*HD,QL)
        s[p+'attn_kv.weight']=(HD,H); s[p+'attn_output_a.weight']=(QQ*HD//8, 8*1024)
        s[p+'attn_output_b.weight']=(H,8*1024)
        s[p+'attn_norm.weight']=(H,); s[p+'attn_q_a_norm.weight']=(QL,); s[p+'attn_kv_a_norm.weight']=(HD,)
        s[p+'attn_sinks.weight']=(QQ,)
        s[p+'ffn_norm.weight']=(H,); s[p+'ffn_gate_inp.weight']=(E,H)
        s[p+'exp_probs_b.bias']=(E,); s[p+'exp_probs_b_vl.bias']=(E,)
        s[p+'ffn_gate_shexp.weight']=(I,H); s[p+'ffn_up_shexp.weight']=(I,H); s[p+'ffn_down_shexp.weight']=(H,I)
        s[p+'ffn_gate_exps.weight']=(H,E*I); s[p+'ffn_up_exps.weight']=(H,E*I); s[p+'ffn_down_exps.weight']=(E*I,H)
        for hc in ('hc_attn_base','hc_attn_fn','hc_attn_scale','hc_ffn_base','hc_ffn_fn','hc_ffn_scale'):
            s[p+hc+'.weight']=(4,)
        if l in (1,14):
            s[p+'engram_gate.weight']=(H,2048); s[p+'engram_kv.weight']=(H,2048)
    for l in (2, 8, 14, 20):
        p = f'blk.{l}.'
        s[p+'attn_compressor_kv.weight']=(HD,H)
        s[p+'attn_compressor_norm.weight']=(HD,)
        if l != 20:
            s[p+'attn_compressor_gate.weight']=(HD,H)
    for l in (2, 8, 14, 20, 24, 28, 32, 36):
        p = f'blk.{l}.'
        if l in (2, 8, 14, 20):
            s[p+'indexer.attn_k.weight']=(IDX,H)
            s[p+'indexer.k_norm.weight']=(IDX,)
        s[p+'indexer.attn_q_b.weight']=(IH*IDX,QL)
        s[p+'indexer.proj.weight']=(IH,H)
    return s

def nbytes(shape, t):
    per, bb = GGML_BLOCK[t]
    n=1
    for d in shape: n*=d
    if shape[0] % per: raise ValueError(f'row {shape[0]} not whole blocks of {t}')
    return n//per*bb

for name in sys.argv[1:]:
    path=Path(name); alloc={}
    for line in path.read_text().splitlines():
        line=line.strip()
        if not line or line.startswith('#'): continue
        k,v=line.rsplit(': ',1); alloc[k]=v.strip()
    sh=shapes()
    missing=[k for k in sh if k not in alloc]
    extra=[k for k in alloc if k not in sh and k!='per_layer_token_embd.weight']
    if missing or extra:
        raise SystemExit(f'{path.name}: unmatched tensors missing={missing[:3]} extra={extra[:3]}')
    dense=sum(nbytes(sh[k], alloc[k]) for k in sh if 'exps' not in k)
    experts=sum(nbytes(sh[k], alloc[k]) for k in sh if 'exps' in k)
    print(f'== {path.name}')
    print(f'   tensors matched: {len(sh)-len(missing)}/{len(sh)}  missing={missing[:3]} extra={extra[:3]}')
    print(f'   dense (non-expert) : {dense/1e9:.2f} GB  {dense/1024**3:.2f} GiB')
    print(f'   routed experts     : {experts/1e9:.2f} GB  {experts/1024**3:.2f} GiB')
    print(f'   total backbone     : {(dense+experts)/1e9:.2f} GB  {(dense+experts)/1024**3:.2f} GiB')
    real = REAL_BYTES.get(path.name)
    if real:
        print(f'   real shards        : {real/1e9:.2f} GB  delta {(dense+experts-real)/1e6:+.1f} MB')
