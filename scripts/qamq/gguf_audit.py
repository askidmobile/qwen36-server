"""Read GGUF headers of the downloaded shards and total bytes by tensor class.

This is the measurement that keeps `dense_budget.py` honest: shapes come from the
files, not from config arithmetic. Run with ``shapes`` as the second argument to
dump one example of every tensor shape.

    python scripts/qamq/gguf_audit.py MODEL_DIR [shapes]
"""
import struct, sys, glob, collections, os

SCALAR = {0:'<B',1:'<b',2:'<H',3:'<h',4:'<I',5:'<i',6:'<f',7:'<?',10:'<Q',11:'<q',12:'<d'}
TYPE_BYTES = {0:(1,4),1:(1,2),2:(32,18),3:(32,20),6:(32,22),7:(32,24),8:(32,34),9:(32,36),
 10:(256,84),11:(256,110),12:(256,144),13:(256,176),14:(256,210),15:(256,292),
 16:(256,66),17:(256,74),18:(256,98),19:(256,50),20:(32,18),21:(256,110),22:(256,82),
 23:(256,136),24:(1,1),25:(1,2),26:(1,4),27:(1,8),28:(1,8),29:(256,56),30:(1,2),34:(1,1),39:(32,17)}
NAMES = {0:'F32',1:'F16',2:'Q4_0',3:'Q4_1',6:'Q5_0',7:'Q5_1',8:'Q8_0',9:'Q8_1',10:'Q2_K',
 11:'Q3_K',12:'Q4_K',13:'Q5_K',14:'Q6_K',15:'Q8_K',16:'IQ2_XXS',17:'IQ2_XS',18:'IQ2_S',
 19:'IQ3_XXS',20:'IQ4_NL',21:'IQ3_S',22:'IQ2_M',23:'IQ4_XS',24:'I8',25:'I16',26:'I32',
 27:'I64',28:'F64',29:'IQ1_S',30:'BF16',34:'TQ1_0',39:'MXFP4'}

def read_names(path):
    """Just the tensor names, for layout checks that do not need shapes."""
    return [t[0] for t in read(path)]


def read(path):
    with open(path,'rb') as f:
        buf = f.read(64 << 20)  # 64 MiB: the 129k-token vocabulary alone is ~4 MB
        if buf[:4] != b'GGUF': raise ValueError('not gguf')
        ver, = struct.unpack_from('<I', buf, 4)
        n_t, n_kv = struct.unpack_from('<QQ', buf, 8)
        off = 24
        def u(fmt):
            nonlocal off
            v, = struct.unpack_from(fmt, buf, off); off += struct.calcsize(fmt); return v
        def s():
            nonlocal off
            n = u('<Q'); v = buf[off:off+n].decode('utf-8', 'replace'); off += n; return v
        def skip(t):
            nonlocal off
            if t == 8: s()
            elif t == 9:
                at = u('<I'); n = u('<Q')
                for _ in range(n): skip(at)
            else: u(SCALAR[t])
        for _ in range(n_kv):
            s(); t = u('<I'); skip(t)
        tensors = []
        for _ in range(n_t):
            name = s(); nd = u('<I')
            dims = [u('<Q') for _ in range(nd)]
            ty = u('<I'); toff = u('<Q')
            tensors.append((name, dims, ty, toff))
    return tensors

def nbytes(dims, ty):
    per, bb = TYPE_BYTES[ty]
    n = 1
    for d in dims: n *= d
    if dims[0] % per: raise ValueError(f'row {dims[0]} not whole {NAMES[ty]} blocks')
    return n // per * bb

def classify(name):
    if 'ffn_gate_exps' in name or 'ffn_up_exps' in name or 'ffn_down_exps' in name: return 'routed_experts'
    if 'per_layer_token_embd' in name: return 'engram_table'
    if 'shexp' in name: return 'shared_expert'
    if 'attn_' in name or 'indexer' in name or 'compressor' in name: return 'attention'
    if 'ffn_' in name: return 'moe_router'
    if name.startswith(('token_embd','output')): return 'embed_head'
    return 'other'

total = collections.Counter()
per_type = collections.Counter()
seen = set()
for path in sorted(glob.glob(os.path.join(sys.argv[1], '*-0000*-of-*.gguf'))):
    if path.endswith('.done'): continue
    for name, dims, ty, _ in read(path):
        if name in seen: continue
        seen.add(name)
        b = nbytes(dims, ty)
        total[classify(name)] += b
        per_type[NAMES.get(ty, ty)] += b
print(f'files {len(glob.glob(os.path.join(sys.argv[1], "*-0000*-of-*.gguf")))} tensors {len(seen)}')
for k, v in total.most_common():
    print(f'  {k:16} {v/1e9:8.3f} GB  {v/1024**3:8.2f} GiB')
backbone = sum(v for k, v in total.items() if k != 'engram_table')
print(f'  {"BACKBONE":16} {backbone/1e9:8.3f} GB  {backbone/1024**3:8.2f} GiB')
if len(sys.argv) > 2:
    for path in sorted(glob.glob(os.path.join(sys.argv[1], '*-0000*-of-*.gguf'))):
        for name, dims, ty, _ in read(path):
            if 'ffn_' in name and 'exps' in name: continue
            print(f'  SHAPE {name} {tuple(int(d) for d in dims)} {NAMES.get(ty,ty)}')
print('  types:', ', '.join(f'{k}={v/1e9:.1f}GB' for k, v in per_type.most_common()))
