#!/usr/bin/env python3
"""Writers-only T2 faithfulness probe. Args: <t2_dir> <t1_path> <chunk_len>.

Generates len-L chunks of the SAME happy-path instructions the T1/T2 were trained
on (valid, non-x0, R/I-ALU), encodes each through frozen T1 -> T2 (route=binding),
and reports the behavioral-fingerprint properties:
  on-sphere    — T2 essence magnitude (all valid -> ~1.0)
  rename       — chunk vs register-relabeling: essence distance (equivariance; ~0)
  commutative  — chunk vs commutative-operand-swap: distance (trained GVN collapse)
  distinct     — unrelated, behaviorally-different chunks: distance (must be large)
A faithful fingerprint: rename ~ commutative << distinct, all on the sphere.
"""
import sys, json
sys.path.insert(0, '/home/tophy/Projects/riscv-thoughts')
import numpy as np, torch, torch.nn.functional as F
from compressor.eval import encode_chunks_t2, _commute_variant, _behaviorally_equal
from scripts._common import resolve_device, load_frozen_encoder, load_t2
from datagen import build_opcode_table, make_anchor_states, precompute_chunk
from datagen.generate import random_instruction, random_relabel

t2_dir, t1_path, L = sys.argv[1], sys.argv[2], int(sys.argv[3])
dev = resolve_device('auto')
t1 = load_frozen_encoder(t1_path, dev)
t2, _ = load_t2(t2_dir, t1, dev)
dist = json.load(open('configs/instr_alu_writers.json'))
tbl = build_opcode_table(dist)
dest = list(range(1, 32))
anchors = make_anchor_states(8, 0)
rng = np.random.default_rng(0)

N = 300
bases, twins, commutes = [], [], []
while len(bases) < N:
    chunk = [random_instruction(rng, opcode_table=tbl, dest_regs=dest) for _ in range(L)]
    try:
        precompute_chunk(chunk, anchors)
    except Exception:
        continue
    cv = _commute_variant(chunk, rng)
    if cv is None:
        continue
    bases.append(chunk); twins.append(random_relabel(chunk, rng)); commutes.append(cv)

def enc(chunks):
    return encode_chunks_t2(t1, t2, chunks, dev, route='binding')

eb_raw = enc(bases)
norm = eb_raw.norm(dim=-1).cpu().numpy()
eb = F.normalize(eb_raw, dim=-1)
et = F.normalize(enc(twins), dim=-1)
ec = F.normalize(enc(commutes), dim=-1)
d_rename = (eb - et).norm(dim=-1).mean().item()
# Split the commutative pairs by whether the swap actually PRESERVED the
# function. Swapping a commutative op's operands reorders the chunk's first
# reads, so if a later instruction reads one of them the two chunks compute
# different canonical functions; those ~2% sit at the `distinct` scale and
# inflate a pooled mean several-fold.
d_all = (eb - ec).norm(dim=-1)
eq = [i for i in range(len(bases))
      if _behaviorally_equal(bases[i], commutes[i], anchors) is True]
ne = [i for i in range(len(bases))
      if _behaviorally_equal(bases[i], commutes[i], anchors) is False]
d_behav = d_all[eq].mean().item() if eq else float('nan')
d_behav_ne = d_all[ne].mean().item() if ne else float('nan')

perm = rng.permutation(len(bases))
d_distinct = []
for i, j in enumerate(perm):
    if i == j:
        continue
    if _behaviorally_equal(bases[i], bases[j], anchors) is False:
        d_distinct.append((eb[i] - eb[j]).norm().item())
d_distinct = float(np.mean(d_distinct)) if d_distinct else float('nan')

print(f"on-sphere: T2 essence norm median={np.median(norm):.3f} p1={np.percentile(norm,1):.3f} min={norm.min():.3f} frac<0.9={100*(norm<0.9).mean():.1f}%")
print(f"rename      mean dist={d_rename:.4f}   (equivariance -> ~0)")
print(f"commutative mean dist={d_behav:.4f}   (swap PRESERVED the function; "
      f"n={len(eq)} — trained GVN collapse -> small) <- judge collapse on this")
print(f"commut-diff mean dist={d_behav_ne:.4f}   (swap CHANGED the function; "
      f"n={len(ne)} — SHOULD be large)")
print(f"distinct    mean dist={d_distinct:.4f}   (must be >> the collapse row)")
ratio = d_distinct / max(d_behav, 1e-6)
print(f"separation ratio distinct/commutative = {ratio:.1f}x   "
      f"(equivalent pairs only)")
