#!/usr/bin/env python3
"""Reconstruction + coverage eval for a T2 model. Args: <t2_dir> <t1_path>.
Prints head coverage, eff-dim, and the mission metric: essence -> recovered T1
vectors -> instruction opcode/immediate accuracy (baseline single-head: 67%/69%)."""
import sys, json
sys.path.insert(0, '/home/tophy/Projects/riscv-thoughts')
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, subprocess
from datagen import RVT_FORMAT, Batch
from compressor.train import (_t2_assemble, _split_to_per_instruction,
                              encode_instrs, _immediate_rotary_np)
from tokenizer import decode_instruction
from emulator import I_TYPE
from datagen.generate import _ALU_R_OPS, _ALU_I_OPS
from scripts._common import resolve_device, load_frozen_encoder, load_t2

MODEL, T1PATH = sys.argv[1], sys.argv[2]
dev = resolve_device('auto'); t1 = load_frozen_encoder(T1PATH, dev)
t2, _ = load_t2(MODEL, t1, dev); core = t2.core
L = int(sys.argv[3]) if len(sys.argv) > 3 else 8   # chunk len; MUST match
                                                   # the model's training len
COVER = core.n_pool_heads > 1 and not getattr(core, 'pool_reads_order', False)
OPS = sorted(set(_ALU_R_OPS) | set(_ALU_I_OPS)); op2i = {o: i for i, o in enumerate(OPS)}
EV = f"/tmp/rec_ev_{L}.rvt"
subprocess.run(f"python -m datagen --rule cap={L} --twins 2 --batch-size 128 "
               f"--n-states 8 --anchor-seed 0 --config configs/instr_alu_writers.json "
               f"--count 12 > {EV} 2>/dev/null", shell=True)
with open(EV, 'rb') as f:
    batches = [b for i, b in zip(range(12), RVT_FORMAT.reader(f, Batch))]
ESS = []; TV = []; OPC = []; IMM = []; ISI = []; cover = []
k = core.n_pool_heads
for batch in batches:
    split = _split_to_per_instruction(batch)
    if split[0].shape[0] == 0: continue
    nch, mni = batch.instr_lens.shape
    ct1, c0, c1, co, cact = _t2_assemble(t1, split, nch, mni, dev, route='binding')
    with torch.no_grad():
        T, ess = core(ct1, c0, c1, co, cact)
        if COVER:
            value = T[..., :core.d_value]; q = core.pool_q.view(k, -1)
            d = 2 if core.pool_mode == 'compete' else 2
            att = torch.softmax(core.pool_k(value) @ q.t(), dim=d)
            owner = att.argmax(2)
    for b in range(batch.tokens.shape[0]):
        if not batch.valid[b]: continue
        if COVER: cover.append(len(set(owner[b].tolist())))
        toks = batch.tokens[b].tolist(); ins = []; pos = 0
        try:
            for _ in range(L): i, pos = decode_instruction(toks, pos); ins.append(i)
        except Exception: continue
        if len(ins) != L: continue
        with torch.no_grad(): tv = encode_instrs(t1, ins, dev).cpu().numpy()
        ESS.append(ess[b].cpu().numpy()); TV.append(tv)
        OPC.append([op2i[x.opcode] for x in ins])
        IMM.append([x.args[2] if x.opcode in I_TYPE else 0 for x in ins])
        ISI.append([1 if x.opcode in I_TYPE else 0 for x in ins])
ESS = np.array(ESS, np.float32); TV = np.array(TV, np.float32)
OPC = np.array(OPC); IMM = np.array(IMM); ISI = np.array(ISI); n = len(ESS)
mu = ESS.mean(0); lam = np.linalg.svd(ESS - mu, compute_uv=False) ** 2
pr = (lam.sum() ** 2) / (lam ** 2).sum()
covs = f"{np.mean(cover):.1f}/{k}" if COVER else "n/a"
idx = np.random.default_rng(0).permutation(n); tr, te = idx[:int(.8 * n)], idx[int(.8 * n):]
X = torch.tensor(ESS).to(dev); Y = torch.tensor(TV.reshape(n, -1)).to(dev)
net = nn.Sequential(nn.Linear(256, 1024), nn.ReLU(), nn.Linear(1024, L * 64)).to(dev)
opt = torch.optim.Adam(net.parameters(), 2e-3)
for _ in range(1200): opt.zero_grad(); F.mse_loss(net(X[tr]), Y[tr]).backward(); opt.step()
with torch.no_grad(): REC = net(X[te]).reshape(-1, L, 64)
Xt = torch.tensor(TV[tr].reshape(-1, 64)).to(dev); Yt = torch.tensor(OPC[tr].reshape(-1)).to(dev)
clf = nn.Linear(64, len(OPS)).to(dev); o2 = torch.optim.Adam(clf.parameters(), 1e-2)
for _ in range(800): o2.zero_grad(); F.cross_entropy(clf(Xt), Yt).backward(); o2.step()
with torch.no_grad(): op_pred = clf(REC.reshape(-1, 64)).argmax(1).cpu().numpy().reshape(-1, L)
cand = np.arange(-2048, 2048); ctab = torch.from_numpy(_immediate_rotary_np(cand)).to(dev)
with torch.no_grad():
    imp = cand[torch.cdist(t1.imm_head(F.normalize(REC.reshape(-1, 64), dim=-1)), ctab).argmin(1).cpu().numpy()].reshape(-1, L)
m = ISI[te].astype(bool)
opa = 100 * (op_pred == OPC[te]).mean(); ima = 100 * (imp[m] == IMM[te][m]).mean()
print(f"  coverage={covs}  eff-dim(PR)={pr:.1f}  RECON opcode={opa:.1f}% immediate={ima:.1f}%  (baseline 66.9/69.5)")
