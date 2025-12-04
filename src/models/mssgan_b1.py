# mssgan_b1.py
import os
import math
import random
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---------------------------
# Utility modules
# ---------------------------
class MLP(nn.Module):
    def __init__(self, insz, outsz, hidden=256, n_layers=2, dropout=0.0):
        super().__init__()
        layers = []
        prev = insz
        for i in range(n_layers - 1):
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU(True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden
        layers.append(nn.Linear(prev, outsz))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

# ---------------------------
# Episodic Memory (simplified tree + attention)
# ---------------------------
class BinaryTreeLSTMCell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # simple parametrisation to produce parent from two children
        self.fc = nn.Linear(dim * 4, dim * 3)  # gates: i, f, o (simplified) + candidate
    def forward(self, hL, uL, hR, uR):
        # hL, uL, hR, uR are (B, dim)
        cat = torch.cat([hL, hR, uL, uR], dim=1)
        out = self.fc(cat)
        i, f, o_and_c = torch.split(out, self.dim, dim=1)  # i,f,o/c simplified
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o_and_c)  # reuse as output gate
        # candidate
        candidate = torch.tanh(o_and_c)
        uP = f * uL + (1 - f) * uR + i * candidate
        hP = torch.tanh(uP) * o
        return hP, uP

class EpisodicMemory(nn.Module):
    def __init__(self, dim=256, mem_len=1100, extract_depth=3):
        super().__init__()
        self.dim = dim
        self.N = mem_len
        self.l = extract_depth
        self.queue: List[torch.Tensor] = []  # list of tensors (B x dim) appended over training
        self.cell = BinaryTreeLSTMCell(dim)

    def append(self, c_t):
        c_t = c_t.detach()  # stays on GPU
    
        for i in range(c_t.size(0)):
            self.queue.append(c_t[i:i+1].clone())  # stays on GPU
    
        if len(self.queue) > self.N:
            self.queue = self.queue[-self.N:]

    def build_tree_topk(self, device):
        # returns stacked nodes => tensor (M, dim) on device
        if len(self.queue) == 0:
            return None
        # build small binary tree on CPU nodes, then return top l layers nodes
        nodes = [t.to(device) for t in self.queue]  # list of B x dim cpu tensors
        # convert each to device when stacking
        # ensure even length
        while len(nodes) % 2 == 1 and len(nodes) > 1:
            nodes.append(nodes[-1])
        layers = [nodes]
        # merge until single root
        while len(layers[-1]) > 1:
            cur = layers[-1]
            parents = []
            for i in range(0, len(cur), 2):
                hL = cur[i].to(device)
                hR = cur[i+1].to(device) if i+1 < len(cur) else cur[i].to(device)
                # use h as cell as well (u ~ h) for simplicity
                hP, uP = self.cell(hL, hL, hR, hR)
                parents.append(hP.detach())
            layers.append(parents)
        # extract top l layers (from root downwards)
        top_nodes = []
        top_idx = len(layers) - 1
        for depth in range(self.l):
            idx = max(0, top_idx - depth)
            top_nodes.extend(layers[idx])
        # stack per-batch: nodes are cpu B x dim; convert to tensor shape (B, M, dim)
        # note: queue items were per-batch; stacking them duplicates for batch below
        # We'll return nodes as a list of tensors (B x dim) on device
        stacked = top_nodes
        if len(stacked) == 0:
            return None
        # concatenate along new dim -> (B, M, dim)
        MEM = torch.stack(stacked, dim=1)  # B x M x dim
        return MEM

    def read(self, c_t: torch.Tensor):
        # c_t: B x dim
        device = c_t.device
        MEM = self.build_tree_topk(device)
        if MEM is None:
            return torch.zeros_like(c_t)
        # attention: dot product between c_t and MEM_j
        q = c_t.unsqueeze(1)  # B x 1 x dim
        att = (q * MEM).sum(dim=2)  # B x M
        alpha = F.softmax(att, dim=1)
        mEM = (alpha.unsqueeze(2) * MEM).sum(dim=1)  # B x dim
        return mEM

# ---------------------------
# Semantic Memory (SM)
# ---------------------------
class SemanticMemory(nn.Module):
    def __init__(self, dim=256, b=80):
        super().__init__()
        self.k = dim
        self.b = b
        # initialise M (k x b)
        self.M = nn.Parameter(torch.randn(dim, b) * 0.1)

    def read(self, c_t: torch.Tensor):
        # c_t: B x k
        # compute attention between c_t and M columns (b)
        # M_t: k x b -> transpose to b x k
        Mbk = self.M.t().unsqueeze(0).expand(c_t.size(0), -1, -1)  # B x b x k
        q = c_t.unsqueeze(1)  # B x 1 x k
        att = (q * Mbk).sum(dim=2)  # B x b
        alpha = F.softmax(att, dim=1)
        mSM = (alpha.unsqueeze(2) * Mbk).sum(dim=1)  # B x k
        return mSM

    def update(self, mEM_t: torch.Tensor):
        # mEM_t: B x k
        # simple moving average update: M <- (1 - gamma)*M + gamma*(avg_mEM outer alpha)
        # Here we compute a column-wise soft assignment from mean mEM to update columns
        with torch.no_grad():
            mean_m = mEM_t.mean(dim=0)  # k
            scores = torch.matmul(self.M.t(), mean_m)  # b
            alpha = F.softmax(scores, dim=0)  # b
            update = torch.ger(mean_m, alpha)  # k x b
            # moving average with small gamma
            gamma = 0.1
            self.M.data = (1 - gamma) * self.M.data + gamma * update

# ---------------------------
# Encoder (for categorical sequence step)
# ---------------------------
class StepEncoder(nn.Module):
    def __init__(self, n_zones=11, n_types=11, emb_dim=64, ctx_dim=6, hidden=256):
        super().__init__()
        # note: zone vocab used '0'..'9' but your shot_vocab has size 11 etc.
        self.zone_emb = nn.Embedding(n_zones, emb_dim, padding_idx=0)
        self.type_emb = nn.Embedding(n_types, emb_dim, padding_idx=0)
        self.ctx_mlp = MLP(ctx_dim, emb_dim, hidden=emb_dim, n_layers=2)
        # fuse embeddings and context into a small GRU cell output c_t
        self.fuse = nn.Linear(emb_dim * 3, hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.hidden = hidden

    def forward(self, zone_idx, type_idx, ctx, h_prev=None):
        # zone_idx, type_idx: (B,) long
        # ctx: (B, ctx_dim) float
        ze = self.zone_emb(zone_idx)    # B x emb
        te = self.type_emb(type_idx)    # B x emb
        ce = self.ctx_mlp(ctx)          # B x emb
        cat = torch.cat([ze, te, ce], dim=1)  # B x (3*emb)
        x = torch.relu(self.fuse(cat))         # B x hidden
        if h_prev is None:
            h_prev = torch.zeros(zone_idx.size(0), self.hidden, device=zone_idx.device)
        h = self.gru(x, h_prev)  # B x hidden
        return h  # c_t

# ---------------------------
# Generator (outputs categorical distributions)
# ---------------------------
class GeneratorCategorical(nn.Module):
    def __init__(self, cdim=256, latent_dim=64, n_zones=11, n_types=11):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc = MLP(cdim * 3 + latent_dim, 512, hidden=512, n_layers=3)
        self.zone_head = nn.Linear(512, n_zones)   # logits for zone
        self.type_head = nn.Linear(512, n_types)   # logits for type

    def forward(self, c_t, mEM, mSM, z=None):
        if z is None:
            z = torch.randn(c_t.size(0), self.latent_dim, device=c_t.device)
        x = torch.cat([c_t, mEM, mSM, z], dim=1)
        h = F.relu(self.fc(x))
        zone_logits = self.zone_head(h)
        type_logits = self.type_head(h)
        zone_probs = F.softmax(zone_logits, dim=1)
        type_probs = F.softmax(type_logits, dim=1)
        return zone_logits, type_logits, zone_probs, type_probs

# ---------------------------
# Discriminator
# ---------------------------
class DiscriminatorCategorical(nn.Module):
    def __init__(self, n_zones=11, n_types=11, ctx_dim=6, emb_dim=64, hidden=256):
        super().__init__()

        self.zone_proj = nn.Linear(n_zones, emb_dim)
        self.type_proj = nn.Linear(n_types, emb_dim)

        # FIXED INPUT SIZE: zproj (64) + tproj (64) + ctx (6) = 134
        input_dim = emb_dim * 2 + ctx_dim   # 64 + 64 + 6 = 134

        self.cond_proj = MLP(input_dim, hidden, n_layers=2)

        self.adv_head = nn.Linear(hidden, 1)
        self.cls_head = nn.Linear(hidden, n_types)

    def forward(self, zone_vec, type_vec, zone_input, type_input, ctx_vec):
        zproj = self.zone_proj(zone_input)
        tproj = self.type_proj(type_input)
        x = torch.cat([zproj, tproj, ctx_vec], dim=1)
        h = F.relu(self.cond_proj(x))
        adv = torch.sigmoid(self.adv_head(h))
        cls_logits = self.cls_head(h)
        return adv, cls_logits

# ---------------------------
# MSS-GAN Trainer adapted to B1
# ---------------------------
class MSSGAN_B1_Trainer:
    def __init__(self, dataset, device=None,
                 n_zones=11, n_types=11,
                 embed_dim=256, latent_dim=64,
                 em_N=1100, em_l=3, sm_b=80):
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset
        # model components
        self.encoder = StepEncoder(n_zones=n_zones, n_types=n_types, emb_dim=64, ctx_dim=6, hidden=embed_dim).to(self.device)
        self.EM = EpisodicMemory(dim=embed_dim, mem_len=em_N, extract_depth=em_l).to(self.device)
        self.SM = SemanticMemory(dim=embed_dim, b=sm_b).to(self.device)
        self.G = GeneratorCategorical(cdim=embed_dim, latent_dim=latent_dim, n_zones=n_zones, n_types=n_types).to(self.device)
        self.D = DiscriminatorCategorical(n_zones=n_zones, n_types=n_types, ctx_dim=6, emb_dim=64, hidden=512).to(self.device)

        # optimizers
        g_params = list(self.encoder.parameters()) + list(self.G.parameters()) + list(self.SM.parameters())
        d_params = list(self.D.parameters())
        self.optG = torch.optim.Adam(g_params, lr=2e-4, betas=(0.5, 0.999))
        self.optD = torch.optim.Adam(d_params, lr=2e-4, betas=(0.5, 0.999))

        # losses
        self.bce = nn.BCELoss()
        self.ce = nn.CrossEntropyLoss(ignore_index=0)  # ignore pad if present
        self.latent_dim = latent_dim

    def _flatten_batch_steps(self, batch):
        # batch: list/dict from Dataset: each item has x_zone (T), x_type (T), context (6), y_target (T)
        # Convert to flattened per-step tensors:
        zones = []
        types = []
        ctxs = []
        tgt_zones = []
        tgt_types = []
        for item in batch:
            x_zone = item['x_zone'].numpy() if isinstance(item['x_zone'], torch.Tensor) else item['x_zone']
            x_type = item['x_type'].numpy() if isinstance(item['x_type'], torch.Tensor) else item['x_type']
            y_target = item['y_target'].numpy() if isinstance(item['y_target'], torch.Tensor) else item['y_target']
            ctx = item['context'].unsqueeze(0).numpy() if isinstance(item['context'], torch.Tensor) else item['context']
            T = len(x_zone)
            for i in range(T):
                if x_zone[i] == 0:  # padded position
                    continue
                # target zone for that shot is y_target[i] (may be 0 if missing)
                if y_target[i] == 0:
                    continue
                zones.append(int(x_zone[i]))
                types.append(int(x_type[i]))
                ctxs.append(ctx.reshape(-1))
                tgt_zones.append(int(y_target[i]))
                tgt_types.append(int(x_type[i]))  # we treat true shot type label as x_type at same index
        if len(zones) == 0:
            return None
        zones_t = torch.tensor(zones, dtype=torch.long)
        types_t = torch.tensor(types, dtype=torch.long)
        ctxs_t = torch.tensor(ctxs, dtype=torch.float32)
        tgt_zones_t = torch.tensor(tgt_zones, dtype=torch.long)
        tgt_types_t = torch.tensor(tgt_types, dtype=torch.long)
        return zones_t, types_t, ctxs_t, tgt_zones_t, tgt_types_t

    def train(self, epochs=10, batch_size=32, dataloader_workers=2):
        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True,
                            num_workers=dataloader_workers, collate_fn=lambda b: b)
        for ep in range(1, epochs + 1):
            total_d_loss = 0.0
            total_g_loss = 0.0
            cnt = 0
            for batch in loader:
                flat = self._flatten_batch_steps(batch)
                if flat is None:
                    continue
                zones_t, types_t, ctxs_t, tgt_zones_t, tgt_types_t = flat
                zones_t = zones_t.to(self.device)
                types_t = types_t.to(self.device)
                ctxs_t = ctxs_t.to(self.device)
                tgt_zones_t = tgt_zones_t.to(self.device)
                tgt_types_t = tgt_types_t.to(self.device)

                B = zones_t.size(0)
                # Encoder: get c_t (no recurrence over history is used here)
                c_t = self.encoder(zones_t, types_t, ctxs_t, h_prev=None)  # B x hidden

                # EM read (may be empty early in training)
                mEM = self.EM.read(c_t.to(self.device))  # B x hidden (on device)
                if isinstance(mEM, torch.Tensor):
                    mEM = mEM.to(self.device)
                else:
                    mEM = torch.zeros_like(c_t)

                # SM read
                mSM = self.SM.read(c_t)  # B x hidden

                # -------------------------
                # Discriminator update
                # -------------------------
                self.optD.zero_grad()
                # Real: build one-hot vectors for true targets
                n_zones = self.G.zone_head.out_features
                n_types = self.G.type_head.out_features
                real_zone_oh = F.one_hot(tgt_zones_t.clamp(min=0), num_classes=n_zones).float().to(self.device)
                real_type_oh = F.one_hot(tgt_types_t.clamp(min=0), num_classes=n_types).float().to(self.device)

                # D on real
                real_adv, real_logits = self.D(None, None, real_zone_oh, real_type_oh, ctxs_t)
                valid = torch.ones_like(real_adv)
                loss_real = self.bce(real_adv, valid)
                loss_cls_real = self.ce(real_logits, tgt_types_t)

                # Fake: generate
                z = torch.randn(B, self.latent_dim, device=self.device)
                zone_logits_fake, type_logits_fake, zone_probs_fake, type_probs_fake = self.G(c_t, mEM, mSM, z)
                fake_zone = zone_probs_fake.detach()  # treat as input probabilities
                fake_type = type_probs_fake.detach()
                fake_adv, fake_logits = self.D(None, None, fake_zone, fake_type, ctxs_t)
                fake_label = torch.zeros_like(fake_adv)
                loss_fake = self.bce(fake_adv, fake_label)

                lossD = loss_real + loss_fake + 0.5 * loss_cls_real
                lossD.backward()
                self.optD.step()

                # -------------------------
                # Generator + encoder update
                # -------------------------
                self.optG.zero_grad()
                # recompute generator outputs to get gradients through G & encoder
                z2 = torch.randn(B, self.latent_dim, device=self.device)
                zone_logits_fake2, type_logits_fake2, zone_probs_fake2, type_probs_fake2 = self.G(c_t, mEM, mSM, z2)
                fake_adv2, fake_logits2 = self.D(None, None, zone_probs_fake2, type_probs_fake2, ctxs_t)
                valid2 = torch.ones_like(fake_adv2)
                lossG_gan = self.bce(fake_adv2, valid2)
                # generator classification loss: encourage generated type to match true type
                lossG_cls = self.ce(type_logits_fake2, tgt_types_t)
                lossG = lossG_gan + 0.5 * lossG_cls
                lossG.backward()
                self.optG.step()

                # -------------------------
                # Memory updates
                # -------------------------
                # append c_t to EM (store on CPU to save GPU mem)
                for i in range(c_t.size(0)):
                    self.EM.append(c_t[i:i+1].detach().cpu())  # shape (1, dim)
                # update SM from mEM (use mEM as obtained earlier)
                try:
                    self.SM.update(mEM.detach().cpu())
                except Exception:
                    pass

                total_d_loss += lossD.item()
                total_g_loss += lossG.item()
                cnt += 1

            avg_d = total_d_loss / max(1, cnt)
            avg_g = total_g_loss / max(1, cnt)
            print(f"Epoch {ep}/{epochs} — D_loss: {avg_d:.4f}, G_loss: {avg_g:.4f}")

    @torch.no_grad()
    def evaluate(self, batch_size=64):
        self.G.eval()
        self.D.eval()
        self.EM.eval()
        
        # Use collate_fn=lambda b: b because _flatten_batch_steps expects a list of items
        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda b: b)
        
        all_zone_pred = []
        all_zone_true = []
        all_type_pred = []
        all_type_true = []
        
        for batch in loader:
            # 1. Flatten the batch into steps (just like in train)
            flat = self._flatten_batch_steps(batch)
            if flat is None:
                continue
            
            zones_t, types_t, ctxs_t, tgt_zones_t, tgt_types_t = flat
            
            # Move to device
            zones_t = zones_t.to(self.device)
            types_t = types_t.to(self.device)
            ctxs_t = ctxs_t.to(self.device)
            tgt_zones_t = tgt_zones_t.to(self.device)
            tgt_types_t = tgt_types_t.to(self.device)
            
            B = zones_t.size(0)
            
            # 2. Encoder: Get context embedding c_t
            c_t = self.encoder(zones_t, types_t, ctxs_t, h_prev=None)
            
            # 3. Read Memories (EM and SM)
            mEM = self.EM.read(c_t)
            # Handle case where EM might be empty or return None/zeros logic from your class
            if not isinstance(mEM, torch.Tensor): 
                mEM = torch.zeros_like(c_t)
            
            mSM = self.SM.read(c_t)
            
            # 4. Generate Predictions
            z = torch.randn(B, self.latent_dim, device=self.device)
            
            # G returns: zone_logits, type_logits, zone_probs, type_probs
            _, _, zone_probs, type_probs = self.G(c_t, mEM, mSM, z)
            
            # 5. Get hard predictions (argmax)
            zone_pred_idx = torch.argmax(zone_probs, dim=1)
            type_pred_idx = torch.argmax(type_probs, dim=1)
            
            # 6. Collect results
            all_zone_pred.extend(zone_pred_idx.cpu().numpy())
            all_zone_true.extend(tgt_zones_t.cpu().numpy())
            all_type_pred.extend(type_pred_idx.cpu().numpy())
            all_type_true.extend(tgt_types_t.cpu().numpy())
        
        print("----- Evaluation Results -----")
        self._compute_metrics(all_zone_true, all_zone_pred, "Zone")
        self._compute_metrics(all_type_true, all_type_pred, "Shot Type")
        print("--------------------------------")

    def _compute_metrics(self, true, pred, label):
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    
        acc = accuracy_score(true, pred)
        prec, rec, f1, _ = precision_recall_fscore_support(true, pred, average='macro', zero_division=0)
        cm = confusion_matrix(true, pred)
    
        print(f"▶ {label} Accuracy: {acc:.4f}")
        print(f"▶ {label} Precision: {prec:.4f}")
        print(f"▶ {label} Recall: {rec:.4f}")
        print(f"▶ {label} F1: {f1:.4f}")
        print(f"{label} Confusion Matrix:")
        print(cm)
        print()