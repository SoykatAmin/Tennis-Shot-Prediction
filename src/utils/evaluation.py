import torch
import numpy as np

class ModelAdapter:
    """Base class to standardize model inputs/outputs."""
    def __init__(self, model, device, dataset):
        self.model = model
        self.device = device
        self.dataset = dataset
        self.model.eval()

    def decode_batch(self, batch):
        """Must return (pred_type, pred_dir, pred_depth) and (true_type, true_dir, true_depth)"""
        raise NotImplementedError

class UnifiedAdapter(ModelAdapter):
    """For SingleHead FFN, RichLSTM, and CristianGPT (Unified)."""
    def __init__(self, model, device, dataset, uni_map):
        super().__init__(model, device, dataset)
        self.uni_map = uni_map # The dictionary mapping UnifiedID -> (Type, Dir, Depth)

    def decode_batch(self, batch):
        # 1. Handle varying input signatures via try/except or checks
        x_seq = batch['x_seq'].to(self.device)
        x_c = batch['context'].to(self.device)
        
        # Check if model expects rich inputs (RichLSTM / CristianGPT)
        if 'x_s_id' in batch and 'x_r_id' in batch:
            x_s = batch['x_s_id'].to(self.device)
            x_r = batch['x_r_id'].to(self.device)
            # Try passing rich inputs; fallback if model signature is different
            try:
                # Common signature for your Rich models
                if 'x_type' in batch: # RichLSTM style
                    logits = self.model(batch['x_type'].to(self.device), batch['x_dir'].to(self.device),
                                      batch['x_depth'].to(self.device), x_s, x_r, x_c)
                else: # CristianGPT style
                    logits = self.model(x_seq, x_c, x_s, x_r)
            except TypeError:
                logits = self.model(x_seq, x_c) # Baseline fallback
        else:
             logits = self.model(x_seq, x_c)

        preds = logits.argmax(dim=-1).cpu().numpy()
        targets = batch['y_target'].cpu().numpy()
        
        # Vectorized decoding using map (much faster than loops)
        # We assume uni_map is a dict. For speed, consider converting uni_map to a lookup array if vocab is static.
        p_t, p_d, p_dp = [], [], []
        t_t, t_d, t_dp = [], [], []
        
        for p, t in zip(preds.flatten(), targets.flatten()):
            pt, pd, pdp = self.uni_map.get(p, (0,0,0))
            tt, td, tdp = self.uni_map.get(t, (0,0,0))
            p_t.append(pt); p_d.append(pd); p_dp.append(pdp)
            t_t.append(tt); t_d.append(td); t_dp.append(tdp)
            
        return (np.array(p_t), np.array(p_d), np.array(p_dp)), \
               (np.array(t_t), np.array(t_d), np.array(t_dp))

class MultiHeadAdapter(ModelAdapter):
    """For Baseline MultiHead, Hierarchical, and Hybrid."""
    def decode_batch(self, batch):
        x_c = batch['context'].to(self.device)
        
        # Determine Input Signature
        if 'x_dir' in batch and 'x_type' in batch: # Hierarchical/Hybrid
            x_d = batch['x_dir'].to(self.device)
            x_t = batch['x_type'].to(self.device)
            # Check for Rich inputs
            if 'x_s_id' in batch:
                x_s = batch['x_s_id'].to(self.device)
                x_r = batch['x_r_id'].to(self.device)
                
                if 'x_depth' in batch: # Hybrid
                    x_dp = batch['x_depth'].to(self.device)
                    l_t, l_d, l_dp = self.model(x_t, x_d, x_dp, x_s, x_r, x_c)
                else: # Hierarchical
                    l_d, l_dp, l_t = self.model(x_d, x_t, x_c, x_s, x_r)
            else:
                 l_d, l_dp, l_t = self.model(x_d, x_t, x_c)
        else: # Baseline Multihead
            x_seq = batch['x_seq'].to(self.device)
            l_t, l_d, l_dp = self.model(x_seq, x_c)

        # Get Preds
        p_t = l_t.argmax(dim=-1).cpu().numpy().flatten()
        p_d = l_d.argmax(dim=-1).cpu().numpy().flatten()
        p_dp = l_dp.argmax(dim=-1).cpu().numpy().flatten()
        
        # Get Targets
        t_t = batch['y_type'].cpu().numpy().flatten()
        t_d = batch['y_dir'].cpu().numpy().flatten()
        t_dp = batch['y_depth'].cpu().numpy().flatten()
        
        return (p_t, p_d, p_dp), (t_t, t_d, t_dp)
    
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
import random

class TennisEvaluator:
    def __init__(self, adapter, test_loader, test_indices):
        self.adapter = adapter
        self.loader = test_loader
        self.dataset = adapter.dataset
        self.test_indices = test_indices
        
        # Standard Vocabs (Inverted)
        self.inv_type = {v: k for k, v in self.dataset.type_vocab.items()}
        self.inv_dir = {v: k for k, v in self.dataset.dir_vocab.items()}
        self.inv_depth = {v: k for k, v in self.dataset.depth_vocab.items()}
        
    def run_all(self):
        """Runs the entire pipeline."""
        self.part1_tactical_metrics()
        self.part6_surface_analysis()
        # Add other parts (2, 3, 4, 5) similarly...
        print("Evaluation Complete.")

    def part1_tactical_metrics(self):
        print("\n" + "="*40 + "\n PART 1: OVERALL TACTICAL METRICS \n" + "="*40)
        all_pt, all_pd, all_pdp = [], [], []
        all_tt, all_td, all_tdp = [], [], []
        
        with torch.no_grad():
            for batch in self.loader:
                (pt, pd, pdp), (tt, td, tdp) = self.adapter.decode_batch(batch)
                
                # Filter Masks (Ignore 0)
                mask = (tt != 0)
                if mask.sum() > 0:
                    all_pt.extend(pt[mask]); all_tt.extend(tt[mask])
                    all_pd.extend(pd[mask]); all_td.extend(td[mask])
                    all_pdp.extend(pdp[mask]); all_tdp.extend(tdp[mask])

        # Reports
        self._print_report(all_tt, all_pt, self.inv_type, "SHOT TYPE")
        self._print_report(all_td, all_pd, self.inv_dir, "DIRECTION")
        self._print_report(all_tdp, all_pdp, self.inv_depth, "DEPTH")

    def _print_report(self, y_true, y_pred, inv_vocab, title):
        print(f"\n=== {title} REPORT ===")
        labels = sorted([k for k in inv_vocab.keys() if k in np.unique(y_true) and k != 0])
        names = [inv_vocab[k] for k in labels]
        print(classification_report(y_true, y_pred, labels=labels, target_names=names, zero_division=0))

    def part6_surface_analysis(self):
        print("\n" + "="*40 + "\n PART 6: SURFACE ANALYSIS \n" + "="*40)
        # 1. Map indices to surfaces
        surf_map = {'Hard': [], 'Clay': [], 'Grass': []}
        for idx in self.test_indices:
            mid = self.dataset.sample_match_ids[idx]
            s = self.dataset.match_meta.get(mid, {}).get('surface', 'Hard')
            if 'Clay' in s: surf_map['Clay'].append(idx)
            elif 'Grass' in s: surf_map['Grass'].append(idx)
            else: surf_map['Hard'].append(idx)

        results = []
        # Reuse the batch logic by subsetting
        for surf, indices in surf_map.items():
            if not indices: continue
            # Create a temp loader for this surface
            subset = torch.utils.data.Subset(self.dataset, indices[:2000]) # Limit samples for speed
            loader = torch.utils.data.DataLoader(subset, batch_size=256, shuffle=False)
            
            with torch.no_grad():
                for batch in loader:
                    (pt, pd, pdp), (tt, td, tdp) = self.adapter.decode_batch(batch)
                    
                    # Calculate vectorized errors
                    mask = (tt != 0)
                    t_err = (pt[mask] != tt[mask]).astype(float)
                    d_err = (pd[mask] != td[mask]).astype(float)
                    dp_err = (pdp[mask] != tdp[mask]).astype(float) # Note: You might want deeper masking logic here
                    
                    for i in range(len(t_err)):
                        results.append({
                            'Surface': surf,
                            'Type Error': t_err[i],
                            'Direction Error': d_err[i],
                            'Depth Error': dp_err[i]
                        })
        
        if results:
            df = pd.DataFrame(results)
            print(df.groupby('Surface').mean() * 100)
            # Plot
            df_melt = df.melt(id_vars=['Surface'], value_vars=['Type Error', 'Direction Error', 'Depth Error'], value_name='Error')
            plt.figure(figsize=(10,5))
            sns.barplot(data=df_melt, x='Surface', y='Error', hue='variable', palette='viridis')
            plt.title("Error Rates by Surface")
            plt.show()

def get_universal_decoder_map(dataset):
    """
    Creates a lookup table: Unified_ID -> (Type_ID, Dir_ID, Depth_ID).
    Robustly handles Serves, Specials, and standard shots.
    """
    # 1. Define Standard Mappings (Ensure these match your EVAL_*_VOCABs)
    # You can pass these as arguments if they change, but usually they are static.
    EVAL_SHOT_VOCAB = {'<pad>': 0, 'f': 1, 'b': 2, 'r': 3, 'v': 4, 'o': 5, 's': 6, 'u': 7, 'l': 8, 'm': 9, 'z': 10}
    EVAL_DIR_VOCAB  = {'<pad>': 0, '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6}
    EVAL_DEPTH_VOCAB = {'<pad>': 0, '0': 0, '7': 1, '8': 2, '9': 3}
    
    uni_map = {}
    serve_type_id = EVAL_SHOT_VOCAB.get('s', 6)
    
    # 2. Iterate through the dataset's vocabulary
    # Ensure we use the correct inverse vocab attribute
    if hasattr(dataset, 'inv_unified_vocab'):
        inv_vocab = dataset.inv_unified_vocab
    else:
        # Fallback if inverse dict isn't pre-built
        inv_vocab = {v: k for k, v in dataset.unified_vocab.items()}

    for uid, key in inv_vocab.items():
        # A. Handle Padding / Unknowns (IDs 0 and 1)
        if uid <= 1: 
            uni_map[uid] = (0, 0, 0)
            continue
            
        parts = key.split('_')
        token_type = parts[0]
        
        # B. Handle Serves (e.g., "Serve_4")
        if token_type == 'Serve' or token_type == 'S':
            # Format: (ServeType, Direction, Depth=0)
            d_str = parts[1] if len(parts) > 1 else '0'
            uni_map[uid] = (
                serve_type_id, 
                EVAL_DIR_VOCAB.get(d_str, 0), 
                0
            )
            
        # C. Handle Specials (e.g., "LET", "SPECIAL") -> Map to Padding (0)
        elif token_type in ['LET', 'SPECIAL', 'PAD', 'UNK']:
            uni_map[uid] = (0, 0, 0)
            
        # D. Handle Standard Shots (e.g., "f_2_8" or "b_1")
        else:
            # Safe parsing: use defaults if parts are missing
            t_str = parts[0]
            d_str = parts[1] if len(parts) > 1 else '0'
            dp_str = parts[2] if len(parts) > 2 else '0'
            
            uni_map[uid] = (
                EVAL_SHOT_VOCAB.get(t_str, 0),
                EVAL_DIR_VOCAB.get(d_str, 0),
                EVAL_DEPTH_VOCAB.get(dp_str, 0)
            )
            
    return uni_map