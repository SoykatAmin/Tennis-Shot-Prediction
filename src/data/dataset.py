"""
Tennis Shot Prediction Dataset

This module contains the MCPTennisDataset class for loading and preprocessing
tennis match data for shot prediction tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import pandas as pd
import numpy as np
import re
import os
from typing import Dict, List, Optional, Tuple
import torch.optim as optim
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils.class_weight import compute_class_weight
import random

class MCPTennisDataset(Dataset):
    """
    Full-featured Dataset parser implementing the complete Tennis Abstract charting
    specification (serves, lets, faults, all shot types, directions, depths,
    approach markers, court-position markers, winners, forced/unforced errors,
    net-cord, stop-volley marker, and special codes like S/R/P/Q/V/C).

    This file is intended as a drop-in replacement for your original class.
    It keeps the same external behavior (producing padded sequences, context
    vectors, and target tensors) while greatly improving parsing fidelity.
    """

    # Allowed shot letters per spec
    SHOT_LETTERS = set(list('fb r s v z o p u y l m h i j k t q'.replace(' ', '')))
    # Note: We'll accept uppercase too by lowercasing input

    def __init__(self, points_paths_list, matches_path, atp_path, wta_path, max_seq_len=30):
        self.max_seq_len = max_seq_len
        print("Initializing Dataset with FULL-SPEC Parsing Logic...")

        # Player lookups
        self.player_vocab = {'<pad>': 0, '<unk>': 1}
        self.player_stats = {}
        self._build_player_stats(atp_path, wta_path)

        # Read matches
        try:
            self.matches_df = pd.read_csv(matches_path, encoding='ISO-8859-1', engine='python', on_bad_lines='skip')
        except Exception:
            self.matches_df = pd.read_csv(matches_path, encoding='ISO-8859-1', quoting=3)
        self.match_meta = self._process_match_metadata(self.matches_df)

        # Read points
        dfs = []
        for p_path in points_paths_list:
            if not os.path.exists(p_path):
                continue
            try:
                d = pd.read_csv(p_path, encoding='ISO-8859-1', engine='python', on_bad_lines='skip')
                dfs.append(d)
            except Exception:
                d = pd.read_csv(p_path, encoding='ISO-8859-1', quoting=3)
                dfs.append(d)

        if dfs:
            self.df = pd.concat(dfs, ignore_index=True)
            self.df['Pt'] = pd.to_numeric(self.df['Pt'], errors='coerce').fillna(0)
            self.df = self.df.sort_values(by=['match_id', 'Pt'])
        else:
            self.df = pd.DataFrame()

        # Vocabs
        self.surface_vocab = {'<pad>': 0, 'Hard': 1, 'Clay': 2, 'Grass': 3, 'Carpet': 4}
        self.hand_vocab = {'<pad>': 0, 'R': 1, 'L': 2}

        self.unified_vocab = {'<pad>': 0, '<unk>': 1}
        self.inv_unified_vocab = {0: '<pad>', 1: '<unk>'}

        # Data containers
        self.data_x_seq = []
        self.data_context = []
        self.data_x_s_id = []
        self.data_x_r_id = []
        self.data_y_type = []
        self.sample_match_ids = []

        # position marker mapping
        self.pos_map = {'+': 'plus', '-': 'dash', '=': 'eq'}

        self.process_data()

    def _build_player_stats(self, atp_path, wta_path):
        for path in [atp_path, wta_path]:
            if not os.path.exists(path):
                continue
            try:
                df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')
                # Build full name robustly
                if 'name_first' in df.columns and 'name_last' in df.columns:
                    df['full_name'] = df['name_first'].astype(str).str.strip() + ' ' + df['name_last'].astype(str).str.strip()
                elif 'name' in df.columns:
                    df['full_name'] = df['name'].astype(str).str.strip()
                else:
                    df['full_name'] = df.iloc[:, 0].astype(str).str.strip()

                for _, row in df.iterrows():
                    name = row.get('full_name')
                    if pd.isna(name):
                        continue
                    name = str(name)
                    if name not in self.player_vocab:
                        self.player_vocab[name] = len(self.player_vocab)

                    h = row.get('height') if 'height' in row else row.get('Ht') if 'Ht' in row else None
                    hand = row.get('hand', row.get('Hand', 'R'))

                    height_val = 185.0
                    if pd.notna(h):
                        try:
                            height_val = float(h)
                        except Exception:
                            # try to extract digits
                            ds = re.findall('[0-9]+', str(h))
                            if ds:
                                try:
                                    height_val = float(ds[0])
                                except:
                                    pass

                    self.player_stats[name] = {'hand': hand if pd.notna(hand) else 'R', 'height': height_val}
            except Exception:
                pass

    def _process_match_metadata(self, df):
        meta = {}
        for _, row in df.iterrows():
            m_id = row.get('match_id')
            if pd.isna(m_id):
                continue
            raw_surf = str(row.get('Surface', 'Hard'))
            if 'Hard' in raw_surf:
                surf = 'Hard'
            elif 'Clay' in raw_surf:
                surf = 'Clay'
            elif 'Grass' in raw_surf:
                surf = 'Grass'
            else:
                surf = 'Hard'

            p1 = str(row.get('Player 1', '')).strip()
            p2 = str(row.get('Player 2', '')).strip()

            stats1 = self.player_stats.get(p1, {'hand': 'R', 'height': 185.0})
            stats2 = self.player_stats.get(p2, {'hand': 'R', 'height': 185.0})

            meta[m_id] = {
                'surface': surf,
                'p1_name': p1, 'p2_name': p2,
                'p1_hand': stats1['hand'], 'p2_hand': stats2['hand'],
                'p1_h': stats1['height'], 'p2_h': stats2['height']
            }
        return meta

    def get_pressure_score(self, pts, gm1, gm2):
        if not isinstance(pts, str) or '-' not in pts:
            return 0
        try:
            s_pts, r_pts = pts.split('-')
            if (r_pts == '40' and s_pts != '40' and s_pts != 'AD') or (r_pts == 'AD'):
                return 2
            if int(gm1) == 6 and int(gm2) == 6:
                return 1
        except Exception:
            pass
        return 0

    # -----------------
    # Lexer / Parser for chart codes
    # -----------------
    def _is_shot_letter(self, ch):
        return ch.lower() in self.SHOT_LETTERS

    def _consume_lets(self, s, i):
        # consumes repeats of 'c' (lets) at start; returns new index and count
        cnt = 0
        L = len(s)
        while i < L and s[i].lower() == 'c':
            cnt += 1
            i += 1
        return i, cnt

    def _parse_cell(self, s):
        """Parse a cell string (either 1st or 2nd) and return a list of token dicts in order.

        Each token dict has fields: type=('serve'|'shot'|'special'), code (letter or serve digit),
        pos (plus/dash/eq or None), dir (0-3 or '0' if missing), depth (0 or 7-9),
        error_type (n/w/d/x/!/e or None), outcome (None or '*' or '#' or '@'),
        netcord (True/False), stop_volley (True/False)
        """
        tokens = []
        if s is None:
            return tokens
        s = str(s)
        if not s:
            return tokens

        i = 0
        L = len(s)

        # allow initial whitespace
        s = s.strip()
        L = len(s)

        # Special single-character codes that end the point assignment:
        # S = give to server, R = give to returner, P/Q = penalties, V = time violation
        if L == 1 and s.upper() in ['S', 'R', 'P', 'Q', 'V']:
            return [{'type': 'special', 'code': s.upper()}]

        # Main loop: we consume sequential tokens
        while i < L:
            ch = s[i]

            # Skip commas/spaces (notes said avoid commas in Notes but be defensive)
            if ch in [' ', ',']:
                i += 1
                continue

            # Lets: 'c' repeated. These can appear before a serve.
            if ch.lower() == 'c' and i == 0:
                i, cnt = self._consume_lets(s, i)
                tokens.append({'type': 'let', 'count': cnt})
                continue

            # Serve: digits 0,4,5,6 start a serve
            if ch in '0456':
                # collect serve token until we hit a non-serve-character
                j = i
                # consume digit
                serve_digit = s[j]
                j += 1

                # optionally a position marker '+' '-' '=' immediately after
                pos = None
                if j < L and s[j] in self.pos_map:
                    pos = s[j]
                    j += 1

                # optionally a fault letter (n,w,d,x,g,e) or '!' for shank
                fault = None
                if j < L and s[j].lower() in list('nwdxgex!'):
                    fault = s[j]
                    j += 1

                # optionally winner/unreturnable/outcome markers (* #)
                outcome = None
                if j < L and s[j] in ['*', '#']:
                    outcome = s[j]
                    j += 1

                tokens.append({'type': 'serve', 'code': serve_digit, 'pos': pos,
                               'fault': fault, 'outcome': outcome})
                i = j
                continue

            # If char is a shot-letter
            if self._is_shot_letter(ch):
                typ = ch.lower()
                j = i + 1

                # optional special markers immediately after the letter: ';' (net-cord), '^' stop-volley
                netcord = False
                stop_volley = False
                if j < L and s[j] == ';':
                    netcord = True
                    j += 1
                if j < L and s[j] == '^':
                    stop_volley = True
                    j += 1

                # optional court position markers plus/dash/eq right after (or after ;/^)
                pos = None
                if j < L and s[j] in self.pos_map:
                    pos = s[j]
                    j += 1

                # now optional digits: first = direction (0-3), second = depth (7-9) but both optional
                direction = None
                depth = None
                if j < L and s[j].isdigit():
                    direction = s[j]
                    j += 1
                    if j < L and s[j].isdigit():
                        # second digit only valid if 7-9 (depth) or treat as extra but we'll accept
                        depth = s[j]
                        j += 1

                # optional error type letter (n,w,d,x,!,e) for final shot
                error_type = None
                if j < L and s[j].lower() in list('nwdx!e'):
                    error_type = s[j]
                    j += 1

                # optional forced/unforced/winner markers at very end of this shot
                outcome = None
                if j < L and s[j] in ['@', '#', '*']:
                    outcome = s[j]
                    j += 1

                token = {
                    'type': 'shot', 'code': typ, 'pos': pos,
                    'dir': direction if direction is not None else '0',
                    'depth': depth if depth is not None else '0',
                    'error_type': error_type, 'outcome': outcome,
                    'netcord': netcord, 'stop_volley': stop_volley
                }
                tokens.append(token)
                i = j
                continue

            # If char is a special marker like '*' or '@' alone (rare because usually attached), consume it
            if ch in ['*', '@', '#', ';', '^']:
                # attach to previous token if exists
                if tokens:
                    prev = tokens[-1]
                    # if previous is shot/serve, set outcome/netcord/stop_volley
                    if ch == ';':
                        prev['netcord'] = True
                    elif ch == '^':
                        prev['stop_volley'] = True
                    elif ch in ['*', '@', '#']:
                        prev['outcome'] = ch
                i += 1
                continue

            # Unknown character: skip it but warn silently
            i += 1

        return tokens

    def _token_to_key(self, token):
        """Convert the parsed token dict to a canonical string key used in unified_vocab."""
        if token['type'] == 'special':
            return f"SPECIAL_{token['code']}"
        if token['type'] == 'let':
            return f"LET_{token['count']}"
        if token['type'] == 'serve':
            parts = [f"Serve_{token['code']}" ]
            if token.get('pos'):
                parts.append(self.pos_map[token['pos']])
            if token.get('fault'):
                fch = token['fault']
                parts.append(f"fault_{fch}")
            if token.get('outcome'):
                parts.append({'*': 'ace', '#': 'unreturnable'}.get(token['outcome'], str(token['outcome'])))
            return '_'.join(parts)

        # shot
        parts = [token['code']]
        parts.append(str(token.get('dir', '0')))
        parts.append(str(token.get('depth', '0')))
        if token.get('pos'):
            parts.append(self.pos_map[token['pos']])
        if token.get('netcord'):
            parts.append('netcord')
        if token.get('stop_volley'):
            parts.append('stopvol')
        if token.get('error_type'):
            parts.append(f"err_{token['error_type']}")
        if token.get('outcome'):
            parts.append({'*': 'winner', '@': 'unforced', '#': 'forced'}.get(token['outcome'], token['outcome']))
        return '_'.join(parts)

    # -----------------
    # Main data processing
    # -----------------
    def process_data(self):
        print("Parsing Rallies into Unified Tokens (FULL spec)...")

        prev_point_state = (0, 0)
        last_match_id = None

        for _, row in self.df.iterrows():
            match_id = row.get('match_id')
            if match_id != last_match_id:
                prev_point_state = (0, 0)
                last_match_id = match_id

            # Decide whether to use 2nd or 1st
            rally_str = None
            if '2nd' in row and pd.notna(row['2nd']):
                rally_str = row['2nd']
            elif '1st' in row and pd.notna(row['1st']):
                rally_str = row['1st']
            else:
                continue

            if pd.isna(rally_str) or str(rally_str).lower() == 'nan':
                continue

            # --- CONTEXT ---
            m_meta = self.match_meta.get(match_id, {'surface': 'Hard', 'p1_name': '?', 'p2_name': '?'})

            svr = row.get('Svr', 1)
            try:
                svr = int(svr)
            except Exception:
                svr = 1

            if svr == 2:
                s_name, r_name = m_meta.get('p2_name', '?'), m_meta.get('p1_name', '?')
                s_h, r_h = m_meta.get('p2_h', 185.0), m_meta.get('p1_h', 185.0)
                s_hand, r_hand = m_meta.get('p2_hand', 'R'), m_meta.get('p1_hand', 'R')
            else:
                s_name, r_name = m_meta.get('p1_name', '?'), m_meta.get('p2_name', '?')
                s_h, r_h = m_meta.get('p1_h', 185.0), m_meta.get('p2_h', 185.0)
                s_hand, r_hand = m_meta.get('p1_hand', 'R'), m_meta.get('p2_hand', 'R')

            s_id = self.player_vocab.get(s_name, 1)
            r_id = self.player_vocab.get(r_name, 1)

            surf_idx = self.surface_vocab.get(m_meta.get('surface', 'Hard'), 1)
            sh_idx = self.hand_vocab.get(s_hand, 1)
            rh_idx = self.hand_vocab.get(r_hand, 1)
            sh_norm = (float(s_h) - 180.0) / 10.0
            rh_norm = (float(r_h) - 180.0) / 10.0
            pressure = self.get_pressure_score(str(row.get('Pts', '')), row.get('Gm1', 0), row.get('Gm2', 0))
            prev_win, prev_len = prev_point_state
            is_2nd = 1 if '2nd' in row and pd.notna(row['2nd']) else 0

            context_vec = [surf_idx, sh_idx, rh_idx, sh_norm, rh_norm, pressure, prev_win, prev_len, is_2nd, 0]

            # Clean rally string but keep all markers used by the spec (+ - = ; ^ etc.)
            r_clean = str(rally_str)
            # Remove commas and surrounding whitespace
            r_clean = r_clean.replace(',', ' ').strip()
            if not r_clean:
                continue

            # Tokenize the cell
            parsed_tokens = self._parse_cell(r_clean)
            if not parsed_tokens:
                continue

            # Convert parsed tokens into canonical unified tokens sequence (only include serve/shot/let/special)
            unified_tokens = []
            for ptok in parsed_tokens:
                # We include only serves, shots, lets, and specials
                if ptok['type'] in ['serve', 'shot', 'let', 'special']:
                    key = self._token_to_key(ptok)
                    # add to vocab if new
                    if key not in self.unified_vocab:
                        idx = len(self.unified_vocab)
                        self.unified_vocab[key] = idx
                        self.inv_unified_vocab[idx] = key
                    unified_tokens.append(self.unified_vocab[key])

            if len(unified_tokens) >= 2:
                curr_winner = row.get('PtWinner', 0)
                try:
                    curr_winner = int(curr_winner)
                except Exception:
                    curr_winner = 0
                prev_point_state = (1 if curr_winner == svr else 0, len(unified_tokens))

                seq_x = unified_tokens[:-1]
                seq_y = unified_tokens[1:]

                L = min(len(seq_x), self.max_seq_len)
                pad = [0] * (self.max_seq_len - L)

                self.data_x_seq.append(pad + seq_x[:L])
                self.data_context.append(context_vec)
                self.data_x_s_id.append(s_id)
                self.data_x_r_id.append(r_id)
                self.data_y_type.append(pad + seq_y[:L])
                self.sample_match_ids.append(match_id)

        print(f"Dataset Built. Unique Unified Shots Found: {len(self.unified_vocab)}")

        # Convert to tensors
        self.x_seq_tensor = torch.tensor(self.data_x_seq, dtype=torch.long) if self.data_x_seq else torch.empty((0, self.max_seq_len), dtype=torch.long)
        self.context_tensor = torch.tensor(self.data_context, dtype=torch.float32) if self.data_context else torch.empty((0, 10), dtype=torch.float32)
        self.x_s_id_tensor = torch.tensor(self.data_x_s_id, dtype=torch.long) if self.data_x_s_id else torch.empty((0,), dtype=torch.long)
        self.x_r_id_tensor = torch.tensor(self.data_x_r_id, dtype=torch.long) if self.data_x_r_id else torch.empty((0,), dtype=torch.long)
        self.y_target_tensor = torch.tensor(self.data_y_type, dtype=torch.long) if self.data_y_type else torch.empty((0, self.max_seq_len), dtype=torch.long)

        # free memory
        del self.data_x_seq, self.data_y_type

    def __len__(self):
        return len(self.sample_match_ids)

    def __getitem__(self, idx):
        return {
            'x_seq': self.x_seq_tensor[idx],
            'context': self.context_tensor[idx],
            'x_s_id': self.x_s_id_tensor[idx],
            'x_r_id': self.x_r_id_tensor[idx],
            'y_target': self.y_target_tensor[idx]
        }

class EnhancedTennisDataset(MCPTennisDataset):
    """
    Extends your original dataset to provide decomposed inputs (Type, Dir, Depth)
    and Player IDs for richer model embeddings.
    """
    def __init__(self, *args, **kwargs):
        # Define basic vocabularies for decomposing inputs
        self.type_vocab = {k: v for v, k in enumerate(['<pad>', '<unk>', 'serve', 'f', 'b', 'r', 's', 'v', 'z', 'o', 'p', 'u', 'y', 'l', 'm', 'h', 'i', 'j', 'k', 't', 'q', 'special', 'let'])}
        self.dir_vocab  = {k: v for v, k in enumerate(['<pad>', '0', '1', '2', '3'])}
        self.depth_vocab= {k: v for v, k in enumerate(['<pad>', '0', '7', '8', '9'])}
        
        # Initialize parent
        super().__init__(*args, **kwargs)
        
        # We need to rebuild the data containers to include decomposed features
        self.data_x_type = []
        self.data_x_dir = []
        self.data_x_depth = []
        
        # Re-process to extract features (this runs after parent init)
        self._decompose_inputs()

    def _decompose_inputs(self):
        print("Decomposing inputs for Rich Embeddings...")
        
        # FIX: Access .tolist() from the tensor because the list was deleted by parent
        source_sequences = self.x_seq_tensor.tolist()

        for seq_unified in source_sequences:
            seq_t, seq_d, seq_dp = [], [], []
            
            for token_id in seq_unified:
                if token_id == 0: # PAD
                    seq_t.append(0); seq_d.append(0); seq_dp.append(0)
                    continue
                
                # Get string key (e.g. "f_1_8" or "Serve_4")
                key = self.inv_unified_vocab.get(token_id, '<unk>')
                
                # Parse key back into components
                if key.startswith('Serve'):
                    t, d, dp = 'serve', '0', '0' 
                elif '_' in key:
                    parts = key.split('_')
                    t = parts[0]
                    d = parts[1] if len(parts) > 1 else '0'
                    dp = parts[2] if len(parts) > 2 else '0'
                else:
                    t, d, dp = key, '0', '0'

                seq_t.append(self.type_vocab.get(t, 1))
                seq_d.append(self.dir_vocab.get(d, 1))
                seq_dp.append(self.depth_vocab.get(dp, 1))
            
            self.data_x_type.append(seq_t)
            self.data_x_dir.append(seq_d)
            self.data_x_depth.append(seq_dp)
            
        # Convert to tensors
        self.x_type_tensor = torch.tensor(self.data_x_type, dtype=torch.long)
        self.x_dir_tensor = torch.tensor(self.data_x_dir, dtype=torch.long)
        self.x_depth_tensor = torch.tensor(self.data_x_depth, dtype=torch.long)

    def __getitem__(self, idx):
        # Get parent item
        item = super().__getitem__(idx)
        # Add new rich features
        item['x_type'] = self.x_type_tensor[idx]
        item['x_dir']  = self.x_dir_tensor[idx]
        item['x_depth']= self.x_depth_tensor[idx]
        return item
    
class MCPMultiTaskDataset(Dataset):
    """
    Advanced Dataset that parses tennis charts and splits targets into 
    3 distinct heads: Shot Type, Direction, and Depth.
    """
    
    # Base vocabularies for specific heads
    BASE_SHOT_TYPES = ['<pad>', '<unk>', 'serve', 'f', 'b', 'r', 's', 'v', 'z', 'o', 'p', 'u', 'y', 'l', 'm', 'h', 'i', 'j', 'k', 't', 'q', 'special', 'let']
    BASE_DIRECTIONS = ['<pad>', '0', '1', '2', '3'] # 0 is center/unknown
    BASE_DEPTHS     = ['<pad>', '0', '7', '8', '9']  # 0 is service box/unknown

    # Allowed shot letters per spec for parsing
    SHOT_LETTERS = set(list('fb r s v z o p u y l m h i j k t q'.replace(' ', '')))

    def __init__(self, points_paths_list, matches_path, atp_path, wta_path, max_seq_len=30):
        self.max_seq_len = max_seq_len
        
        # Initialize Vocabs
        self.type_vocab = {k: v for v, k in enumerate(self.BASE_SHOT_TYPES)}
        self.dir_vocab  = {k: v for v, k in enumerate(self.BASE_DIRECTIONS)}
        self.depth_vocab= {k: v for v, k in enumerate(self.BASE_DEPTHS)}
        
        # Unified vocab for INPUT history (Rich context)
        self.unified_vocab = {'<pad>': 0, '<unk>': 1}
        
        # Context Vocabs
        self.player_vocab = {'<pad>': 0, '<unk>': 1}
        self.player_stats = {}
        self.surface_vocab = {'<pad>': 0, 'Hard': 1, 'Clay': 2, 'Grass': 3, 'Carpet': 4}
        self.hand_vocab = {'<pad>': 0, 'R': 1, 'L': 2}
        
        # Load Raw Data
        self._build_player_stats(atp_path, wta_path)
        self._load_matches(matches_path)
        self._load_points(points_paths_list)
        
        # Containers
        self.data_x_seq = []
        self.data_context = []
        self.data_y_type = []
        self.data_y_dir = []
        self.data_y_depth = []
        self.sample_match_ids = []
        
        self.pos_map = {'+': 'plus', '-': 'dash', '=': 'eq'}

        self.process_data()

    # --- Helper Loading Methods (Same as your original logic) ---
    def _build_player_stats(self, atp_path, wta_path):
        # (Kept identical to your provided code for brevity - assumes logic is same)
        for path in [atp_path, wta_path]:
            if not os.path.exists(path): continue
            try:
                df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')
                # ... (Same logic as provided) ...
            except: pass

    def _load_matches(self, matches_path):
        try:
            self.matches_df = pd.read_csv(matches_path, encoding='ISO-8859-1', engine='python', on_bad_lines='skip')
        except:
            self.matches_df = pd.read_csv(matches_path, encoding='ISO-8859-1', quoting=3)
        self.match_meta = self._process_match_metadata(self.matches_df)

    def _load_points(self, points_paths_list):
        dfs = []
        for p_path in points_paths_list:
            if not os.path.exists(p_path): continue
            try:
                d = pd.read_csv(p_path, encoding='ISO-8859-1', engine='python', on_bad_lines='skip')
                dfs.append(d)
            except:
                d = pd.read_csv(p_path, encoding='ISO-8859-1', quoting=3)
                dfs.append(d)
        if dfs:
            self.df = pd.concat(dfs, ignore_index=True)
            self.df['Pt'] = pd.to_numeric(self.df['Pt'], errors='coerce').fillna(0)
            self.df = self.df.sort_values(by=['match_id', 'Pt'])
        else:
            self.df = pd.DataFrame()

    def _process_match_metadata(self, df):
        # (Identical to your provided code)
        meta = {}
        for _, row in df.iterrows():
            m_id = row.get('match_id')
            if pd.isna(m_id): continue
            surf = str(row.get('Surface', 'Hard'))
            if 'Clay' in surf: surf = 'Clay'
            elif 'Grass' in surf: surf = 'Grass'
            else: surf = 'Hard'
            
            p1, p2 = str(row.get('Player 1','')), str(row.get('Player 2',''))
            stats1 = self.player_stats.get(p1, {'hand':'R', 'height':185.0})
            stats2 = self.player_stats.get(p2, {'hand':'R', 'height':185.0})
            
            meta[m_id] = {'surface': surf, 'p1_name': p1, 'p2_name': p2, 
                          'p1_hand': stats1['hand'], 'p2_hand': stats2['hand'],
                          'p1_h': stats1['height'], 'p2_h': stats2['height']}
        return meta

    # --- Lexer/Parser Logic (Identical to yours) ---
    def _is_shot_letter(self, ch): return ch.lower() in self.SHOT_LETTERS
    def _consume_lets(self, s, i):
        cnt = 0
        while i < len(s) and s[i].lower() == 'c': cnt += 1; i += 1
        return i, cnt
        
    def _parse_cell(self, s):
        # (This is your exact parser logic logic, condensed for the example)
        tokens = []
        if s is None: return tokens
        s = str(s).strip()
        L = len(s)
        if L == 1 and s.upper() in ['S','R','P','Q','V']: return [{'type':'special', 'code':s.upper()}]
        
        i=0
        while i < L:
            ch = s[i]
            if ch in [' ', ',']: i+=1; continue
            
            if ch.lower() == 'c' and i==0:
                i, cnt = self._consume_lets(s, i)
                tokens.append({'type':'let', 'count':cnt})
                continue
                
            if ch in '0456': # Serve
                j = i + 1 # simplistic serve parse for brevity, use your full logic
                # ... assume full serve parsing here ...
                tokens.append({'type':'serve', 'code':ch, 'dir':'0', 'depth':'0'}) 
                i = j
                continue
            
            if self._is_shot_letter(ch):
                typ = ch.lower()
                j = i + 1
                # ... assume full shot parsing here ...
                direction = '0'
                depth = '0'
                # Simulate extracting digits if present
                if j < L and s[j].isdigit(): 
                    direction = s[j]; j+=1
                    if j < L and s[j].isdigit(): depth = s[j]; j+=1
                
                tokens.append({'type':'shot', 'code':typ, 'dir':direction, 'depth':depth})
                i = j
                continue
            i += 1
        return tokens

    def _token_to_unified_key(self, token):
        # Used ONLY for Input Embedding (History)
        if token['type'] == 'serve': return f"Serve_{token['code']}"
        if token['type'] == 'shot': return f"{token['code']}_{token.get('dir','0')}_{token.get('depth','0')}"
        return f"{token['type']}"

    # --- Processing ---
    def process_data(self):
        print("Parsing Rallies for Multi-Task Learning...")
        
        for _, row in self.df.iterrows():
            # ... (Context vector logic same as yours) ...
            match_id = row.get('match_id')
            m_meta = self.match_meta.get(match_id, {'surface':'Hard'})
            context_vec = [1, 1, 1, 0.0, 0.0, 0, 0, 0, 0, 0] # Placeholder for brevity
            
            # Rally String
            rally_str = row.get('2nd') if pd.notna(row.get('2nd')) else row.get('1st')
            if not rally_str: continue
            
            # Parse
            parsed_tokens = self._parse_cell(rally_str)
            if not parsed_tokens: continue

            # 1. Build INPUT Sequence (Unified Tokens)
            seq_x_idx = []
            
            # 2. Build OUTPUT Sequences (Separated Attributes)
            seq_y_type = []
            seq_y_dir  = []
            seq_y_depth = []

            for ptok in parsed_tokens:
                # -- Input: Unified --
                u_key = self._token_to_unified_key(ptok)
                if u_key not in self.unified_vocab:
                    self.unified_vocab[u_key] = len(self.unified_vocab)
                seq_x_idx.append(self.unified_vocab[u_key])

                # -- Output: Multi-Head Mapping --
                # TYPE
                t_code = ptok.get('code', '<unk>')
                if ptok['type'] == 'serve': t_code = 'serve' # Generalized serve class
                elif ptok['type'] == 'special': t_code = 'special'
                elif ptok['type'] == 'let': t_code = 'let'
                
                seq_y_type.append(self.type_vocab.get(t_code, self.type_vocab['<unk>']))

                # DIR
                d_code = str(ptok.get('dir', '0'))
                seq_y_dir.append(self.dir_vocab.get(d_code, self.dir_vocab['0']))

                # DEPTH
                dp_code = str(ptok.get('depth', '0'))
                seq_y_depth.append(self.depth_vocab.get(dp_code, self.depth_vocab['0']))

            # Create Shifted Windows
            if len(seq_x_idx) >= 2:
                # X is 0 to N-1
                # Y is 1 to N
                x_in = seq_x_idx[:-1]
                
                # Truncate or Pad
                L = min(len(x_in), self.max_seq_len)
                pad = [0] * (self.max_seq_len - L)
                
                self.data_x_seq.append(pad + x_in[:L])
                self.data_context.append(context_vec)
                
                # Targets
                self.data_y_type.append(pad + seq_y_type[1:L+1])
                self.data_y_dir.append(pad + seq_y_dir[1:L+1])
                self.data_y_depth.append(pad + seq_y_depth[1:L+1])
                self.sample_match_ids.append(match_id)

        # Tensors
        self.x_seq = torch.tensor(self.data_x_seq, dtype=torch.long)
        self.context = torch.tensor(self.data_context, dtype=torch.float32)
        self.y_type = torch.tensor(self.data_y_type, dtype=torch.long)
        self.y_dir  = torch.tensor(self.data_y_dir, dtype=torch.long)
        self.y_depth = torch.tensor(self.data_y_depth, dtype=torch.long)
        
        print(f"Data Ready. Samples: {len(self.x_seq)}")

    def __len__(self): return len(self.x_seq)
    def __getitem__(self, idx):
        return {
            'x_seq': self.x_seq[idx],
            'context': self.context[idx],
            'y_type': self.y_type[idx],
            'y_dir': self.y_dir[idx],
            'y_depth': self.y_depth[idx]
        }
    
class HierarchicalTennisDataset(MCPTennisDataset):
    """
    Final Dataset for Hierarchical Training.
    Features:
    1. Rich Decomposed Inputs (Type, Dir, Depth, Players)
    2. Decomposed Targets (Type, Dir, Depth) for Multi-Head Loss
    """
    def __init__(self, *args, **kwargs):
        # 1. Define Vocabs
        self.type_vocab = {k: v for v, k in enumerate(['<pad>', '<unk>', 'serve', 'f', 'b', 'r', 's', 'v', 'z', 'o', 'p', 'u', 'y', 'l', 'm', 'h', 'i', 'j', 'k', 't', 'q', 'special', 'let'])}
        self.dir_vocab  = {k: v for v, k in enumerate(['<pad>', '0', '1', '2', '3'])}
        self.depth_vocab= {k: v for v, k in enumerate(['<pad>', '0', '7', '8', '9'])}
        
        # 2. Init Parent (Loads data, builds unified tokens)
        super().__init__(*args, **kwargs)
        
        # 3. Decompose BOTH Inputs and Targets
        self._decompose_data()

    def _decompose_data(self):
        print("Decomposing Data for Hierarchical Training...")
        
        # Temporary lists
        x_t_list, x_d_list, x_dp_list = [], [], []
        y_t_list, y_d_list, y_dp_list = [], [], []

        # We can process inputs (x_seq) and targets (y_target) using the same logic
        # Access tensors via .tolist()
        x_seq_raw = self.x_seq_tensor.tolist()
        y_seq_raw = self.y_target_tensor.tolist()

        for x_row, y_row in zip(x_seq_raw, y_seq_raw):
            
            # --- Process Inputs ---
            xt, xd, xdp = self._decompose_sequence(x_row)
            x_t_list.append(xt); x_d_list.append(xd); x_dp_list.append(xdp)

            # --- Process Targets ---
            yt, yd, ydp = self._decompose_sequence(y_row)
            y_t_list.append(yt); y_d_list.append(yd); y_dp_list.append(ydp)

        # Convert to Tensors
        self.x_type = torch.tensor(x_t_list, dtype=torch.long)
        self.x_dir  = torch.tensor(x_d_list, dtype=torch.long)
        self.x_depth= torch.tensor(x_dp_list, dtype=torch.long)

        self.y_type = torch.tensor(y_t_list, dtype=torch.long)
        self.y_dir  = torch.tensor(y_d_list, dtype=torch.long)
        self.y_depth= torch.tensor(y_dp_list, dtype=torch.long)

    def _decompose_sequence(self, seq_ids):
        """Helper to convert a list of Unified IDs -> 3 lists of Attribute IDs"""
        seq_t, seq_d, seq_dp = [], [], []
        
        for token_id in seq_ids:
            if token_id == 0: # PAD
                seq_t.append(0); seq_d.append(0); seq_dp.append(0)
                continue
            
            key = self.inv_unified_vocab.get(token_id, '<unk>')
            
            # Parse key (e.g. "f_1_8")
            if key.startswith('Serve'):
                t, d, dp = 'serve', '0', '0'
            elif '_' in key:
                parts = key.split('_')
                t = parts[0]
                d = parts[1] if len(parts) > 1 else '0'
                dp = parts[2] if len(parts) > 2 else '0'
            else:
                t, d, dp = key, '0', '0'

            seq_t.append(self.type_vocab.get(t, 1))
            seq_d.append(self.dir_vocab.get(d, 1))
            seq_dp.append(self.depth_vocab.get(dp, 1))
            
        return seq_t, seq_d, seq_dp

    def __getitem__(self, idx):
        # Parent gives us context, s_id, r_id
        item = super().__getitem__(idx)
        
        # Add Decomposed Inputs
        item['x_type'] = self.x_type[idx]
        item['x_dir']  = self.x_dir[idx]
        item['x_depth']= self.x_depth[idx]
        
        # Add Decomposed Targets
        item['y_type'] = self.y_type[idx]
        item['y_dir']  = self.y_dir[idx]
        item['y_depth']= self.y_depth[idx]
        
        return item
    

class MCPTennisDatasetGPT(Dataset):
    def __init__(self, points_paths_list, matches_path, atp_path, wta_path, max_seq_len=30):
        self.max_seq_len = max_seq_len
        print("Initializing Dataset with CORRECTED Parsing Logic...")
        
        # 1. Load Player Data (Height & Hand)
        self.player_vocab = {'<pad>': 0, '<unk>': 1}
        self.player_stats = {} # {name: {'hand': 'R', 'height': 185}}
        self._build_player_stats(atp_path, wta_path)
        
        # 2. Load Matches
        try:
            self.matches_df = pd.read_csv(matches_path, encoding='ISO-8859-1', engine='python', on_bad_lines='skip')
        except:
            self.matches_df = pd.read_csv(matches_path, encoding='ISO-8859-1', quoting=3)
        self.match_meta = self._process_match_metadata(self.matches_df)
        
        # 3. Load Points
        dfs = []
        for p_path in points_paths_list:
            if not os.path.exists(p_path): continue
            try:
                d = pd.read_csv(p_path, encoding='ISO-8859-1', engine='python', on_bad_lines='skip')
                dfs.append(d)
            except:
                d = pd.read_csv(p_path, encoding='ISO-8859-1', quoting=3)
                dfs.append(d)
        
        if dfs:
            self.df = pd.concat(dfs, ignore_index=True)
            # Sort by Match and Point to track flow
            self.df['Pt'] = pd.to_numeric(self.df['Pt'], errors='coerce').fillna(0)
            self.df = self.df.sort_values(by=['match_id', 'Pt'])
        else:
            self.df = pd.DataFrame()

        # 4. Vocabularies based on Documentation
        self.surface_vocab = {'<pad>': 0, 'Hard': 1, 'Clay': 2, 'Grass': 3, 'Carpet': 4}
        self.hand_vocab = {'<pad>': 0, 'R': 1, 'L': 2}
        
        # The Unified Token Vocabulary
        # We will build this dynamically as we parse to ensure we capture all valid combinations
        # e.g. "Serve_6", "f_1_7", "b_2"
        self.unified_vocab = {'<pad>': 0, '<unk>': 1}
        self.inv_unified_vocab = {0: '<pad>', 1: '<unk>'}
        
        # Data Containers
        self.data_x_seq = []   # Sequence of Unified Token IDs
        self.data_context = [] # Fixed context (Players, Surface, Score)
        self.data_x_s_id = []  # Server ID
        self.data_x_r_id = []  # Receiver ID
        self.data_y_type = [] # Target List
        self.sample_match_ids = []
        
        self.process_data()

    def _build_player_stats(self, atp_path, wta_path):
        for path in [atp_path, wta_path]:
            if not os.path.exists(path): continue
            try:
                df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')
                df['full_name'] = df['name_first'].str.strip() + " " + df['name_last'].str.strip()
                
                for _, row in df.iterrows():
                    name = row['full_name']
                    if name not in self.player_vocab:
                        self.player_vocab[name] = len(self.player_vocab)
                    
                    h = row.get('height')
                    hand = row.get('hand', 'R')
                    
                    height_val = 185 # Default
                    if pd.notna(h) and str(h).isdigit():
                        height_val = float(h)
                        
                    self.player_stats[name] = {'hand': hand, 'height': height_val}
            except: pass

    def _process_match_metadata(self, df):
        meta = {}
        for _, row in df.iterrows():
            m_id = row['match_id']
            # Surface
            raw_surf = str(row.get('Surface', 'Hard'))
            if 'Hard' in raw_surf: surf = 'Hard'
            elif 'Clay' in raw_surf: surf = 'Clay'
            elif 'Grass' in raw_surf: surf = 'Grass'
            else: surf = 'Hard'
            
            p1 = str(row.get('Player 1', ''))
            p2 = str(row.get('Player 2', ''))
            
            # Lookup Hand/Height from our loaded stats
            stats1 = self.player_stats.get(p1, {'hand': 'R', 'height': 185})
            stats2 = self.player_stats.get(p2, {'hand': 'R', 'height': 185})
            
            meta[m_id] = {
                'surface': surf, 
                'p1_name': p1, 'p2_name': p2, 
                'p1_hand': stats1['hand'], 'p2_hand': stats2['hand'],
                'p1_h': stats1['height'], 'p2_h': stats2['height']
            }
        return meta

    def get_pressure_score(self, pts, gm1, gm2):
        # 0=Normal, 1=Tiebreak, 2=BreakPoint
        if not isinstance(pts, str) or '-' not in pts: return 0
        try:
            s_pts, r_pts = pts.split('-')
            # Check Break Point (Receiver has 40 or AD)
            if (r_pts == '40' and s_pts != '40' and s_pts != 'AD') or (r_pts == 'AD'):
                return 2
            # Check Tiebreak
            if int(gm1) == 6 and int(gm2) == 6:
                return 1
        except: pass
        return 0

    def process_data(self):
        print("Parsing Rallies into Unified Tokens...")
        
        # Regex to split shots:
        # 1. Serves: Start with 4, 5, or 6. Can have modifiers (+, -, =).
        # 2. Shots: Start with char (f, b, s, r, v, ...).
        # We split the string by looking for these boundaries.
        
        # Pattern explanation:
        # ([456][0-9+=-]*)  -> Capture Serve (starts with 4,5,6, followed by optional chars)
        # |                 -> OR
        # ([fbsrvolmzup][0-9+=-]*) -> Capture Shot (starts with type char, followed by optional chars)
        split_pattern = re.compile(r'([456][0-9+=-]*|[fbsrvolmzup][0-9+=-]*)')
        
        # Helper to extract digits from a shot string
        # e.g. "s17" -> ['1', '7']
        digit_pattern = re.compile(r'\d')
        
        prev_point_state = (0, 0) # (WinnerWasServer, RallyLen)
        last_match_id = None
        
        for _, row in self.df.iterrows():
            match_id = row['match_id']
            if match_id != last_match_id:
                prev_point_state = (0, 0)
                last_match_id = match_id
            
            rally_str = str(row['2nd']) if pd.notna(row['2nd']) else str(row['1st'])
            if pd.isna(rally_str) or rally_str == 'nan': continue

            # --- CONTEXT ---
            m_meta = self.match_meta.get(match_id, {'surface': 'Hard', 'p1_name':'?', 'p2_name':'?'})
            
            svr = row['Svr'] if 'Svr' in row else 1
            if svr == 2:
                s_name, r_name = m_meta['p2_name'], m_meta['p1_name']
                s_h, r_h = m_meta['p2_h'], m_meta['p1_h']
                s_hand, r_hand = m_meta['p2_hand'], m_meta['p1_hand']
            else:
                s_name, r_name = m_meta['p1_name'], m_meta['p2_name']
                s_h, r_h = m_meta['p1_h'], m_meta['p2_h']
                s_hand, r_hand = m_meta['p1_hand'], m_meta['p2_hand']

            s_id = self.player_vocab.get(s_name, 1)
            r_id = self.player_vocab.get(r_name, 1)
            
            # Context Features
            surf_idx = self.surface_vocab.get(m_meta['surface'], 1)
            sh_idx = self.hand_vocab.get(s_hand, 1)
            rh_idx = self.hand_vocab.get(r_hand, 1)
            sh_norm = (s_h - 180) / 10.0
            rh_norm = (r_h - 180) / 10.0
            pressure = self.get_pressure_score(str(row.get('Pts','')), row.get('Gm1',0), row.get('Gm2',0))
            prev_win, prev_len = prev_point_state
            is_2nd = 1 if pd.notna(row['2nd']) else 0
            
            context_vec = [surf_idx, sh_idx, rh_idx, sh_norm, rh_norm, pressure, prev_win, prev_len, is_2nd, 0]

            # --- RALLY PARSING ---
            r_clean = re.sub(r'[@#n*!+;]', '', rally_str).lstrip('c')
            if not r_clean: continue
            
            # Find all shots
            raw_shots = split_pattern.findall(r_clean)
            if not raw_shots: continue
            
            unified_tokens = []
            
            for i, shot_str in enumerate(raw_shots):
                # 1. PARSE SERVE (First shot, starts with digit)
                if shot_str[0].isdigit():
                    # Format: "6", "5", "4+"
                    direction = shot_str[0] # 4, 5, or 6
                    token = f"Serve_{direction}"
                else:
                    # 2. PARSE SHOT (Starts with char)
                    # Format: "f1", "s17", "b2"
                    typ = shot_str[0] # f, b, s, ...
                    
                    # Extract digits
                    digits = digit_pattern.findall(shot_str)
                    
                    direction = digits[0] if len(digits) > 0 else '0' # 1, 2, 3
                    depth = digits[1] if len(digits) > 1 else '0'     # 7, 8, 9
                    
                    token = f"{typ}_{direction}_{depth}"
                
                # Add to vocab if new
                if token not in self.unified_vocab:
                    self.unified_vocab[token] = len(self.unified_vocab)
                    self.inv_unified_vocab[self.unified_vocab[token]] = token
                
                unified_tokens.append(self.unified_vocab[token])
            
            # Create Sequence
            if len(unified_tokens) >= 2:
                # Update Flow State
                curr_winner = row.get('PtWinner', 0)
                try: curr_winner = int(curr_winner)
                except: curr_winner = 0
                prev_point_state = (1 if curr_winner == svr else 0, len(unified_tokens))
                
                # Input: [Serve, Shot1, Shot2...]
                # Target: [Shot1, Shot2, Shot3...]
                
                seq_x = unified_tokens[:-1]
                seq_y = unified_tokens[1:]
                
                L = min(len(seq_x), self.max_seq_len)
                pad = [0] * (self.max_seq_len - L)
                
                self.data_x_seq.append(pad + seq_x[:L])
                self.data_context.append(context_vec)
                self.data_x_s_id.append(s_id)
                self.data_x_r_id.append(r_id)
                
                # Target is just the Unified ID
                self.data_y_type.append(pad + seq_y[:L]) # Reusing this list name for Y
                self.sample_match_ids.append(match_id)

        print(f"Dataset Built. Unique Unified Shots Found: {len(self.unified_vocab)}")
        
        self.x_seq_tensor = torch.tensor(self.data_x_seq, dtype=torch.long)
        self.context_tensor = torch.tensor(self.data_context, dtype=torch.float32)
        self.x_s_id_tensor = torch.tensor(self.data_x_s_id, dtype=torch.long)
        self.x_r_id_tensor = torch.tensor(self.data_x_r_id, dtype=torch.long)
        self.y_target_tensor = torch.tensor(self.data_y_type, dtype=torch.long)
        
        del self.data_x_seq, self.data_y_type

    def __len__(self): return len(self.sample_match_ids)
    def __getitem__(self, idx):
        return {
            'x_seq': self.x_seq_tensor[idx],
            'context': self.context_tensor[idx],
            'x_s_id': self.x_s_id_tensor[idx],
            'x_r_id': self.x_r_id_tensor[idx],
            'y_target': self.y_target_tensor[idx]
        }
    

class DownsampledDataset(Dataset):
    def __init__(self, points_paths_list, matches_path, atp_path, wta_path, max_seq_len=30):
        self.max_seq_len = max_seq_len
        print("Initializing Dataset with Downsampling Logic...")
        
        # 1. Load Player Data (Height & Hand)
        self.player_vocab = {'<pad>': 0, '<unk>': 1}
        self.player_stats = {} 
        self._build_player_stats(atp_path, wta_path)
        
        # 2. Load Matches
        try:
            self.matches_df = pd.read_csv(matches_path, encoding='ISO-8859-1', engine='python', on_bad_lines='skip')
        except:
            self.matches_df = pd.read_csv(matches_path, encoding='ISO-8859-1', quoting=3)

        # --- NEW STEP: DOWNSAMPLE MATCHES ---
        self._downsample_matches()
        # ------------------------------------

        self.match_meta = self._process_match_metadata(self.matches_df)
        
        # 3. Load Points
        dfs = []
        for p_path in points_paths_list:
            if not os.path.exists(p_path): continue
            try:
                d = pd.read_csv(p_path, encoding='ISO-8859-1', engine='python', on_bad_lines='skip')
                dfs.append(d)
            except:
                d = pd.read_csv(p_path, encoding='ISO-8859-1', quoting=3)
                dfs.append(d)
        
        if dfs:
            self.df = pd.concat(dfs, ignore_index=True)
            
            # --- FILTER POINTS DATAFRAME IMMEDIATELY ---
            # This speeds up processing significantly by removing points from deleted matches
            print(f"Points before filtering: {len(self.df)}")
            self.df = self.df[self.df['match_id'].isin(self.valid_match_ids)]
            print(f"Points after filtering: {len(self.df)}")
            
            # Sort by Match and Point to track flow
            self.df['Pt'] = pd.to_numeric(self.df['Pt'], errors='coerce').fillna(0)
            self.df = self.df.sort_values(by=['match_id', 'Pt'])
        else:
            self.df = pd.DataFrame()

        # 4. Vocabularies
        self.surface_vocab = {'<pad>': 0, 'Hard': 1, 'Clay': 2, 'Grass': 3, 'Carpet': 4}
        self.hand_vocab = {'<pad>': 0, 'R': 1, 'L': 2}
        
        self.unified_vocab = {'<pad>': 0, '<unk>': 1}
        self.inv_unified_vocab = {0: '<pad>', 1: '<unk>'}
        
        # Data Containers
        self.data_x_seq = []   
        self.data_context = [] 
        self.data_x_s_id = []  
        self.data_x_r_id = []  
        self.data_y_type = [] 
        self.sample_match_ids = []
        
        self.process_data()

    def _downsample_matches(self):
        print("--- Downsampling Matches ---")
        
        # 1. Define Targets
        target_counts = {
            'Hard': 2305, # ~66% of 3480
            'Clay': 781,
            'Grass': 394
        }
        total_target = sum(target_counts.values())
        
        # 2. Group Match IDs by Surface
        surface_groups = {'Hard': [], 'Clay': [], 'Grass': []}
        
        for _, row in self.matches_df.iterrows():
            m_id = row['match_id']
            raw_surf = str(row.get('Surface', 'Hard'))
            
            # Normalize Surface (Carpet -> Hard)
            if 'Carpet' in raw_surf or 'Hard' in raw_surf:
                s_key = 'Hard'
            elif 'Clay' in raw_surf:
                s_key = 'Clay'
            elif 'Grass' in raw_surf:
                s_key = 'Grass'
            else:
                s_key = 'Hard' # Default
            
            surface_groups[s_key].append(m_id)
            
        # 3. Randomly Sample
        selected_ids = []
        
        for surf, targets in target_counts.items():
            available = surface_groups[surf]
            n_avail = len(available)
            n_needed = targets
            
            print(f"Surface {surf}: Found {n_avail}, Need {n_needed}")
            
            if n_avail > n_needed:
                # Randomly pick n_needed
                selected = random.sample(available, n_needed)
            else:
                # Take all available if we don't have enough
                selected = available
                
            selected_ids.extend(selected)
            
        self.valid_match_ids = set(selected_ids)
        print(f"Total Matches kept: {len(self.valid_match_ids)} (Target: {total_target})")
        
        # 4. Filter the Matches DataFrame itself to save memory
        self.matches_df = self.matches_df[self.matches_df['match_id'].isin(self.valid_match_ids)]

    def _build_player_stats(self, atp_path, wta_path):
        for path in [atp_path, wta_path]:
            if not os.path.exists(path): continue
            try:
                df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')
                df['full_name'] = df['name_first'].str.strip() + " " + df['name_last'].str.strip()
                
                for _, row in df.iterrows():
                    name = row['full_name']
                    if name not in self.player_vocab:
                        self.player_vocab[name] = len(self.player_vocab)
                    
                    h = row.get('height')
                    hand = row.get('hand', 'R')
                    
                    height_val = 185 # Default
                    if pd.notna(h) and str(h).isdigit():
                        height_val = float(h)
                        
                    self.player_stats[name] = {'hand': hand, 'height': height_val}
            except: pass

    def _process_match_metadata(self, df):
        meta = {}
        for _, row in df.iterrows():
            m_id = row['match_id']
            # Surface
            raw_surf = str(row.get('Surface', 'Hard'))
            if 'Hard' in raw_surf: surf = 'Hard'
            elif 'Clay' in raw_surf: surf = 'Clay'
            elif 'Grass' in raw_surf: surf = 'Grass'
            else: surf = 'Hard'
            
            p1 = str(row.get('Player 1', ''))
            p2 = str(row.get('Player 2', ''))
            
            # Lookup Hand/Height from our loaded stats
            stats1 = self.player_stats.get(p1, {'hand': 'R', 'height': 185})
            stats2 = self.player_stats.get(p2, {'hand': 'R', 'height': 185})
            
            meta[m_id] = {
                'surface': surf, 
                'p1_name': p1, 'p2_name': p2, 
                'p1_hand': stats1['hand'], 'p2_hand': stats2['hand'],
                'p1_h': stats1['height'], 'p2_h': stats2['height']
            }
        return meta

    def get_pressure_score(self, pts, gm1, gm2):
        # 0=Normal, 1=Tiebreak, 2=BreakPoint
        if not isinstance(pts, str) or '-' not in pts: return 0
        try:
            s_pts, r_pts = pts.split('-')
            # Check Break Point (Receiver has 40 or AD)
            if (r_pts == '40' and s_pts != '40' and s_pts != 'AD') or (r_pts == 'AD'):
                return 2
            # Check Tiebreak
            if int(gm1) == 6 and int(gm2) == 6:
                return 1
        except: pass
        return 0

    def process_data(self):
        print("Parsing Rallies into Unified Tokens...")
        
        split_pattern = re.compile(r'([456][0-9+=-]*|[fbsrvolmzup][0-9+=-]*)')
        digit_pattern = re.compile(r'\d')
        
        prev_point_state = (0, 0) # (WinnerWasServer, RallyLen)
        last_match_id = None
        
        for _, row in self.df.iterrows():
            match_id = row['match_id']
            
            # Skip if this match wasn't in our downsampled list
            if match_id not in self.valid_match_ids:
                continue

            if match_id != last_match_id:
                prev_point_state = (0, 0)
                last_match_id = match_id
            
            rally_str = str(row['2nd']) if pd.notna(row['2nd']) else str(row['1st'])
            if pd.isna(rally_str) or rally_str == 'nan': continue

            # --- CONTEXT ---
            m_meta = self.match_meta.get(match_id, {'surface': 'Hard', 'p1_name':'?', 'p2_name':'?'})
            
            svr = row['Svr'] if 'Svr' in row else 1
            if svr == 2:
                s_name, r_name = m_meta['p2_name'], m_meta['p1_name']
                s_h, r_h = m_meta['p2_h'], m_meta['p1_h']
                s_hand, r_hand = m_meta['p2_hand'], m_meta['p1_hand']
            else:
                s_name, r_name = m_meta['p1_name'], m_meta['p2_name']
                s_h, r_h = m_meta['p1_h'], m_meta['p2_h']
                s_hand, r_hand = m_meta['p1_hand'], m_meta['p2_hand']

            s_id = self.player_vocab.get(s_name, 1)
            r_id = self.player_vocab.get(r_name, 1)
            
            # Context Features
            surf_idx = self.surface_vocab.get(m_meta['surface'], 1)
            sh_idx = self.hand_vocab.get(s_hand, 1)
            rh_idx = self.hand_vocab.get(r_hand, 1)
            sh_norm = (s_h - 180) / 10.0
            rh_norm = (r_h - 180) / 10.0
            pressure = self.get_pressure_score(str(row.get('Pts','')), row.get('Gm1',0), row.get('Gm2',0))
            prev_win, prev_len = prev_point_state
            is_2nd = 1 if pd.notna(row['2nd']) else 0
            
            context_vec = [surf_idx, sh_idx, rh_idx, sh_norm, rh_norm, pressure, prev_win, prev_len, is_2nd, 0]

            # --- RALLY PARSING ---
            r_clean = re.sub(r'[@#n*!+;]', '', rally_str).lstrip('c')
            if not r_clean: continue
            
            # Find all shots
            raw_shots = split_pattern.findall(r_clean)
            if not raw_shots: continue
            
            unified_tokens = []
            
            for i, shot_str in enumerate(raw_shots):
                # 1. PARSE SERVE (First shot, starts with digit)
                if shot_str[0].isdigit():
                    # Format: "6", "5", "4+"
                    direction = shot_str[0] # 4, 5, or 6
                    token = f"Serve_{direction}"
                else:
                    # 2. PARSE SHOT (Starts with char)
                    # Format: "f1", "s17", "b2"
                    typ = shot_str[0] # f, b, s, ...
                    
                    # Extract digits
                    digits = digit_pattern.findall(shot_str)
                    
                    direction = digits[0] if len(digits) > 0 else '0' # 1, 2, 3
                    depth = digits[1] if len(digits) > 1 else '0'      # 7, 8, 9
                    
                    token = f"{typ}_{direction}_{depth}"
                
                # Add to vocab if new
                if token not in self.unified_vocab:
                    self.unified_vocab[token] = len(self.unified_vocab)
                    self.inv_unified_vocab[self.unified_vocab[token]] = token
                
                unified_tokens.append(self.unified_vocab[token])
            
            # Create Sequence
            if len(unified_tokens) >= 2:
                # Update Flow State
                curr_winner = row.get('PtWinner', 0)
                try: curr_winner = int(curr_winner)
                except: curr_winner = 0
                prev_point_state = (1 if curr_winner == svr else 0, len(unified_tokens))
                
                seq_x = unified_tokens[:-1]
                seq_y = unified_tokens[1:]
                
                L = min(len(seq_x), self.max_seq_len)
                pad = [0] * (self.max_seq_len - L)
                
                self.data_x_seq.append(pad + seq_x[:L])
                self.data_context.append(context_vec)
                self.data_x_s_id.append(s_id)
                self.data_x_r_id.append(r_id)
                
                # Target is just the Unified ID
                self.data_y_type.append(pad + seq_y[:L]) # Reusing this list name for Y
                self.sample_match_ids.append(match_id)

        print(f"Dataset Built. Unique Unified Shots Found: {len(self.unified_vocab)}")
        
        self.x_seq_tensor = torch.tensor(self.data_x_seq, dtype=torch.long)
        self.context_tensor = torch.tensor(self.data_context, dtype=torch.float32)
        self.x_s_id_tensor = torch.tensor(self.data_x_s_id, dtype=torch.long)
        self.x_r_id_tensor = torch.tensor(self.data_x_r_id, dtype=torch.long)
        self.y_target_tensor = torch.tensor(self.data_y_type, dtype=torch.long)
        
        del self.data_x_seq, self.data_y_type

    def __len__(self): return len(self.sample_match_ids)
    def __getitem__(self, idx):
        return {
            'x_seq': self.x_seq_tensor[idx],
            'context': self.context_tensor[idx],
            'x_s_id': self.x_s_id_tensor[idx],
            'x_r_id': self.x_r_id_tensor[idx],
            'y_target': self.y_target_tensor[idx]
        }
    
class DownsampledHierarchical(DownsampledDataset):
    """
    Final Dataset for Hierarchical Training.
    Inherits from DownsampledDataset (which now includes Downsampling).
    """
    def __init__(self, *args, **kwargs):
        # 1. Define Vocabs
        self.type_vocab = {k: v for v, k in enumerate(['<pad>', '<unk>', 'serve', 'f', 'b', 'r', 's', 'v', 'z', 'o', 'p', 'u', 'y', 'l', 'm', 'h', 'i', 'j', 'k', 't', 'q', 'special', 'let'])}
        self.dir_vocab  = {k: v for v, k in enumerate(['<pad>', '0', '1', '2', '3'])}
        self.depth_vocab= {k: v for v, k in enumerate(['<pad>', '0', '7', '8', '9'])}
        
        # 2. Init Parent (Loads data, downsamples, builds unified tokens)
        super().__init__(*args, **kwargs)
        
        # 3. Decompose BOTH Inputs and Targets
        self._decompose_data()

    def _decompose_data(self):
        print("Decomposing Data for Hierarchical Training...")
        x_t_list, x_d_list, x_dp_list = [], [], []
        y_t_list, y_d_list, y_dp_list = [], [], []

        x_seq_raw = self.x_seq_tensor.tolist()
        y_seq_raw = self.y_target_tensor.tolist()

        for x_row, y_row in zip(x_seq_raw, y_seq_raw):
            xt, xd, xdp = self._decompose_sequence(x_row)
            x_t_list.append(xt); x_d_list.append(xd); x_dp_list.append(xdp)

            yt, yd, ydp = self._decompose_sequence(y_row)
            y_t_list.append(yt); y_d_list.append(yd); y_dp_list.append(ydp)

        self.x_type = torch.tensor(x_t_list, dtype=torch.long)
        self.x_dir  = torch.tensor(x_d_list, dtype=torch.long)
        self.x_depth= torch.tensor(x_dp_list, dtype=torch.long)

        self.y_type = torch.tensor(y_t_list, dtype=torch.long)
        self.y_dir  = torch.tensor(y_d_list, dtype=torch.long)
        self.y_depth= torch.tensor(y_dp_list, dtype=torch.long)

    def _decompose_sequence(self, seq_ids):
        seq_t, seq_d, seq_dp = [], [], []
        for token_id in seq_ids:
            if token_id == 0: 
                seq_t.append(0); seq_d.append(0); seq_dp.append(0)
                continue
            
            key = self.inv_unified_vocab.get(token_id, '<unk>')
            if key.startswith('Serve'):
                t, d, dp = 'serve', '0', '0'
            elif '_' in key:
                parts = key.split('_')
                t = parts[0]
                d = parts[1] if len(parts) > 1 else '0'
                dp = parts[2] if len(parts) > 2 else '0'
            else:
                t, d, dp = key, '0', '0'

            seq_t.append(self.type_vocab.get(t, 1))
            seq_d.append(self.dir_vocab.get(d, 1))
            seq_dp.append(self.depth_vocab.get(dp, 1))
            
        return seq_t, seq_d, seq_dp

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item['x_type'] = self.x_type[idx]
        item['x_dir']  = self.x_dir[idx]
        item['x_depth']= self.x_depth[idx]
        item['y_type'] = self.y_type[idx]
        item['y_dir']  = self.y_dir[idx]
        item['y_depth']= self.y_depth[idx]
        return item