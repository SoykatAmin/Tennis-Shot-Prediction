"""
Tennis Shot Prediction Dataset

This module contains the MCPTennisDataset class for loading and preprocessing
tennis match data for shot prediction tasks.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import re
import os
from typing import Dict, List, Optional, Tuple


class MCPTennisDataset(Dataset):
    """Dataset for tennis shot prediction with player embeddings and data augmentation."""
    
    def __init__(
        self, 
        points_path: str, 
        matches_path: str, 
        atp_path: str, 
        wta_path: str, 
        max_seq_len: int = 30
    ):
        """
        Initialize the tennis dataset.
        
        Args:
            points_path: Path to tennis points CSV file
            matches_path: Path to tennis matches CSV file
            atp_path: Path to ATP players CSV file
            wta_path: Path to WTA players CSV file
            max_seq_len: Maximum sequence length for padding
        """
        self.max_seq_len = max_seq_len
        
        print("Initializing Dataset (Full Version with Player IDs)...")
        
        # Load bio-data (handedness + player vocabulary)
        try:
            self.name_to_hand = self._load_handedness(atp_path, wta_path)
            self.player_vocab = {'<pad>': 0, '<unk>': 1}
            self._build_player_vocab(atp_path, wta_path)
            print(f"Bio-data loaded. Found {len(self.player_vocab)} unique players.")
        except Exception as e:
            print(f"⚠️ Warning: Could not load player bio-data ({e}). Defaulting.")        
            self.name_to_hand = {}
            
        # Load matches
        print(f"Loading matches: {matches_path}...")
        try:
            self.matches_df = pd.read_csv(matches_path, encoding='ISO-8859-1', engine='python', on_bad_lines='skip')
        except:
            self.matches_df = pd.read_csv(matches_path, encoding='ISO-8859-1', quoting=3)
            
        self.match_meta = self._process_match_metadata(self.matches_df)
        
        # Load points
        print(f"Loading points: {points_path}...")
        try:
            self.df = pd.read_csv(points_path, encoding='ISO-8859-1', engine='python', on_bad_lines='skip')
        except:
            self.df = pd.read_csv(points_path, encoding='ISO-8859-1', quoting=3)
        
        # Initialize vocabularies
        self._initialize_vocabularies()
        
        # Process data
        self.samples = []
        self.sample_match_ids = []
        self.process_data()

    def _initialize_vocabularies(self):
        """Initialize all vocabulary mappings."""
        self.shot_vocab = {
            '<pad>': 0, 'f': 1, 'b': 2, 'r': 3, 'v': 4, 
            'o': 5, 's': 6, 'u': 7, 'l': 8, 'm': 9, 'z': 10
        }
        self.zone_vocab = {
            '<pad>': 0, '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, 
            '5': 5, '6': 6, '7': 7, '8': 8, '9': 9
        }
        self.surface_vocab = {'<pad>': 0, 'Hard': 1, 'Clay': 2, 'Grass': 3, 'Carpet': 4}
        self.hand_vocab = {'<pad>': 0, 'R': 1, 'L': 2}
        self.score_map = {'0': 0, '15': 1, '30': 2, '40': 3, 'AD': 4}

    def _load_handedness(self, atp_path: str, wta_path: str) -> Dict[str, str]:
        """Load player handedness data from ATP/WTA files."""
        hand_map = {}
        for path in [atp_path, wta_path]:
            if not os.path.exists(path):
                continue
            try:
                df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')
                # Create "First Last" key
                df['full_name'] = df['name_first'].str.strip() + " " + df['name_last'].str.strip()
                subset = df[['full_name', 'hand']].dropna()
                for _, row in subset.iterrows():
                    if row['hand'] != 'U': 
                        hand_map[row['full_name']] = row['hand']
                    else: 
                        hand_map[row['full_name']] = 'R'
            except:
                pass
        return hand_map

    def _build_player_vocab(self, atp_path: str, wta_path: str):
        """Build vocabulary of player names to integer IDs."""
        for path in [atp_path, wta_path]:
            if not os.path.exists(path):
                continue
            try:
                df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')
                df['full_name'] = df['name_first'].str.strip() + " " + df['name_last'].str.strip()
                for name in df['full_name'].dropna().unique():
                    if name not in self.player_vocab:
                        self.player_vocab[name] = len(self.player_vocab)
            except:
                pass

    def _process_match_metadata(self, df: pd.DataFrame) -> Dict:
        """Process match metadata including surface and player information."""
        meta = {}
        for _, row in df.iterrows():
            m_id = row['match_id']
            
            # Robust surface check
            if 'Surface' in row:
                surf = str(row['Surface'])
            else:
                surf = str(row.get('surface', 'Hard'))
            
            if 'Hard' in surf:
                surf = 'Hard'
            elif 'Clay' in surf:
                surf = 'Clay'
            elif 'Grass' in surf:
                surf = 'Grass'
            else:
                surf = 'Hard'
            
            # Robust player check
            if 'Player1' in row:
                p1 = str(row['Player1']).strip()
            else:
                p1 = str(row.get('player1', '')).strip()
            
            if 'Player2' in row:
                p2 = str(row['Player2']).strip()
            else:
                p2 = str(row.get('player2', '')).strip()
            
            meta[m_id] = {
                'surface': surf,
                'p1_name': p1,
                'p2_name': p2,
                'p1_hand': self.name_to_hand.get(p1, 'R'),
                'p2_hand': self.name_to_hand.get(p2, 'R')
            }
        return meta

    def process_data(self):
        """Process rally data with data augmentation (left/right flip)."""
        print(f"Parsing rallies with Data Augmentation (Left/Right Flip)...")
        shot_pattern = re.compile(r'([0-9]*)([a-zA-Z])([0-9]*)')
        
        # Define flip mappings (Left <-> Right)
        # 1<->3, 4<->6, 7<->9. Center zones (2,5,8) and 0 stay same.
        flip_zone_map = {1: 3, 3: 1, 4: 6, 6: 4, 7: 9, 9: 7}
        
        for _, row in self.df.iterrows():
            match_id = row['match_id']
            
            rally_str = str(row['2nd']) if pd.notna(row['2nd']) else str(row['1st'])
            if pd.isna(rally_str) or rally_str == 'nan':
                continue

            # Get context
            m_meta = self.match_meta.get(
                match_id, 
                {'surface': 'Hard', 'p1_name':'?', 'p2_name':'?', 'p1_hand': 'R', 'p2_hand': 'R'}
            )
            
            # Surface
            surf_idx = self.surface_vocab.get(m_meta['surface'], 1)
            
            # Score
            sc_s, sc_r = 0, 0
            if 'Pts' in row and isinstance(row['Pts'], str):
                try:
                    parts = row['Pts'].split('-')
                    if len(parts) == 2:
                        sc_s = self.score_map.get(parts[0], 0)
                        sc_r = self.score_map.get(parts[1], 0)
                except:
                    pass
            
            # Determine server/receiver identity
            svr = row['Svr'] if 'Svr' in row else 1
            if svr == 2:
                s_hand = m_meta['p2_hand']
                r_hand = m_meta['p1_hand']
                s_name = m_meta['p2_name']
                r_name = m_meta['p1_name']
            else:
                s_hand = m_meta['p1_hand']
                r_hand = m_meta['p2_hand']
                s_name = m_meta['p1_name']
                r_name = m_meta['p2_name']
                
            sh_idx = self.hand_vocab.get(s_hand, 1)
            rh_idx = self.hand_vocab.get(r_hand, 1)
            
            s_id = self.player_vocab.get(s_name, 1)
            r_id = self.player_vocab.get(r_name, 1)

            is_2nd = 1 if pd.notna(row['2nd']) else 0
            
            # Context vector: [Surface, ScoreS, ScoreR, Is2nd, HandS, HandR]
            context_vec = [surf_idx, sc_s, sc_r, is_2nd, sh_idx, rh_idx]

            # Process rally
            r_clean = re.sub(r'[@#n\\*\\!\\+;]', '', rally_str)
            matches = shot_pattern.findall(r_clean)
            if len(matches) < 2:
                continue
            
            seq_arr, seq_typ, seq_tgt = [], [], []
            last_tgt = 0
            
            for m in matches:
                ac, sc, tc = m
                arr = self.zone_vocab.get(ac, 0) if ac else last_tgt
                typ = self.shot_vocab.get(sc.lower(), 0)
                tgt = self.zone_vocab.get(tc, 0) if tc else 0
                last_tgt = tgt
                
                if typ != 0:
                    seq_arr.append(arr)
                    seq_typ.append(typ)
                    seq_tgt.append(tgt)
            
            if len(seq_tgt) > 1:
                L = min(len(seq_arr), self.max_seq_len)
                pad = [0] * (self.max_seq_len - L)
                
                # Original sample
                self.samples.append({
                    'x_zone': torch.tensor(pad + seq_arr[:L], dtype=torch.long),
                    'x_type': torch.tensor(pad + seq_typ[:L], dtype=torch.long),
                    'x_s_id': torch.tensor(s_id, dtype=torch.long),
                    'x_r_id': torch.tensor(r_id, dtype=torch.long),
                    'context': torch.tensor(context_vec, dtype=torch.float32),
                    'y_target': torch.tensor(pad + seq_tgt[:L], dtype=torch.long)
                })
                self.sample_match_ids.append(match_id)
                
                # Augmented sample (mirrored)
                seq_arr_aug = [flip_zone_map.get(z, z) for z in seq_arr[:L]]
                seq_tgt_aug = [flip_zone_map.get(z, z) for z in seq_tgt[:L]]
                
                # Flip handedness (R->L, L->R)
                c_aug = context_vec.copy()
                c_aug[4] = 2 if c_aug[4] == 1 else (1 if c_aug[4] == 2 else c_aug[4])
                c_aug[5] = 2 if c_aug[5] == 1 else (1 if c_aug[5] == 2 else c_aug[5])

                self.samples.append({
                    'x_zone': torch.tensor(pad + seq_arr_aug, dtype=torch.long),
                    'x_type': torch.tensor(pad + seq_typ[:L], dtype=torch.long),
                    'x_s_id': torch.tensor(s_id, dtype=torch.long),
                    'x_r_id': torch.tensor(r_id, dtype=torch.long),
                    'context': torch.tensor(c_aug, dtype=torch.float32),
                    'y_target': torch.tensor(pad + seq_tgt_aug, dtype=torch.long)
                })
                self.sample_match_ids.append(match_id)
                
        print(f"Dataset ready: {len(self.samples)} samples (including augmented).")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        return self.samples[idx]


class SimplifiedTennisDataset(Dataset):
    """Simplified version of tennis dataset without player embeddings."""
    
    def __init__(
        self, 
        points_path: str, 
        matches_path: str, 
        atp_path: str, 
        wta_path: str, 
        max_seq_len: int = 30
    ):
        """Initialize simplified dataset."""
        self.max_seq_len = max_seq_len
        print("Initializing Simplified Dataset...")
        
        self.name_to_hand = self._load_handedness(atp_path, wta_path)
        
        try:
            self.matches_df = pd.read_csv(matches_path, encoding='ISO-8859-1', engine='python', on_bad_lines='skip')
        except:
            self.matches_df = pd.read_csv(matches_path, encoding='ISO-8859-1', quoting=3)
            
        self.match_meta = self._process_match_metadata(self.matches_df)
        
        try:
            self.df = pd.read_csv(points_path, encoding='ISO-8859-1', engine='python', on_bad_lines='skip')
        except:
            self.df = pd.read_csv(points_path, encoding='ISO-8859-1', quoting=3)
        
        self._initialize_vocabularies()
        
        self.samples = []
        self.sample_match_ids = []
        self.process_data()

    def _load_handedness(self, atp_path: str, wta_path: str) -> Dict[str, str]:
        """Load player handedness data."""
        hand_map = {}
        for path in [atp_path, wta_path]:
            if not os.path.exists(path):
                continue
            try:
                df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')
                df['full_name'] = df['name_first'].str.strip() + " " + df['name_last'].str.strip()
                subset = df[['full_name', 'hand']].dropna()
                for _, row in subset.iterrows():
                    hand_map[row['full_name']] = row['hand']
            except:
                pass
        return hand_map

    def _initialize_vocabularies(self):
        """Initialize vocabulary mappings."""
        self.shot_vocab = {
            '<pad>': 0, 'f': 1, 'b': 2, 'r': 3, 'v': 4, 
            'o': 5, 's': 6, 'u': 7, 'l': 8, 'm': 9, 'z': 10
        }
        self.zone_vocab = {
            '<pad>': 0, '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, 
            '5': 5, '6': 6, '7': 7, '8': 8, '9': 9
        }
        self.surface_vocab = {'<pad>': 0, 'Hard': 1, 'Clay': 2, 'Grass': 3, 'Carpet': 4}
        self.hand_vocab = {'<pad>': 0, 'R': 1, 'L': 2}
        self.score_map = {'0': 0, '15': 1, '30': 2, '40': 3, 'AD': 4}

    def _process_match_metadata(self, df: pd.DataFrame) -> Dict:
        """Process match metadata."""
        meta = {}
        for _, row in df.iterrows():
            m_id = row['match_id']
            surf = str(row.get('Surface', row.get('surface', 'Hard')))
            
            if 'Hard' in surf:
                surf = 'Hard'
            elif 'Clay' in surf:
                surf = 'Clay'
            elif 'Grass' in surf:
                surf = 'Grass'
            else:
                surf = 'Hard'
                
            p1 = str(row.get('Player1', row.get('player1', '')))
            p2 = str(row.get('Player2', row.get('player2', '')))
            
            meta[m_id] = {
                'surface': surf, 
                'p1_hand': self.name_to_hand.get(p1, 'R'), 
                'p2_hand': self.name_to_hand.get(p2, 'R')
            }
        return meta

    def process_data(self):
        """Process rally data."""
        print(f"Parsing rallies...")
        shot_pattern = re.compile(r'([0-9]*)([a-zA-Z])([0-9]*)')
        
        for _, row in self.df.iterrows():
            match_id = row['match_id']
            rally_str = str(row['2nd']) if pd.notna(row['2nd']) else str(row['1st'])
            
            if pd.isna(rally_str) or rally_str == 'nan':
                continue

            m_meta = self.match_meta.get(match_id, {'surface': 'Hard', 'p1_hand': 'R', 'p2_hand': 'R'})
            surf_idx = self.surface_vocab.get(m_meta['surface'], 1)
            
            sc_s, sc_r = 0, 0
            if 'Pts' in row and isinstance(row['Pts'], str):
                try:
                    parts = row['Pts'].split('-')
                    if len(parts) == 2:
                        sc_s = self.score_map.get(parts[0], 0)
                        sc_r = self.score_map.get(parts[1], 0)
                except:
                    pass
            
            svr = row['Svr'] if 'Svr' in row else 1
            if svr == 2:
                s_hand, r_hand = m_meta['p2_hand'], m_meta['p1_hand']
            else:
                s_hand, r_hand = m_meta['p1_hand'], m_meta['p2_hand']
            
            sh_idx = self.hand_vocab.get(s_hand, 1)
            rh_idx = self.hand_vocab.get(r_hand, 1)
            is_2nd = 1 if pd.notna(row['2nd']) else 0
            
            context_vec = [surf_idx, sc_s, sc_r, is_2nd, sh_idx, rh_idx]

            r_clean = re.sub(r'[@#n\*\!\+;]', '', rally_str)
            matches = shot_pattern.findall(r_clean)
            if len(matches) < 2:
                continue
            
            seq_arr, seq_typ, seq_tgt = [], [], []
            last_tgt = 0
            
            for m in matches:
                ac, sc, tc = m
                arr = self.zone_vocab.get(ac, 0) if ac else last_tgt
                typ = self.shot_vocab.get(sc.lower(), 0)
                tgt = self.zone_vocab.get(tc, 0) if tc else 0
                last_tgt = tgt
                
                if typ != 0:
                    seq_arr.append(arr)
                    seq_typ.append(typ)
                    seq_tgt.append(tgt)
            
            if len(seq_tgt) > 1:
                L = min(len(seq_arr), self.max_seq_len)
                pad = [0] * (self.max_seq_len - L)
                
                self.samples.append({
                    'x_zone': torch.tensor(pad + seq_arr[:L], dtype=torch.long),
                    'x_type': torch.tensor(pad + seq_typ[:L], dtype=torch.long),
                    'context': torch.tensor(context_vec, dtype=torch.float32),
                    'y_target': torch.tensor(pad + seq_tgt[:L], dtype=torch.long)
                })
                self.sample_match_ids.append(match_id)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]