"""
Prediction/inference script for tennis shot prediction.

This script provides functionality to:
- Load a trained model
- Make predictions on new rally sequences
- Demonstrate model capabilities with examples
"""

import torch
import numpy as np
import re
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Local imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data import MCPTennisDataset
from src.models import create_model
from src.utils import load_checkpoint


class TennisShotPredictor:
    """Tennis shot prediction interface."""
    
    def __init__(
        self,
        model_path: str,
        config_path: str,
        dataset_paths: Dict[str, str],
        device: str = 'auto'
    ):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to training configuration
            dataset_paths: Dictionary with paths to dataset files
            device: Device to use for inference
        """
        # Setup device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize dataset to get vocabularies
        self.dataset = MCPTennisDataset(
            points_path=dataset_paths['points'],
            matches_path=dataset_paths['matches'],
            atp_path=dataset_paths['atp'],
            wta_path=dataset_paths['wta'],
            max_seq_len=self.config.get('seq_len', 30)
        )
        
        # Initialize model
        model_type = self.config.get('model_type', 'player_aware')
        
        model_kwargs = {
            'context_dim': 6,
            'embed_dim': self.config.get('embed_dim', 64),
            'n_head': self.config.get('n_head', 4),
            'n_cycles': self.config.get('n_cycles', 3),
            'seq_len': self.config.get('seq_len', 30),
            'dropout': 0.0  # No dropout during inference
        }
        
        if model_type == 'player_aware':
            model_kwargs['num_players'] = len(self.dataset.player_vocab)
        
        self.model = create_model(
            model_type=model_type,
            zone_vocab_size=len(self.dataset.zone_vocab),
            type_vocab_size=len(self.dataset.shot_vocab),
            **model_kwargs
        ).to(self.device)
        
        # Load trained weights
        load_checkpoint(model_path, self.model)
        self.model.eval()
        
        self.model_type = model_type
        
        # Create reverse vocabularies for decoding
        self.idx_to_zone = {v: k for k, v in self.dataset.zone_vocab.items()}
        self.idx_to_shot = {v: k for k, v in self.dataset.shot_vocab.items()}
        
        print(f"Predictor initialized with {model_type} model")
    
    def parse_rally(self, rally_string: str) -> Tuple[List[int], List[int], List[int]]:
        """
        Parse rally string into sequences.
        
        Args:
            rally_string: String representation of rally (e.g., "4s 1f 3b")
            
        Returns:
            Tuple of (arrival_zones, shot_types, target_zones)
        """
        shot_pattern = re.compile(r'([0-9]*)([a-zA-Z])([0-9]*)')
        
        # Clean the rally string
        rally_clean = re.sub(r'[@#n\\*\\!\\+;]', '', rally_string)
        matches = shot_pattern.findall(rally_clean)
        
        if not matches:
            raise ValueError(f"No valid shots found in rally: {rally_string}")
        
        seq_arr, seq_typ, seq_tgt = [], [], []
        last_tgt = 0
        
        for match in matches:
            arrival_zone, shot_type, target_zone = match
            
            # Process arrival zone
            arr = self.dataset.zone_vocab.get(arrival_zone, 0) if arrival_zone else last_tgt
            
            # Process shot type
            typ = self.dataset.shot_vocab.get(shot_type.lower(), 0)
            
            # Process target zone
            tgt = self.dataset.zone_vocab.get(target_zone, 0) if target_zone else 0
            last_tgt = tgt
            
            if typ != 0:  # Valid shot type
                seq_arr.append(arr)
                seq_typ.append(typ)
                seq_tgt.append(tgt)
        
        return seq_arr, seq_typ, seq_tgt
    
    def create_context_vector(
        self,
        surface: str = 'Hard',
        score: str = '0-0',
        is_second_serve: bool = False,
        server_hand: str = 'R',
        receiver_hand: str = 'R'
    ) -> List[float]:
        """
        Create context vector from match conditions.
        
        Args:
            surface: Court surface ('Hard', 'Clay', 'Grass')
            score: Current score (e.g., '30-15', '40-40')
            is_second_serve: Whether it's a second serve
            server_hand: Server handedness ('R' or 'L')
            receiver_hand: Receiver handedness ('R' or 'L')
            
        Returns:
            Context vector
        """
        # Surface
        surf_idx = self.dataset.surface_vocab.get(surface, 1)
        
        # Score
        sc_s, sc_r = 0, 0
        if '-' in score:
            try:
                parts = score.split('-')
                if len(parts) == 2:
                    sc_s = self.dataset.score_map.get(parts[0], 0)
                    sc_r = self.dataset.score_map.get(parts[1], 0)
            except:
                pass
        
        # Second serve flag
        is_2nd = 1 if is_second_serve else 0
        
        # Handedness
        sh_idx = self.dataset.hand_vocab.get(server_hand, 1)
        rh_idx = self.dataset.hand_vocab.get(receiver_hand, 1)
        
        return [surf_idx, sc_s, sc_r, is_2nd, sh_idx, rh_idx]
    
    def predict_next_shot(
        self,
        rally_string: str,
        surface: str = 'Hard',
        score: str = '0-0',
        is_second_serve: bool = False,
        server_hand: str = 'R',
        receiver_hand: str = 'R',
        server_name: str = 'Unknown',
        receiver_name: str = 'Unknown',
        top_k: int = 3
    ) -> Dict:
        """
        Predict the next shot in a rally.
        
        Args:
            rally_string: Current rally sequence
            surface: Court surface
            score: Current score
            is_second_serve: Whether it's a second serve
            server_hand: Server handedness
            receiver_hand: Receiver handedness
            server_name: Server name (for player-aware model)
            receiver_name: Receiver name (for player-aware model)
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with predictions and probabilities
        """
        # Parse rally
        try:
            seq_arr, seq_typ, seq_tgt = self.parse_rally(rally_string)
        except ValueError as e:
            return {'error': str(e)}
        
        if len(seq_arr) == 0:
            return {'error': 'No valid shots found'}
        
        # Create context
        context_vec = self.create_context_vector(
            surface=surface,
            score=score,
            is_second_serve=is_second_serve,
            server_hand=server_hand,
            receiver_hand=receiver_hand
        )
        
        # Prepare sequences
        seq_len = self.config.get('seq_len', 30)
        L = min(len(seq_arr), seq_len)
        pad = [0] * (seq_len - L)
        
        # Create tensors
        x_zone = torch.tensor(pad + seq_arr[:L], dtype=torch.long).unsqueeze(0).to(self.device)
        x_type = torch.tensor(pad + seq_typ[:L], dtype=torch.long).unsqueeze(0).to(self.device)
        x_context = torch.tensor(context_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Handle player embeddings for player-aware model
        if self.model_type == 'player_aware':
            s_id = self.dataset.player_vocab.get(server_name, 1)
            r_id = self.dataset.player_vocab.get(receiver_name, 1)
            x_s = torch.tensor(s_id, dtype=torch.long).unsqueeze(0).to(self.device)
            x_r = torch.tensor(r_id, dtype=torch.long).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            if self.model_type == 'player_aware':
                logits = self.model(x_zone, x_type, x_context, x_s, x_r)
            else:
                logits = self.model(x_zone, x_type, x_context)
            
            # Get prediction for the last valid position
            last_pos = seq_len - 1
            while last_pos >= 0 and x_zone[0, last_pos].item() == 0:
                last_pos -= 1
            
            if last_pos < 0:
                return {'error': 'No valid position found'}
            
            # Get probabilities
            probs = torch.softmax(logits[0, last_pos], dim=-1)
            top_probs, top_idxs = torch.topk(probs, top_k)
            
            # Format predictions
            predictions = []
            for prob, idx in zip(top_probs, top_idxs):
                zone = self.idx_to_zone.get(idx.item(), '?')
                if zone != '<pad>' and zone != '0':
                    predictions.append({
                        'zone': zone,
                        'probability': prob.item(),
                        'confidence': prob.item() * 100
                    })
        
        return {
            'rally': rally_string,
            'context': {
                'surface': surface,
                'score': score,
                'server_hand': server_hand,
                'receiver_hand': receiver_hand,
                'is_second_serve': is_second_serve
            },
            'predictions': predictions
        }
    
    def demonstrate(self):
        """Demonstrate model capabilities with example predictions."""
        print("\n" + "="*60)
        print("TENNIS SHOT PREDICTION DEMONSTRATION")
        print("="*60)
        
        examples = [
            {
                'rally': '4s 1f 3b',
                'description': 'Standard rally: serve to zone 4, forehand to 1, backhand to 3',
                'surface': 'Hard'
            },
            {
                'rally': '6s 2f',
                'description': 'Serve return: serve to zone 6, forehand return to 2',
                'surface': 'Clay',
                'score': '30-40'
            },
            {
                'rally': '5s',
                'description': 'Second serve return: serve to zone 5',
                'surface': 'Grass',
                'is_second_serve': True
            },
            {
                'rally': '4s 1f 4b 3f 6b',
                'description': 'Long rally: baseline exchanges',
                'surface': 'Clay',
                'score': '40-40'
            }
        ]
        
        for i, example in enumerate(examples, 1):
            print(f"\nExample {i}: {example['description']}")
            print(f"Rally: {example['rally']}")
            
            # Get prediction
            result = self.predict_next_shot(**example)
            
            if 'error' in result:
                print(f"Error: {result['error']}")
                continue
            
            print(f"Surface: {result['context']['surface']}")
            print(f"Score: {result['context']['score']}")
            
            print("Top predictions:")
            for j, pred in enumerate(result['predictions'], 1):
                print(f"  {j}. Zone {pred['zone']}: {pred['confidence']:.1f}%")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Tennis shot prediction inference')
    
    # Model and config paths
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to training configuration file')
    
    # Dataset paths
    parser.add_argument('--points_path', type=str, required=True,
                        help='Path to tennis points CSV file')
    parser.add_argument('--matches_path', type=str, required=True,
                        help='Path to tennis matches CSV file')
    parser.add_argument('--atp_path', type=str, required=True,
                        help='Path to ATP players CSV file')
    parser.add_argument('--wta_path', type=str, required=True,
                        help='Path to WTA players CSV file')
    
    # Prediction parameters
    parser.add_argument('--rally', type=str, default=None,
                        help='Rally string to predict (e.g., "4s 1f 3b")')
    parser.add_argument('--surface', type=str, default='Hard',
                        choices=['Hard', 'Clay', 'Grass'],
                        help='Court surface')
    parser.add_argument('--score', type=str, default='0-0',
                        help='Current score (e.g., "30-15")')
    parser.add_argument('--second_serve', action='store_true',
                        help='Is this a second serve?')
    parser.add_argument('--server_hand', type=str, default='R',
                        choices=['R', 'L'], help='Server handedness')
    parser.add_argument('--receiver_hand', type=str, default='R',
                        choices=['R', 'L'], help='Receiver handedness')
    
    # Other options
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--demo', action='store_true',
                        help='Run demonstration with example predictions')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    # Setup dataset paths
    dataset_paths = {
        'points': args.points_path,
        'matches': args.matches_path,
        'atp': args.atp_path,
        'wta': args.wta_path
    }
    
    # Initialize predictor
    try:
        predictor = TennisShotPredictor(
            model_path=args.model_path,
            config_path=args.config_path,
            dataset_paths=dataset_paths,
            device=args.device
        )
    except Exception as e:
        print(f"Error initializing predictor: {e}")
        return
    
    # Run demonstration or single prediction
    if args.demo:
        predictor.demonstrate()
    elif args.rally:
        # Single prediction
        result = predictor.predict_next_shot(
            rally_string=args.rally,
            surface=args.surface,
            score=args.score,
            is_second_serve=args.second_serve,
            server_hand=args.server_hand,
            receiver_hand=args.receiver_hand
        )
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            return
        
        print(f"\nRally: {result['rally']}")
        print(f"Context: {result['context']['surface']} court, "
              f"Score {result['context']['score']}")
        print("\nTop predictions:")
        for i, pred in enumerate(result['predictions'], 1):
            print(f"  {i}. Zone {pred['zone']}: {pred['confidence']:.1f}%")
    else:
        print("Please specify --rally for single prediction or --demo for examples")


if __name__ == '__main__':
    main()