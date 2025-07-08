# src/inference/predictor.py
"""Enhanced Transformer Inference Pipeline for CSV Prediction Export - FIXED CACHE."""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import glob
import logging

from ..data.fetcher import create_fetcher
from ..data.preprocess import create_preprocessor
from ..models.factory import create_model

logger = logging.getLogger(__name__)

class EnhancedTransformerPredictor:
    """
    Enhanced Transformer inference pipeline for generating trading predictions.
    
    Supports loading trained models and generating CSV predictions for multiple pairs.
    """
    
    def __init__(self, model_dir: str = "models", output_dir: str = "predictions"):
        """
        Initialize predictor.
        
        Args:
            model_dir: Directory containing trained models
            output_dir: Directory to save prediction CSVs
        """
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Inference device: {self.device}")
        
        # Store loaded models and preprocessors
        self.models = {}
        self.preprocessors = {}
        self.model_configs = {}
    
    def load_model(self, pair_name: str, model_path: str = None) -> bool:
        """
        Load trained model for a specific pair.
        
        Args:
            pair_name: Currency pair name (e.g., "EUR_USD")
            model_path: Specific model path (if None, auto-detect latest)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if model_path is None:
                # Auto-detect latest model for pair
                model_pattern = f"{pair_name}_enhanced_transformer_model_*.pth"
                model_files = list(self.model_dir.glob(model_pattern))
                
                if not model_files:
                    print(f"❌ No models found for {pair_name}")
                    return False
                
                # Get latest model
                model_path = max(model_files, key=lambda x: x.stat().st_mtime)
            
            print(f"📁 Loading model for {pair_name}: {model_path}")
            
            # Load model data
            model_data = torch.load(model_path, map_location=self.device)
            
            # Extract configuration
            config = model_data['config']
            input_size = model_data['input_size']
            target_mode = model_data.get('target_mode', 'binary')
            
            # Create model using factory
            model = create_model(
                model_type='enhanced_transformer',
                config=config,
                n_features=input_size,
                device=self.device
            )
            
            # Load weights
            model.load_state_dict(model_data['model_state_dict'])
            model.eval()
            
            # Store model and config
            self.models[pair_name] = model
            self.model_configs[pair_name] = {
                'config': config,
                'target_mode': target_mode,
                'input_size': input_size,
                'feature_columns': model_data.get('feature_columns', [])
            }
            
            print(f"✅ {pair_name} model loaded successfully")
            print(f"   📊 Input size: {input_size}")
            print(f"   🎯 Target mode: {target_mode}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model for {pair_name}: {str(e)}")
            return False
    
    def prepare_data(self, pair_name: str, lookback_candles: int = 30000, 
                    granularity: str = "M15", cache_enabled: bool = True) -> Optional[pd.DataFrame]:
        """
        Prepare data for inference - FIXED CACHE FORMAT.
        
        Args:
            pair_name: Currency pair name (e.g., "EUR_USD")
            lookback_candles: Number of candles to fetch
            granularity: Data granularity
            cache_enabled: Use cache if available
            
        Returns:
            Prepared DataFrame or None if failed
        """
        try:
            if pair_name not in self.model_configs:
                print(f"❌ Model not loaded for {pair_name}")
                return None
            
            config = self.model_configs[pair_name]['config']
            
            # Create fetcher with cache enabled
            fetcher = create_fetcher(config, cache_enabled=cache_enabled)
            
            # Fetch data - FIXED: Use pair_name as-is (EUR_USD format)
            print(f"📊 Fetching data for {pair_name}...")
            data = fetcher.fetch_pair_data(
                instrument=pair_name,  # ✅ FIXED: Keep EUR_USD format (not EUR/USD)
                granularity=granularity,
                lookback_candles=lookback_candles,
                target_candles=lookback_candles
            )
            
            if data is None or len(data) < 1000:
                print(f"❌ Insufficient data for {pair_name}")
                return None
            
            print(f"✅ {pair_name}: {len(data)} records fetched")
            return data
            
        except Exception as e:
            print(f"❌ Data preparation failed for {pair_name}: {str(e)}")
            return None
    
    def generate_predictions(self, pair_name: str, data: pd.DataFrame, 
                           seq_len: int = 64) -> List[Dict]:
        """
        Generate predictions for a pair.
        
        Args:
            pair_name: Currency pair name
            data: Market data
            seq_len: Sequence length
            
        Returns:
            List of prediction dictionaries
        """
        try:
            if pair_name not in self.models:
                print(f"❌ Model not loaded for {pair_name}")
                return []
            
            model = self.models[pair_name]
            config = self.model_configs[pair_name]['config']
            target_mode = self.model_configs[pair_name]['target_mode']
            
            # Create preprocessor
            if pair_name not in self.preprocessors:
                preprocessor = create_preprocessor(pair_name, config)
                preprocessor.target_mode = target_mode
                self.preprocessors[pair_name] = preprocessor
            else:
                preprocessor = self.preprocessors[pair_name]
            
            print(f"🔮 Generating predictions for {pair_name}...")
            
            # Prepare features with targets (needed for scaler fitting)
            features, target = preprocessor.prepare_pair_data(data, horizon=64)
            
            if len(features) < seq_len:
                print(f"❌ Insufficient features for {pair_name}")
                return []
            
            # ✅ FIXED: Create sequences to fit the scaler
            X, y = preprocessor.create_sequences(features, target)
            
            if len(X) < seq_len:
                print(f"❌ Insufficient sequences for {pair_name}")
                return []
            
            # Now scaler is fitted, we can use it for inference
            features_scaled = preprocessor.feature_scaler.transform(features)
            
            # Generate predictions
            predictions = []
            model.eval()
            
            with torch.no_grad():
                for i in range(seq_len, len(features_scaled)):
                    # Get sequence
                    sequence = features_scaled[i-seq_len:i]
                    sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                    
                    # Get timestamp
                    timestamp = features.index[i]
                    
                    # Predict
                    output = model(sequence_tensor)
                    
                    if target_mode == 'binary':
                        # Binary prediction
                        probability = torch.sigmoid(output).cpu().numpy()[0][0]
                        prediction = 1 if probability > 0.5 else 0
                        confidence = abs(probability - 0.5) * 2  # Convert to 0-1 range
                        
                        # Filter: Only include confident predictions
                        if confidence > 0.1:  # Minimum confidence threshold
                            predictions.append({
                                'datetime': timestamp,
                                'prediction': prediction,
                                'confidence_score': float(confidence),
                                'raw_probability': float(probability),
                                'signal_type': 'long' if prediction == 1 else 'short'
                            })
                    
                    else:  # three_class
                        # Three-class prediction
                        probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
                        predicted_class = np.argmax(probabilities)
                        confidence = np.max(probabilities)
                        
                        # Filter: Only non-flat predictions with good confidence
                        if predicted_class != 1 and confidence > 0.4:  # Not flat, good confidence
                            signal_type = 'long' if predicted_class == 0 else 'short'
                            binary_pred = 1 if predicted_class == 0 else 0
                            
                            predictions.append({
                                'datetime': timestamp,
                                'prediction': binary_pred,
                                'confidence_score': float(confidence),
                                'raw_probabilities': probabilities.tolist(),
                                'signal_type': signal_type
                            })
            
            print(f"✅ {pair_name}: {len(predictions)} predictions generated")
            return predictions
            
        except Exception as e:
            print(f"❌ Prediction generation failed for {pair_name}: {str(e)}")
            return []
    
    def save_predictions_csv(self, pair_name: str, predictions: List[Dict]) -> str:
        """
        Save predictions to CSV file.
        
        Args:
            pair_name: Currency pair name
            predictions: List of predictions
            
        Returns:
            Path to saved CSV file
        """
        if not predictions:
            print(f"⚠️ No predictions to save for {pair_name}")
            return ""
        
        # Convert to DataFrame
        df = pd.DataFrame(predictions)
        
        # Sort by datetime
        df = df.sort_values('datetime')
        
        # Format datetime
        df['datetime'] = pd.to_datetime(df['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Select final columns
        final_columns = ['datetime', 'prediction', 'confidence_score', 'signal_type']
        df_final = df[final_columns]
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"predictions_{pair_name}_{timestamp}.csv"
        filepath = self.output_dir / filename
        
        df_final.to_csv(filepath, index=False)
        
        print(f"💾 {pair_name} predictions saved: {filepath}")
        print(f"   📊 Total signals: {len(df_final)}")
        print(f"   📈 Long signals: {len(df_final[df_final['signal_type'] == 'long'])}")
        print(f"   📉 Short signals: {len(df_final[df_final['signal_type'] == 'short'])}")
        
        return str(filepath)
    
    def run_inference_pipeline(self, pairs: List[str], 
                             lookback_candles: int = 30000,
                             granularity: str = "M15",
                             seq_len: int = 64,
                             cache_enabled: bool = True) -> Dict[str, str]:
        """
        Run complete inference pipeline for multiple pairs.
        
        Args:
            pairs: List of currency pairs
            lookback_candles: Number of candles to fetch
            granularity: Data granularity
            seq_len: Sequence length
            cache_enabled: Use cached data
            
        Returns:
            Dictionary mapping pair names to CSV file paths
        """
        print(f"🚀 ENHANCED TRANSFORMER INFERENCE PIPELINE")
        print(f"{'='*60}")
        print(f"   📊 Pairs: {pairs}")
        print(f"   📈 Lookback candles: {lookback_candles}")
        print(f"   ⏰ Granularity: {granularity}")
        print(f"   📐 Sequence length: {seq_len}")
        print(f"   💾 Cache enabled: {cache_enabled}")
        
        results = {}
        
        for pair_name in pairs:
            print(f"\n{'='*50}")
            print(f"🎯 Processing {pair_name}")
            print(f"{'='*50}")
            
            # Load model
            if not self.load_model(pair_name):
                continue
            
            # Prepare data - FIXED: Pass cache_enabled parameter
            data = self.prepare_data(pair_name, lookback_candles, granularity, cache_enabled)
            if data is None:
                continue
            
            # Generate predictions
            predictions = self.generate_predictions(pair_name, data, seq_len)
            if not predictions:
                continue
            
            # Save CSV
            csv_path = self.save_predictions_csv(pair_name, predictions)
            if csv_path:
                results[pair_name] = csv_path
        
        # Summary
        print(f"\n🎉 INFERENCE PIPELINE COMPLETED")
        print(f"{'='*50}")
        successful_pairs = len(results)
        total_pairs = len(pairs)
        print(f"📈 Success rate: {successful_pairs}/{total_pairs}")
        
        for pair_name, csv_path in results.items():
            print(f"✅ {pair_name}: {Path(csv_path).name}")
        
        return results


def run_inference_cli(pairs: List[str], lookback_candles: int = 30000,
                     granularity: str = "M15", seq_len: int = 64,
                     model_dir: str = "models", output_dir: str = "predictions",
                     cache_enabled: bool = True):
    """
    CLI wrapper for inference pipeline - FIXED CACHE SUPPORT.
    
    Args:
        pairs: List of currency pairs
        lookback_candles: Number of candles
        granularity: Data granularity
        seq_len: Sequence length
        model_dir: Models directory
        output_dir: Output directory
        cache_enabled: Use cached data
    """
    predictor = EnhancedTransformerPredictor(model_dir, output_dir)
    return predictor.run_inference_pipeline(pairs, lookback_candles, granularity, seq_len, cache_enabled)


if __name__ == "__main__":
    # Example usage
    pairs = ["EUR_USD", "GBP_USD", "USD_JPY"]
    
    results = run_inference_cli(
        pairs=pairs,
        lookback_candles=30000,
        granularity="M15",
        seq_len=64,
        cache_enabled=True
    )
    
    print("\n📁 Generated CSV files:")
    for pair, path in results.items():
        print(f"   {pair}: {path}")
