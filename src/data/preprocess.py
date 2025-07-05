"""Enhanced data preprocessing module with three-class support and forced SMOTE."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from typing import Tuple, Dict, List, Optional
import warnings
from ..utils.labels import compute_k_pips, create_labels, analyze_label_distribution, auto_adjust_k_pips
warnings.filterwarnings('ignore')

class PairSpecificPreprocessor:
    """Enhanced pair-specific data preprocessor with three-class support."""
    
    def __init__(self, pair_name: str, sequence_length: int = 64, 
                 target_mode: str = 'binary', use_smote: bool = False):
        self.pair_name = pair_name
        self.sequence_length = sequence_length
        self.target_mode = target_mode
        self.use_smote = use_smote
        self.feature_scaler = StandardScaler()
        self.feature_columns = []
        
        # Force SMOTE for three-class mode
        if self.target_mode == 'three_class':
            self.use_smote = True
            print(f"   🔄 Force enabling SMOTE for three-class mode")
        
        # Enhanced pair-specific parameters
        self.pair_configs = {
            'EUR_USD': {
                'volatility_window': 16, 
                'momentum_periods': [4, 8, 16], 
                'ma_periods': [4, 16, 64],
                'enhanced_dropout': 0.55
            },
            'GBP_USD': {
                'volatility_window': 12, 
                'momentum_periods': [3, 6, 12], 
                'ma_periods': [8, 21, 55],
                'enhanced_dropout': 0.50
            },
            'USD_JPY': {
                'volatility_window': 20, 
                'momentum_periods': [5, 10, 20], 
                'ma_periods': [5, 20, 80],
                'enhanced_dropout': 0.50,
                'force_target_mode': 'three_class',
                'force_smote': True
            },
            'AUD_USD': {
                'volatility_window': 14, 
                'momentum_periods': [4, 7, 14], 
                'ma_periods': [4, 16, 64],
                'enhanced_dropout': 0.55
            },
            'XAU_USD': {
                'volatility_window': 8, 
                'momentum_periods': [2, 4, 8], 
                'ma_periods': [3, 9, 21],
                'enhanced_dropout': 0.50,
                'force_target_mode': 'three_class',
                'force_smote': True
            },
            'USD_CHF': {
                'volatility_window': 18, 
                'momentum_periods': [4, 8, 16], 
                'ma_periods': [5, 20, 55],
                'enhanced_dropout': 0.45
            },
            'USD_CAD': {
                'volatility_window': 16, 
                'momentum_periods': [4, 8, 16], 
                'ma_periods': [6, 18, 50],
                'enhanced_dropout': 0.45
            }
        }
        
        self.config = self.pair_configs.get(pair_name, self.pair_configs['EUR_USD'])
        
        # Apply forced configurations for specific pairs
        if 'force_target_mode' in self.config:
            original_mode = self.target_mode
            self.target_mode = self.config['force_target_mode']
            if original_mode != self.target_mode:
                print(f"   🛡️ Force override: {pair_name} target_mode {original_mode} → {self.target_mode}")
        
        if 'force_smote' in self.config and self.config['force_smote']:
            self.use_smote = True
            print(f"   🛡️ Force enabling SMOTE for {pair_name}")
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period, min_periods=1).mean()
        
        return atr
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period, min_periods=1).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD"""
        fast_ema = prices.ewm(span=12).mean()
        slow_ema = prices.ewm(span=26).mean()
        return fast_ema - slow_ema
    
    def create_multi_timeframe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced multi-timeframe features with safety checks."""
        
        print(f"   🔍 Multi-timeframe features creating...")
        print(f"   📊 Available columns before: {len(df.columns)} features")
        
        # SAFETY CHECK: Required columns exist
        required_cols = ['close', 'high', 'low', 'open']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"   ❌ Missing required columns: {missing_cols}")
            return df
        
        try:
            # Primary signals (M5/M15)
            df['primary_rsi'] = self.calculate_rsi(df['close'], 14)
            df['primary_macd'] = self.calculate_macd(df['close'])
            
            # Higher timeframe confirmation (H1)
            # Create a safe copy and ensure proper datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                print(f"   ⚠️ Index is not DatetimeIndex, skipping H1 resampling")
                # Create default H1 trend
                df['h1_trend'] = 0.5  # Neutral value
                df['trend_alignment'] = 0.5
                return df
            
            # Resample to H1 safely
            try:
                df_h1 = df.resample('1H').agg({
                    'open': 'first', 
                    'high': 'max',
                    'low': 'min', 
                    'close': 'last'
                }).dropna()
                
                if len(df_h1) == 0:
                    print(f"   ⚠️ H1 resampling produced empty DataFrame")
                    df['h1_trend'] = 0.5
                    df['trend_alignment'] = 0.5
                    return df
                
                # H1 trend indicators
                df_h1['h1_ema_21'] = df_h1['close'].ewm(span=21, min_periods=1).mean()
                df_h1['h1_trend'] = (df_h1['close'] > df_h1['h1_ema_21']).astype(int)
                
                # Merge back to original timeframe with forward fill
                df = df.merge(df_h1[['h1_trend']], left_index=True, right_index=True, how='left')
                df['h1_trend'] = df['h1_trend'].fillna(method='ffill').fillna(0.5)
                
            except Exception as e:
                print(f"   ⚠️ H1 resampling failed: {e}, using default values")
                df['h1_trend'] = 0.5
            
            # Create SMA features FIRST before using them
            # Check if required SMA columns exist, if not create them
            sma_cols_needed = []
            for period in self.config['ma_periods']:
                sma_col = f'sma_{period}'
                if sma_col not in df.columns:
                    df[sma_col] = df['close'].rolling(period, min_periods=1).mean()
                    sma_cols_needed.append(sma_col)
            
            if sma_cols_needed:
                print(f"   🔧 Created missing SMA columns: {sma_cols_needed}")
            
            # Now safely create trend alignment using existing SMA
            # Find the smallest SMA period for trend alignment
            available_sma_periods = []
            for period in sorted(self.config['ma_periods']):
                sma_col = f'sma_{period}'
                if sma_col in df.columns:
                    available_sma_periods.append(period)
            
            if available_sma_periods:
                # Use the first available SMA period (usually smallest)
                trend_sma_period = available_sma_periods[0]
                sma_col = f'sma_{trend_sma_period}'
                
                print(f"   📈 Using {sma_col} for trend alignment")
                
                # Trend alignment score
                price_above_sma = (df['close'] > df[sma_col]).astype(int)
                df['trend_alignment'] = (price_above_sma + df['h1_trend']) / 2
                
            else:
                print(f"   ⚠️ No SMA columns available for trend alignment")
                df['trend_alignment'] = df['h1_trend']  # Fall back to H1 trend only
            
            print(f"   ✅ {self.pair_name} sequences: {len(sequences):,}")
        print(f"   📐 Shape: {X.shape}")
        
        # Final distribution logging
        if self.target_mode == 'binary':
            class_counts = np.bincount(y.astype(int))
            if len(class_counts) > 1:
                minority_ratio = class_counts.min() / len(y)
                print(f"   🎯 Final minority ratio: {minority_ratio:.3f}")
            else:
                print(f"   ⚠️ Single class detected: {class_counts}")
        else:
            unique, counts = np.unique(y, return_counts=True)
            class_names = {0: 'Long', 1: 'Flat', 2: 'Short'}
            class_breakdown = {class_names.get(cls, cls): count for cls, count in zip(unique, counts)}
            print(f"   🎯 Final class distribution: {class_breakdown}")
        
        return X, y

def create_preprocessor(pair_name: str, config: dict, target_mode: str = None) -> PairSpecificPreprocessor:
    """Factory function to create preprocessor from configuration."""
    data_config = config.get('data', {})
    model_config = config.get('model', {})
    
    # Override target_mode if specified
    final_target_mode = target_mode or model_config.get('target_mode', 'binary')
    
    return PairSpecificPreprocessor(
        pair_name=pair_name,
        sequence_length=data_config.get('sequence_length', 64),
        target_mode=final_target_mode,
        use_smote=config.get('training', {}).get('use_smote', False)
    )

__all__ = ['PairSpecificPreprocessor', 'create_preprocessor'] Multi-timeframe features created successfully")
            
        except Exception as e:
            print(f"   ❌ Multi-timeframe feature creation failed: {e}")
            # Create safe fallback values
            df['primary_rsi'] = 50.0  # Neutral RSI
            df['primary_macd'] = 0.0  # Neutral MACD
            df['h1_trend'] = 0.5     # Neutral trend
            df['trend_alignment'] = 0.5  # Neutral alignment
        
        return df
        
    def create_pair_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced technical indicators with three-class optimizations."""
        df = df.copy()
        
        print(f"   🛠️ Creating pair-specific features for {self.pair_name}")
        
        # Basic price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['price_acceleration'] = df['returns'].diff()
        
        # Enhanced price movement features for three-class
        if self.target_mode == 'three_class':
            df['price_velocity'] = df['close'].diff()
            df['price_momentum_3'] = df['close'] / df['close'].shift(3) - 1
            df['price_momentum_5'] = df['close'] / df['close'].shift(5) - 1
        
        # Session features
        df['hour'] = df.index.hour
        df['london_session'] = ((df['hour'] >= 8) & (df['hour'] <= 16)).astype(int)
        df['ny_session'] = ((df['hour'] >= 13) & (df['hour'] <= 21)).astype(int)
        df['tokyo_session'] = ((df['hour'] >= 23) | (df['hour'] <= 7)).astype(int)
        df['overlap_session'] = ((df['hour'] >= 13) & (df['hour'] <= 16)).astype(int)
        
        # Pair-specific session weights
        if self.pair_name in ['EUR_USD', 'GBP_USD']:
            df['session_score'] = (df['london_session'] * 2 + 
                                 df['ny_session'] * 2 + 
                                 df['overlap_session'] * 3)
        elif self.pair_name == 'USD_JPY':
            df['session_score'] = (df['tokyo_session'] * 2 + 
                                 df['london_session'] * 2 + 
                                 df['overlap_session'] * 1.5)
        elif self.pair_name == 'AUD_USD':
            df['session_score'] = (df['tokyo_session'] * 1.5 + 
                                 df['london_session'] * 2 + 
                                 df['ny_session'] * 2)
        elif self.pair_name == 'XAU_USD':
            df['session_score'] = (df['london_session'] * 3 + 
                                 df['ny_session'] * 3 + 
                                 df['overlap_session'] * 4)
            # Gold-specific features
            df['risk_on_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 15)).astype(int)
            df['asian_uncertainty'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        else:
            # Default for USD_CHF, USD_CAD
            df['session_score'] = (df['london_session'] * 1.5 + 
                                 df['ny_session'] * 2 + 
                                 df['overlap_session'] * 2.5)
        
        # Volatility features
        vol_window = self.config['volatility_window']
        df[f'volatility_{vol_window}'] = df['returns'].rolling(vol_window, min_periods=1).std()
        df[f'volatility_ratio'] = (df[f'volatility_{vol_window}'] / 
                                  df[f'volatility_{vol_window}'].rolling(64, min_periods=1).mean())
        
        # Enhanced volatility for three-class
        if self.target_mode == 'three_class':
            df['volatility_short'] = df['returns'].rolling(4, min_periods=1).std()
            df['volatility_long'] = df['returns'].rolling(32, min_periods=1).std()
            df['volatility_regime'] = df['volatility_short'] / (df['volatility_long'] + 1e-8)
        
        # Momentum features
        for period in self.config['momentum_periods']:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
            df[f'rsi_{period}'] = self.calculate_rsi(df['close'], period)
        
        # Moving averages - CREATE THEM HERE
        for period in self.config['ma_periods']:
            df[f'sma_{period}'] = df['close'].rolling(period, min_periods=1).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, min_periods=1).mean()
            
            # MA cross signals for three-class
            if self.target_mode == 'three_class':
                df[f'price_vs_sma_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']
        
        # Gold-specific features
        if self.pair_name == 'XAU_USD':
            df['intraday_range'] = (df['high'] - df['low']) / df['open']
            df['gap_size'] = abs(df['open'] - df['close'].shift(1)) / df['close'].shift(1)
            df['gold_volatility_ultra'] = df['returns'].rolling(4, min_periods=1).std()
            df['gold_volatility_short'] = df['returns'].rolling(8, min_periods=1).std()
            df['gold_momentum_1'] = df['close'] / df['close'].shift(1) - 1
            df['gold_momentum_3'] = df['close'] / df['close'].shift(3) - 1
        
        # Price position features
        for window in [16, 32, 64]:
            high_window = df['high'].rolling(window, min_periods=1).max()
            low_window = df['low'].rolling(window, min_periods=1).min()
            df[f'price_position_{window}'] = ((df['close'] - low_window) / 
                                             (high_window - low_window + 1e-8))
        
        # Bollinger Bands
        bb_period = 20
        df['bb_middle'] = df['close'].rolling(bb_period, min_periods=1).mean()
        bb_std = df['close'].rolling(bb_period, min_periods=1).std()
        df['bb_upper'] = df['bb_middle'] + 2 * bb_std
        df['bb_lower'] = df['bb_middle'] - 2 * bb_std
        df['bb_position'] = ((df['close'] - df['bb_lower']) / 
                           (df['bb_upper'] - df['bb_lower'] + 1e-8))
        
        # MACD
        fast_ema = df['close'].ewm(span=12).mean()
        slow_ema = df['close'].ewm(span=26).mean()
        df['macd'] = fast_ema - slow_ema
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Volume and spread features
        if 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(20, min_periods=1).mean()
            df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-8)
        else:
            df['volume_ratio'] = 1.0  # Default neutral value
        
        if 'spread' in df.columns:
            df['spread_ma'] = df['spread'].rolling(20, min_periods=1).mean()
            df['spread_ratio'] = df['spread'] / (df['spread_ma'] + 1e-8)
        else:
            df['spread_ratio'] = 1.0  # Default neutral value
        
        # Three-class specific features
        if self.target_mode == 'three_class':
            # Trend strength indicators
            df['trend_strength'] = abs(df['macd_histogram'])
            df['price_acceleration_smooth'] = df['price_acceleration'].rolling(3, min_periods=1).mean()
            
            # Support/Resistance levels
            df['support_level'] = df['low'].rolling(20, min_periods=1).min()
            df['resistance_level'] = df['high'].rolling(20, min_periods=1).max()
            df['support_distance'] = (df['close'] - df['support_level']) / df['close']
            df['resistance_distance'] = (df['resistance_level'] - df['close']) / df['close']
        
        print(f"   📊 Created {len(df.columns)} base features")
        
        # Multi-timeframe features - CALL THIS AFTER MA CREATION
        df = self.create_multi_timeframe_features(df)
        
        print(f"   📊 Final feature count: {len(df.columns)}")
        
        # Clean data
        df = df.replace([np.inf, -np.inf], np.nan)
        
        return df
    
    def prepare_pair_data(self, df: pd.DataFrame, horizon: int = 64, k_pips: Optional[float] = None, 
                         dynamic_k_aggressive: bool = True, seq_len: Optional[int] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data with enhanced three-class support and auto k-pips adjustment."""
        print(f"🔧 {self.pair_name} data preparation ({self.target_mode})...")
        
        # Horizon guard
        if horizon < 32:
            print(f"   ⚠️ Horizon {horizon} < 32, setting to 32")
            horizon = 32
        if horizon > 256:
            raise ValueError(f"Horizon {horizon} > 256 safety limit")
        
        # Sequence length calculation
        if seq_len is None:
            self.sequence_length = max(16, round(horizon/16) * 16)
        else:
            self.sequence_length = max(16, seq_len)
        
        print(f"   📐 Sequence length: {self.sequence_length} (horizon: {horizon})")
        
        # Create features
        featured_data = self.create_pair_specific_features(df)
        
        # Dynamic k-pips calculation with enhanced formula
        if k_pips is None:
            k_pips = compute_k_pips(
                featured_data, 
                horizon, 
                pair_name=self.pair_name, 
                mode='enhanced'
            )
        
        # Auto-adjust k-pips to avoid single class problems
        k_pips = auto_adjust_k_pips(
            featured_data, 
            horizon, 
            k_pips, 
            target_mode=self.target_mode, 
            max_iterations=3
        )
        
        # Create targets using enhanced labels module
        target = create_labels(featured_data, k_pips, horizon, self.target_mode)
        
        # Analyze label distribution
        label_stats = analyze_label_distribution(target, self.pair_name, self.target_mode)
        
        # Feature selection with three-class enhancements
        if self.pair_name == 'XAU_USD':
            core_features = [
                'returns', 'log_returns', 'price_acceleration', 'session_score',
                'risk_on_hours', 'asian_uncertainty', 'intraday_range', 'gap_size',
                'gold_volatility_ultra', 'gold_volatility_short',
                'gold_momentum_1', 'gold_momentum_3', 'volume_ratio', 'bb_position', 
                'macd_histogram', 'spread_ratio'
            ]
            if self.target_mode == 'three_class':
                core_features.extend(['trend_strength', 'support_distance', 'resistance_distance'])
        else:
            core_features = [
                'returns', 'log_returns', 'price_acceleration', 'session_score',
                f'volatility_{self.config["volatility_window"]}', 'volatility_ratio',
                'volume_ratio', 'bb_position', 'macd_histogram', 'spread_ratio'
            ]
            
            # Add pair-specific features
            for period in self.config['momentum_periods']:
                core_features.extend([f'momentum_{period}', f'rsi_{period}'])
            
            # Three-class specific features
            if self.target_mode == 'three_class':
                core_features.extend([
                    'volatility_short', 'volatility_long', 'volatility_regime',
                    'price_velocity', 'price_momentum_3', 'price_momentum_5',
                    'trend_strength', 'price_acceleration_smooth'
                ])
                
                # Add MA cross features
                for period in self.config['ma_periods']:
                    feature_name = f'price_vs_sma_{period}'
                    if feature_name in featured_data.columns:
                        core_features.append(feature_name)
        
        # Add multi-timeframe features - with safety check
        multi_timeframe_features = ['primary_rsi', 'primary_macd', 'h1_trend', 'trend_alignment']
        for feature in multi_timeframe_features:
            if feature in featured_data.columns:
                core_features.append(feature)
        
        for window in [16, 32, 64]:
            feature_name = f'price_position_{window}'
            if feature_name in featured_data.columns:
                core_features.append(feature_name)
        
        # Filter existing features
        available_features = [f for f in core_features if f in featured_data.columns]
        features = featured_data[available_features].copy()
        
        print(f"   🎯 {self.pair_name} features: {len(available_features)}")
        print(f"   📊 Target distribution: {label_stats['class_counts']}")
        
        if self.target_mode == 'three_class':
            class_names = {0: 'Long', 1: 'Flat', 2: 'Short'}
            class_str = ', '.join([f"{class_names.get(k, k)}: {v}" for k, v in label_stats['class_counts'].items()])
            print(f"   🎯 Class breakdown: {class_str}")
        
        # Warn about problematic distributions
        if label_stats['is_single_class']:
            print(f"   ⚠️ WARN: Single class problem detected!")
        elif self.target_mode == 'binary' and label_stats.get('minority_ratio', 0) < 0.15:
            print(f"   ⚠️ WARN: Severe class imbalance (minority: {label_stats['minority_ratio']:.3f})")
        
        # Clean data
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        target = target.fillna(method='ffill').fillna(method='bfill')
        
        # Remove outliers
        for col in features.select_dtypes(include=[np.number]).columns:
            mean = features[col].mean()
            std = features[col].std()
            features[col] = features[col].clip(mean - 3*std, mean + 3*std)
        
        # Final valid data
        valid_mask = ~(features.isna().any(axis=1) | target.isna())
        features_clean = features[valid_mask]
        target_clean = target[valid_mask]
        
        print(f"   ✅ {self.pair_name} clean data: {len(features_clean):,} records")
        
        self.feature_columns = features_clean.columns.tolist()
        return features_clean, target_clean
    
    def force_smote_three_class(self, features_scaled: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Enhanced SMOTE for three-class mode with better handling."""
        try:
            unique_classes, class_counts = np.unique(target, return_counts=True)
            print(f"   📊 Pre-SMOTE classes: {dict(zip(unique_classes, class_counts))}")
            
            if len(unique_classes) < 2:
                print(f"   ⚠️ Single class detected - creating synthetic minority classes")
                
                # Create synthetic samples for missing classes
                n_synthetic = max(5, len(features_scaled) // 50)
                
                if self.target_mode == 'three_class':
                    # Ensure all three classes exist
                    all_features = [features_scaled]
                    all_targets = [target]
                    
                    for missing_class in [0, 1, 2]:
                        if missing_class not in unique_classes:
                            # Create synthetic features with slight noise
                            synthetic_indices = np.random.choice(len(features_scaled), n_synthetic)
                            synthetic_features = features_scaled[synthetic_indices] + np.random.normal(0, 0.01, (n_synthetic, features_scaled.shape[1]))
                            synthetic_target = np.full(n_synthetic, missing_class)
                            
                            all_features.append(synthetic_features)
                            all_targets.append(synthetic_target)
                            print(f"   🔄 Added {n_synthetic} synthetic samples for class {missing_class}")
                    
                    features_scaled = np.vstack(all_features)
                    target = np.hstack(all_targets)
            
            # Apply SMOTE with appropriate k_neighbors
            unique_classes_updated = np.unique(target)
            min_class_size = min([np.sum(target == cls) for cls in unique_classes_updated])
            k_neighbors = min(5, max(1, min_class_size - 1))
            
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            features_scaled, target = smote.fit_resample(features_scaled, target)
            
            final_unique, final_counts = np.unique(target, return_counts=True)
            print(f"   ✅ Post-SMOTE classes: {dict(zip(final_unique, final_counts))}")
            
        except Exception as e:
            print(f"   ❌ SMOTE failed: {e}")
        
        return features_scaled, target
    
    def create_sequences(self, features: pd.DataFrame, target: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training with enhanced SMOTE support."""
        # Normalization
        self.feature_scaler.fit(features)
        features_scaled = self.feature_scaler.transform(features)
        
        # Apply SMOTE if needed (forced for three-class mode)
        if self.use_smote:
            if self.target_mode == 'three_class':
                features_scaled, target_array = self.force_smote_three_class(features_scaled, target.values)
                target = pd.Series(target_array)
            else:
                # Binary mode SMOTE
                class_counts = target.value_counts()
                if len(class_counts) > 1:
                    minority_ratio = class_counts.min() / class_counts.sum()
                    print(f"   📊 Minority ratio: {minority_ratio:.3f}")
                    
                    if minority_ratio < 0.15:
                        try:
                            smote = SMOTE(random_state=42, k_neighbors=min(5, class_counts.min() - 1))
                            features_scaled, target_array = smote.fit_resample(features_scaled, target.values)
                            target = pd.Series(target_array)
                            print(f"   ✅ Binary SMOTE applied: {len(features_scaled):,} samples")
                        except Exception as e:
                            print(f"   ❌ Binary SMOTE failed: {e}")
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(self.sequence_length, len(features_scaled)):
            seq = features_scaled[i-self.sequence_length:i]
            tgt = target.iloc[i] if isinstance(target, pd.Series) else target[i]
            
            sequences.append(seq)
            targets.append(tgt)
        
        X = np.array(sequences)
        y = np.array(targets)
        
        print(f"   ✅
