"""
Enhanced Preprocessing Module - PDF Optimized
Full integration with existing system + 50+ features target

Key Improvements:
- Multi-timeframe features (M5+M15+H1) - PDF Priority
- Optimal technical indicators: RSI-21, MACD(8,21,5), BB(14,1.5)
- 50+ features for Enhanced Transformer
- Pair-specific features (EUR/USD, GBP/USD, USD/JPY, XAU/USD, AUD/USD)
- Z-score normalization for transformers
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

from ..utils.labels import create_labels, compute_k_pips, auto_adjust_k_pips, analyze_label_distribution
from .. import config

class PairSpecificPreprocessor:
    """Enhanced preprocessor with PDF optimizations and 50+ features target."""
    
    def __init__(self, pair_name: str, sequence_length: int = 64, target_mode: str = 'binary', use_smote: bool = False):
        self.pair_name = pair_name
        self.sequence_length = sequence_length
        self.target_mode = target_mode
        self.use_smote = use_smote
        self.feature_scaler = StandardScaler()
        self.feature_columns = []
        
        # PDF-optimized indicator parameters
        self.pdf_indicators = {
            'rsi_periods': [14, 21],  # PDF: RSI-21 for M5 reduces noise
            'ema_periods': [8, 12, 21, 55],  # PDF: EMA 18.3% importance
            'sma_periods': [4, 8, 16, 21, 55, 64, 80],
            'macd_config': (8, 21, 5),  # PDF: Faster signals
            'bb_config': (14, 1.5),    # PDF: Tighter volatility detection
            'atr_periods': [14, 32],
            'momentum_periods': [3, 5, 8, 12, 16, 20],
            'volatility_windows': [8, 12, 16, 20],
            'price_position_windows': [16, 32, 64, 128]
        }
        
        # Enhanced pair-specific configurations
        self.pair_configs = {
            'EUR_USD': {
                'volatility_window': 20, 
                'momentum_periods': [5, 8, 12, 20],
                'ma_periods': [8, 21, 55, 80],
                'enhanced_dropout': 0.15,
                'special_features': ['ecb_fed_divergence', 'policy_signals']
            },
            'GBP_USD': {
                'volatility_window': 12, 
                'momentum_periods': [3, 5, 12, 16], 
                'ma_periods': [8, 16, 50],
                'enhanced_dropout': 0.20,
                'special_features': ['brexit_sentiment', 'boe_signals']
            },
            'USD_JPY': {
                'volatility_window': 16,
                'momentum_periods': [5, 10, 16],
                'ma_periods': [5, 20, 60],
                'enhanced_dropout': 0.15,
                'force_target_mode': 'three_class',
                'force_smote': True,
                'special_features': ['risk_sentiment', 'carry_trade']
            },
            'XAU_USD': {
                'volatility_window': 8,
                'momentum_periods': [2, 3, 5, 8],
                'ma_periods': [3, 8, 21],
                'enhanced_dropout': 0.25,
                'force_target_mode': 'three_class',
                'force_smote': True,
                'special_features': ['vix_correlation', 'inflation_hedge']
            },
            'AUD_USD': {
                'volatility_window': 16,
                'momentum_periods': [5, 8, 16, 20],
                'ma_periods': [4, 16, 50],
                'enhanced_dropout': 0.15,
                'special_features': ['commodity_correlation', 'china_sentiment']
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
        """Calculate RSI with PDF optimization."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period, min_periods=1).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, prices: pd.Series, fast: int = 8, slow: int = 21, signal: int = 5) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD with PDF-optimized parameters."""
        fast_ema = prices.ewm(span=fast).mean()
        slow_ema = prices.ewm(span=slow).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def create_enhanced_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive technical indicators - PDF optimized for 50+ features."""
        
        print(f"   📊 Creating enhanced technical indicators (PDF optimized)...")
        
        # 1. PRICE-BASED FEATURES (PDF: Close Price 18.7% importance)
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['price_change'] = df['close'] - df['open']
        df['price_change_pct'] = (df['close'] - df['open']) / df['open']
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']
        df['oc_ratio'] = abs(df['open'] - df['close']) / df['close']
        
        # Price gaps and acceleration
        df['gap_up'] = np.maximum(0, df['low'] - df['close'].shift(1))
        df['gap_down'] = np.maximum(0, df['close'].shift(1) - df['high'])
        df['price_acceleration'] = df['returns'].diff()
        df['price_momentum'] = df['returns'].rolling(3).sum()
        
        # True Range
        df['true_range'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        
        # 2. MOVING AVERAGES (PDF: EMA 18.3% importance)
        for period in self.pdf_indicators['ema_periods']:
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            df[f'ema_slope_{period}'] = df[f'ema_{period}'].diff()
            df[f'price_vs_ema_{period}'] = (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}']
        
        for period in self.pdf_indicators['sma_periods']:
            df[f'sma_{period}'] = df['close'].rolling(period, min_periods=1).mean()
            df[f'price_vs_sma_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']
        
        # Moving average crosses
        df['ema_cross_8_21'] = (df['ema_8'] > df['ema_21']).astype(int)
        df['ema_cross_21_55'] = (df['ema_21'] > df['ema_55']).astype(int)
        df['ma_alignment'] = (df['ema_cross_8_21'] + df['ema_cross_21_55']) / 2
        
        # 3. RSI (PDF: 15.5% importance, RSI-21 optimal for M5)
        for period in self.pdf_indicators['rsi_periods']:
            df[f'rsi_{period}'] = self.calculate_rsi(df['close'], period)
            df[f'rsi_{period}_slope'] = df[f'rsi_{period}'].diff()
            df[f'rsi_{period}_overbought'] = (df[f'rsi_{period}'] > 70).astype(int)
            df[f'rsi_{period}_oversold'] = (df[f'rsi_{period}'] < 30).astype(int)
        
        # RSI divergence
        df['rsi_price_divergence'] = (df['rsi_14'].diff() * df['close'].diff() < 0).astype(int)
        
        # 4. MACD (PDF: 8,21,5 for faster signals)
        fast, slow, signal = self.pdf_indicators['macd_config']
        df['macd'], df['macd_signal'], df['macd_histogram'] = self.calculate_macd(df['close'], fast, slow, signal)
        df['macd_histogram_slope'] = df['macd_histogram'].diff()
        df['macd_cross'] = (df['macd'] > df['macd_signal']).astype(int)
        df['macd_momentum'] = df['macd'].diff()
        
        # 5. BOLLINGER BANDS (PDF: 14.7% importance, 14,1.5 for tighter detection)
        bb_period, bb_std = self.pdf_indicators['bb_config']
        df['bb_middle'] = df['close'].rolling(bb_period, min_periods=1).mean()
        bb_std_val = df['close'].rolling(bb_period, min_periods=1).std()
        df['bb_upper'] = df['bb_middle'] + bb_std * bb_std_val
        df['bb_lower'] = df['bb_middle'] - bb_std * bb_std_val
        df['bb_position'] = ((df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8))
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).mean()).astype(int)
        df['bb_break_upper'] = (df['close'] > df['bb_upper']).astype(int)
        df['bb_break_lower'] = (df['close'] < df['bb_lower']).astype(int)
        
        # 6. ATR and VOLATILITY
        for period in self.pdf_indicators['atr_periods']:
            df[f'atr_{period}'] = self.calculate_atr(df, period)
            df[f'atr_ratio_{period}'] = df['true_range'] / df[f'atr_{period}']
            df[f'atr_normalized_{period}'] = df[f'atr_{period}'] / df['close']
        
        for window in self.pdf_indicators['volatility_windows']:
            df[f'volatility_{window}'] = df['returns'].rolling(window, min_periods=1).std()
            df[f'volatility_ratio_{window}'] = (df[f'volatility_{window}'] / 
                                              df[f'volatility_{window}'].rolling(20).mean())
            df[f'volatility_percentile_{window}'] = df[f'volatility_{window}'].rolling(50).rank(pct=True)
        
        # 7. MOMENTUM INDICATORS
        for period in self.pdf_indicators['momentum_periods']:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
            df[f'roc_{period}'] = df['close'].pct_change(period)
            df[f'momentum_smooth_{period}'] = df[f'momentum_{period}'].rolling(3).mean()
        
        # Momentum divergences
        df['momentum_divergence_5_20'] = df['momentum_5'] - df['momentum_20']
        df['momentum_acceleration'] = df['momentum_5'].diff()
        
        # 8. PRICE POSITION FEATURES
        for window in self.pdf_indicators['price_position_windows']:
            high_window = df['high'].rolling(window, min_periods=1).max()
            low_window = df['low'].rolling(window, min_periods=1).min()
            df[f'price_position_{window}'] = ((df['close'] - low_window) / (high_window - low_window + 1e-8))
            df[f'high_low_ratio_{window}'] = (df['high'] - df['low']) / (high_window - low_window + 1e-8)
            df[f'support_resistance_{window}'] = np.minimum(
                df['close'] - low_window, high_window - df['close']
            ) / df['close']
        
        # 9. VOLUME FEATURES (PDF: 14.6% importance)
        if 'volume' in df.columns:
            df['volume_ma_20'] = df['volume'].rolling(20, min_periods=1).mean()
            df['volume_ratio'] = df['volume'] / (df['volume_ma_20'] + 1e-8)
            df['volume_spike'] = (df['volume'] > df['volume_ma_20'] * 2).astype(int)
            df['volume_momentum'] = df['volume'].pct_change()
            df['volume_price_correlation'] = df['volume'].rolling(20).corr(df['close'])
        else:
            # Synthetic volume based on price action
            df['synthetic_volume'] = abs(df['close'] - df['open']) * (df['high'] - df['low'])
            df['volume_ratio'] = df['synthetic_volume'] / df['synthetic_volume'].rolling(20).mean()
            df['volume_spike'] = 0
            df['volume_momentum'] = 0.0
            df['volume_price_correlation'] = 0.0
        
        # 10. SPREAD FEATURES (if available)
        if 'spread' in df.columns:
            df['spread_ma'] = df['spread'].rolling(20, min_periods=1).mean()
            df['spread_ratio'] = df['spread'] / (df['spread_ma'] + 1e-8)
            df['spread_percentile'] = df['spread'].rolling(50).rank(pct=True)
        else:
            df['spread_ratio'] = 1.0
            df['spread_percentile'] = 0.5
        
        return df
    
    def create_multi_timeframe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced multi-timeframe features with M5+M15+H1 integration.
        PDF Priority: Expected +5-8% accuracy improvement.
        """
        
        print(f"   🔍 Creating multi-timeframe features (PDF Priority: M5+M15+H1)...")
        
        try:
            # M15 timeframe features (confirmation signals)
            m15_window = 3  # 3 x M5 = M15
            df['m15_close'] = df['close'].rolling(m15_window).mean()
            df['m15_ema_8'] = df['m15_close'].ewm(span=8).mean()
            df['m15_ema_21'] = df['m15_close'].ewm(span=21).mean()
            df['m15_trend'] = (df['m15_ema_8'] > df['m15_ema_21']).astype(int)
            
            # M15 RSI for confirmation
            df['m15_rsi'] = self.calculate_rsi(df['m15_close'], 14)
            df['m15_rsi_overbought'] = (df['m15_rsi'] > 70).astype(int)
            df['m15_rsi_oversold'] = (df['m15_rsi'] < 30).astype(int)
            
            # M15 momentum
            df['m15_momentum'] = df['m15_close'] / df['m15_close'].shift(m15_window) - 1
            df['m15_volatility'] = df['returns'].rolling(m15_window).std()
            
            # H1 timeframe features (trend direction)
            h1_window = 12  # 12 x M5 = H1
            df['h1_close'] = df['close'].rolling(h1_window).mean()
            df['h1_ema_8'] = df['h1_close'].ewm(span=8).mean()
            df['h1_ema_21'] = df['h1_close'].ewm(span=21).mean()
            df['h1_trend'] = (df['h1_ema_8'] > df['h1_ema_21']).astype(int)
            
            # H1 momentum and strength
            df['h1_momentum'] = df['h1_close'] / df['h1_close'].shift(h1_window) - 1
            df['h1_atr'] = self.calculate_atr(df, h1_window)
            df['h1_volatility'] = df['returns'].rolling(h1_window).std()
            
            # Cross-timeframe alignment features (PDF key recommendation)
            df['trend_alignment_m5_m15'] = (df['ema_8'] > df['ema_21']) & (df['m15_ema_8'] > df['m15_ema_21'])
            df['trend_alignment_all'] = df['trend_alignment_m5_m15'] & (df['h1_ema_8'] > df['h1_ema_21'])
            df['trend_strength'] = (
                df['trend_alignment_m5_m15'].astype(int) + 
                df['trend_alignment_all'].astype(int)
            ) / 2
            
            # Multi-timeframe RSI analysis
            df['rsi_divergence_m15'] = df['rsi_14'] - df['m15_rsi']
            df['rsi_convergence'] = (abs(df['rsi_divergence_m15']) < 10).astype(int)
            df['rsi_multi_tf_signal'] = (
                (df['rsi_14'] > 50).astype(int) + 
                (df['m15_rsi'] > 50).astype(int)
            ) / 2
            
            # Multi-timeframe momentum analysis
            df['momentum_alignment'] = (
                (df['momentum_5'] > 0).astype(int) + 
                (df['m15_momentum'] > 0).astype(int) + 
                (df['h1_momentum'] > 0).astype(int)
            ) / 3
            
            # Volatility regime across timeframes
            df['volatility_regime'] = (
                df['volatility_8'] > df['volatility_8'].rolling(20).mean()
            ).astype(int)
            df['multi_tf_volatility'] = (df['volatility_8'] + df['m15_volatility'] + df['h1_volatility']) / 3
            
            # Find available SMA for trend alignment
            available_sma_periods = []
            for period in self.pdf_indicators['sma_periods']:
                sma_col = f'sma_{period}'
                if sma_col in df.columns:
                    available_sma_periods.append(period)
            
            if available_sma_periods:
                trend_sma_period = available_sma_periods[0]
                sma_col = f'sma_{trend_sma_period}'
                print(f"   📈 Using {sma_col} for trend alignment")
                
                # Enhanced trend alignment with SMA
                price_above_sma = (df['close'] > df[sma_col]).astype(int)
                df['trend_alignment_final'] = (
                    price_above_sma + 
                    df['h1_trend'] + 
                    df['trend_alignment_all'].astype(int)
                ) / 3
            else:
                print(f"   ⚠️ No SMA columns available for trend alignment")
                df['trend_alignment_final'] = df['h1_trend']
            
            # Create primary signals for backward compatibility
            df['primary_rsi'] = df['rsi_14'] * 0.6 + df['m15_rsi'] * 0.4  # Weighted combination
            df['primary_macd'] = df['macd']
            df['trend_alignment'] = df['trend_alignment_final']
            
            print(f"   ✅ Multi-timeframe features created (M5+M15+H1 full integration)")
            
        except Exception as e:
            print(f"   ❌ Multi-timeframe feature creation failed: {e}")
            # Create fallback features
            df['primary_rsi'] = 50.0
            df['primary_macd'] = 0.0
            df['h1_trend'] = 0.5
            df['trend_alignment'] = 0.5
            df['trend_strength'] = 0.5
            df['momentum_alignment'] = 0.5
        
        return df
    
    def create_pair_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced pair-specific features based on PDF recommendations."""
        
        print(f"   🛠️ Creating pair-specific features for {self.pair_name}")
        
        # Start with enhanced technical indicators
        df = self.create_enhanced_technical_indicators(df)
        
        # Pair-specific enhancements
        if self.pair_name == 'EUR_USD':
            # ECB/Fed policy divergence features (PDF recommendation)
            df['eur_usd_momentum_1'] = df['close'] / df['close'].shift(1) - 1
            df['eur_usd_momentum_5'] = df['close'] / df['close'].shift(5) - 1
            df['eur_usd_policy_signal'] = df['ema_8'] / df['ema_21'] - 1
            df['eur_usd_volatility_regime'] = (df['volatility_20'] > df['volatility_20'].rolling(50).mean()).astype(int)
            df['eur_usd_range_ratio'] = (df['high'] - df['low']) / df['close']
            df['eur_usd_session_strength'] = abs(df['close'] - df['open']) / df['atr_14']
            
        elif self.pair_name == 'GBP_USD':
            # Brexit sentiment and BoE policy features (PDF recommendation)
            df['gbp_usd_volatility_ultra'] = df['returns'].rolling(5).std()
            df['gbp_usd_momentum_3'] = df['close'] / df['close'].shift(3) - 1
            df['gbp_usd_gap_ratio'] = abs(df['open'] - df['close'].shift(1)) / df['close'].shift(1)
            df['gbp_usd_noise_ratio'] = df['volatility_12'] / df['volatility_20']
            df['gbp_usd_breakout_signal'] = (df['close'] > df['bb_upper']).astype(int) - (df['close'] < df['bb_lower']).astype(int)
            df['gbp_usd_trend_persistence'] = df['trend_strength'].rolling(5).mean()
            
        elif self.pair_name == 'USD_JPY':
            # Risk-on/risk-off sentiment features (PDF recommendation)
            df['usd_jpy_carry_signal'] = df['close'].rolling(10).mean() / df['close'].rolling(50).mean()
            df['usd_jpy_momentum_1'] = df['close'] / df['close'].shift(1) - 1
            df['usd_jpy_momentum_10'] = df['close'] / df['close'].shift(10) - 1
            df['usd_jpy_risk_signal'] = df['rsi_14'] / 50 - 1  # Normalized RSI as risk proxy
            df['usd_jpy_yield_proxy'] = df['ema_21'].pct_change(21)  # 21-day yield change proxy
            df['usd_jpy_volatility_regime'] = (df['atr_14'] > df['atr_14'].rolling(30).mean()).astype(int)
            
        elif self.pair_name == 'XAU_USD':
            # VIX correlation and inflation hedge features (PDF recommendation)
            df['gold_intraday_range'] = (df['high'] - df['low']) / df['open']
            df['gold_gap_size'] = abs(df['open'] - df['close'].shift(1)) / df['close'].shift(1)
            df['gold_momentum_ultra'] = df['close'] / df['close'].shift(2) - 1
            df['gold_volatility_regime'] = (df['returns'].rolling(10).std() > df['returns'].rolling(50).std()).astype(int)
            df['gold_safe_haven_signal'] = (df['volatility_8'] > df['volatility_8'].rolling(20).quantile(0.8)).astype(int)
            df['gold_inflation_proxy'] = df['momentum_20']  # 20-period momentum as inflation hedge proxy
            df['gold_dollar_strength'] = -df['roc_10']  # Inverse correlation proxy
            
        elif self.pair_name == 'AUD_USD':
            # Commodity correlation and China sentiment features (PDF recommendation)
            df['aud_usd_commodity_proxy'] = df['volatility_16'].rolling(20).corr(df['close'].pct_change())
            df['aud_usd_momentum_5'] = df['close'] / df['close'].shift(5) - 1
            df['aud_usd_range_expansion'] = (df['true_range'] > df['atr_14']).astype(int)
            df['aud_usd_risk_appetite'] = df['bb_position']  # Risk appetite proxy
            df['aud_usd_china_proxy'] = df['momentum_12']  # China sentiment proxy
            df['aud_usd_commodity_signal'] = (df['rsi_14'] > 50).astype(int)
        
        # Add multi-timeframe features (PDF Priority)
        df = self.create_multi_timeframe_features(df)
        
        # Session-based features (enhanced)
        if hasattr(df.index, 'hour'):
            df['hour'] = df.index.hour
        else:
            df['hour'] = 0
        
        df['london_session'] = ((df['hour'] >= 8) & (df['hour'] <= 16)).astype(int)
        df['ny_session'] = ((df['hour'] >= 13) & (df['hour'] <= 21)).astype(int)
        df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] <= 8)).astype(int)
        df['session_overlap'] = (df['london_session'] & df['ny_session']).astype(int)
        df['session_volatility'] = df['session_overlap'] * df['volatility_8']
        
        # Three-class specific features
        if self.target_mode == 'three_class':
            # Enhanced trend detection for three-class
            df['trend_score'] = (
                (df['ema_8'] > df['ema_21']).astype(int) +
                (df['rsi_14'] > 50).astype(int) +
                (df['macd'] > df['macd_signal']).astype(int) +
                (df['bb_position'] > 0.5).astype(int) +
                df['trend_alignment_all'].astype(int)
            ) / 5
            
            # Flat market detection (enhanced)
            df['flat_signal'] = (
                (abs(df['momentum_5']) < df['volatility_8']) &
                (df['bb_width'] < df['bb_width'].rolling(20).mean()) &
                (abs(df['rsi_14'] - 50) < 15) &
                (abs(df['macd_histogram']) < df['atr_14'] * 0.1)
            ).astype(int)
            
            # Directional strength
            df['long_signal_strength'] = (
                (df['trend_score'] > 0.6) &
                (df['momentum_alignment'] > 0.6) &
                (df['rsi_14'] < 80)
            ).astype(int)
            
            df['short_signal_strength'] = (
                (df['trend_score'] < 0.4) &
                (df['momentum_alignment'] < 0.4) &
                (df['rsi_14'] > 20)
            ).astype(int)
        
        print(f"   📊 Created {len(df.columns)} total features")
        
        # Clean data
        df = df.replace([np.inf, -np.inf], np.nan)
        
        return df
    
    def apply_pdf_normalization(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply PDF-recommended z-score normalization for transformers."""
        
        print(f"   📏 Applying z-score normalization (PDF optimization)...")
        
        # Get numeric columns only
        numeric_columns = features.select_dtypes(include=[np.number]).columns
        
        # Apply z-score normalization (PDF: better than min-max for transformers)
        normalized_features = features.copy()
        
        for col in numeric_columns:
            mean_val = features[col].mean()
            std_val = features[col].std()
            if std_val > 1e-8:  # Avoid division by zero
                normalized_features[col] = (features[col] - mean_val) / std_val
            else:
                normalized_features[col] = 0.0
        
        return normalized_features
    
    def prepare_pair_data(self, df: pd.DataFrame, horizon: int = 64, k_pips: Optional[float] = None, 
                         dynamic_k_aggressive: bool = True, seq_len: Optional[int] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data with enhanced features and PDF optimizations."""
        
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
        
        # Create enhanced features (50+ target)
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
        
        # Feature selection - prioritize most important features
        core_features = []
        
        # PDF Priority features (guaranteed inclusion)
        pdf_priority_features = [
            'close', 'returns', 'log_returns',  # Price features (18.7% importance)
            'rsi_14', 'rsi_21',  # RSI features (15.5% importance)
            'ema_8', 'ema_21', 'ema_55',  # EMA features (18.3% importance)
            'bb_position', 'bb_width',  # Bollinger Bands (14.7% importance)
            'volume_ratio', 'volume_spike',  # Volume features (14.6% importance)
            'macd', 'macd_signal', 'macd_histogram',  # MACD
            'atr_14', 'atr_32',  # ATR
            'momentum_5', 'momentum_12', 'momentum_20',  # Momentum
            'volatility_8', 'volatility_16', 'volatility_20',  # Volatility
        ]
        
        for feature in pdf_priority_features:
            if feature in featured_data.columns:
                core_features.append(feature)
        
        # Multi-timeframe features (PDF Priority #1)
        multi_tf_features = [
            'primary_rsi', 'primary_macd', 'h1_trend', 'trend_alignment',
            'm15_trend', 'm15_rsi', 'h1_momentum', 'trend_strength',
            'momentum_alignment', 'rsi_multi_tf_signal', 'trend_alignment_final'
        ]
        for feature in multi_tf_features:
            if feature in featured_data.columns:
                core_features.append(feature)
        
        # Technical indicator derivatives
        technical_features = [
            'price_vs_ema_8', 'price_vs_ema_21', 'price_vs_sma_8', 'price_vs_sma_21',
            'ema_slope_8', 'ema_slope_21', 'rsi_14_slope', 'rsi_21_slope',
            'macd_histogram_slope', 'macd_cross', 'bb_squeeze', 'bb_break_upper',
            'ema_cross_8_21', 'ma_alignment', 'rsi_divergence_m15'
        ]
        for feature in technical_features:
            if feature in featured_data.columns:
                core_features.append(feature)
        
        # Price position and support/resistance
        position_features = []
        for window in [16, 32, 64, 128]:
            for feature_type in ['price_position', 'high_low_ratio', 'support_resistance']:
                feature_name = f'{feature_type}_{window}'
                if feature_name in featured_data.columns:
                    position_features.append(feature_name)
        core_features.extend(position_features)
        
        # Pair-specific features
        pair_features = []
        if self.pair_name == 'EUR_USD':
            pair_features = ['eur_usd_momentum_1', 'eur_usd_momentum_5', 'eur_usd_policy_signal', 
                           'eur_usd_volatility_regime', 'eur_usd_session_strength']
        elif self.pair_name == 'GBP_USD':
            pair_features = ['gbp_usd_volatility_ultra', 'gbp_usd_momentum_3', 'gbp_usd_gap_ratio',
                           'gbp_usd_breakout_signal', 'gbp_usd_trend_persistence']
        elif self.pair_name == 'USD_JPY':
            pair_features = ['usd_jpy_carry_signal', 'usd_jpy_momentum_1', 'usd_jpy_risk_signal',
                           'usd_jpy_yield_proxy', 'usd_jpy_volatility_regime']
        elif self.pair_name == 'XAU_USD':
            pair_features = ['gold_intraday_range', 'gold_momentum_ultra', 'gold_volatility_regime',
                           'gold_safe_haven_signal', 'gold_inflation_proxy']
        elif self.pair_name == 'AUD_USD':
            pair_features = ['aud_usd_momentum_5', 'aud_usd_range_expansion', 'aud_usd_risk_appetite',
                           'aud_usd_china_proxy', 'aud_usd_commodity_signal']
        
        for feature in pair_features:
            if feature in featured_data.columns:
                core_features.append(feature)
        
        # Session and microstructure features
        session_features = ['london_session', 'ny_session', 'session_overlap', 'session_volatility']
        for feature in session_features:
            if feature in featured_data.columns:
                core_features.append(feature)
        
        # Three-class specific features
        if self.target_mode == 'three_class':
            three_class_features = ['trend_score', 'flat_signal', 'long_signal_strength', 'short_signal_strength']
            for feature in three_class_features:
                if feature in featured_data.columns:
                    core_features.append(feature)
        
        # Additional volatility and momentum features
        additional_features = [
            'price_acceleration', 'price_momentum', 'true_range', 'hl_ratio', 'oc_ratio',
            'gap_up', 'gap_down', 'momentum_divergence_5_20', 'momentum_acceleration',
            'volatility_regime', 'multi_tf_volatility', 'rsi_price_divergence'
        ]
        for feature in additional_features:
            if feature in featured_data.columns:
                core_features.append(feature)
        
        # Remove duplicates while preserving order
        core_features = list(dict.fromkeys(core_features))
        
        # Filter existing features
        available_features = [f for f in core_features if f in featured_data.columns]
        features = featured_data[available_features].copy()
        
        feature_count = len(available_features)
        print(f"   🎯 {self.pair_name} features: {feature_count}")
        print(f"   📊 Target distribution: {label_stats['class_counts']}")
        
        # Feature count validation
        if feature_count < 50:
            print(f"   ⚠️ WARNING: Only {feature_count} features (PDF target: 50+)")
            # Add remaining features if available
            remaining_features = [col for col in featured_data.columns 
                                if col not in available_features and 
                                col not in ['open', 'high', 'low', 'close', 'volume']]
            additional_needed = min(50 - feature_count, len(remaining_features))
            if additional_needed > 0:
                features = pd.concat([features, featured_data[remaining_features[:additional_needed]]], axis=1)
                feature_count = len(features.columns)
                print(f"   ✅ Added {additional_needed} features, total: {feature_count}")
        else:
            print(f"   🎯 SUCCESS: {feature_count} features meets PDF requirements!")
        
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
        
        # Remove outliers using IQR method (PDF robust scaling)
        for col in features.select_dtypes(include=[np.number]).columns:
            Q1 = features[col].quantile(0.25)
            Q3 = features[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            features[col] = features[col].clip(lower_bound, upper_bound)
        
        # Final valid data
        valid_mask = ~(features.isna().any(axis=1) | target.isna())
        features_clean = features[valid_mask]
        target_clean = target[valid_mask]
        
        print(f"   ✅ {self.pair_name} clean data: {len(features_clean):,} records")
        
        self.feature_columns = features_clean.columns.tolist()
        return features_clean, target_clean
    
    def create_sequences(self, features: pd.DataFrame, target: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences with enhanced preprocessing and PDF optimizations."""
        
        print(f"   📊 Creating sequences with PDF optimizations...")
        
        # Convert to numpy arrays for processing
        print(f"   📊 Raw data types: features={type(features)}, target={type(target)}")
        print(f"   🔄 Converting features DataFrame to NumPy...")
        features_array = features.values.astype(np.float32)
        
        print(f"   🔄 Converting target DataFrame to NumPy...")
        if isinstance(target, pd.Series):
            target_array = target.values
        else:
            target_array = target
        
        print(f"   📊 NumPy shapes: features={features_array.shape}, target={target_array.shape}")
        
        # Handle target format
        if target_array.ndim > 1:
            print(f"   📊 Multi-dimensional target detected, using first column")
            target_array = target_array[:, 0]
        else:
            print(f"   🎯 1D target used directly")
        
        # Apply PDF-recommended z-score normalization
        features_normalized = self.apply_pdf_normalization(pd.DataFrame(features_array, columns=self.feature_columns))
        features_array = features_normalized.values.astype(np.float32)
        
        # Apply SMOTE if needed (forced for three-class mode)
        if self.use_smote:
            if self.target_mode == 'three_class':
                features_array, target_array = self.force_smote_three_class(features_array, target_array)
                print(f"   ✅ Three-class SMOTE applied: {len(features_array):,} samples")
            else:
                # Binary mode SMOTE
                unique_classes, class_counts = np.unique(target_array, return_counts=True)
                if len(unique_classes) > 1:
                    minority_ratio = class_counts.min() / class_counts.sum()
                    print(f"   📊 Minority ratio: {minority_ratio:.3f}")
                    
                    if minority_ratio < 0.15:
                        try:
                            smote = SMOTE(random_state=42, k_neighbors=min(5, class_counts.min() - 1))
                            features_array, target_array = smote.fit_resample(features_array, target_array)
                            print(f"   ✅ Binary SMOTE applied: {len(features_array):,} samples")
                        except Exception as e:
                            print(f"   ❌ Binary SMOTE failed: {e}")
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(self.sequence_length, len(features_array)):
            seq = features_array[i-self.sequence_length:i]
            tgt = target_array[i]
            
            sequences.append(seq)
            targets.append(tgt)
        
        X = np.array(sequences, dtype=np.float32)
        y = np.array(targets)
        
        print(f"   📊 Final tensor shapes: X={X.shape}, y={y.shape}")
        print(f"   ✅ {self.pair_name} sequences: {len(sequences):,}")
        
        # Final distribution logging
        if self.target_mode == 'binary':
            unique, counts = np.unique(y, return_counts=True)
            class_dist = dict(zip(unique.astype(int), counts))
            print(f"   🎯 Final binary distribution: {class_dist}")
            if len(unique) > 1:
                minority_ratio = counts.min() / len(y)
                print(f"   🎯 Final minority ratio: {minority_ratio:.3f}")
        else:
            unique, counts = np.unique(y, return_counts=True)
            class_names = {0: 'Long', 1: 'Flat', 2: 'Short'}
            class_breakdown = {class_names.get(int(cls), int(cls)): count for cls, count in zip(unique, counts)}
            print(f"   🎯 Final class distribution: {class_breakdown}")
        
        return X, y
    
    def force_smote_three_class(self, features_scaled: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Enhanced SMOTE for three-class mode with better handling."""
        try:
            unique_classes, class_counts = np.unique(target, return_counts=True)
            print(f"   📊 Pre-SMOTE distribution: {dict(zip(unique_classes, class_counts))}")
            
            if len(unique_classes) < 2:
                print(f"   ⚠️ Single class detected, skipping SMOTE")
                return features_scaled, target
            
            # Use minimum neighbors available
            min_samples = class_counts.min()
            k_neighbors = max(1, min(5, min_samples - 1))
            
            if k_neighbors < 1:
                print(f"   ⚠️ Insufficient samples for SMOTE")
                return features_scaled, target
                
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            features_resampled, target_resampled = smote.fit_resample(features_scaled, target)
            
            unique_after, counts_after = np.unique(target_resampled, return_counts=True)
            print(f"   📊 Post-SMOTE distribution: {dict(zip(unique_after, counts_after))}")
            print(f"   ✅ SMOTE successful: {len(features_resampled):,} samples")
            
            return features_resampled, target_resampled
            
        except Exception as e:
            print(f"   ❌ SMOTE failed: {e}")
            return features_scaled, target
    
    def split_data(self, X: np.ndarray, y: np.ndarray, train_split: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data maintaining temporal order."""
        
        split_index = int(len(X) * train_split)
        
        X_train = X[:split_index]
        y_train = y[:split_index]
        X_val = X[split_index:]
        y_val = y[split_index:]
        
        print(f"   📊 Data split: Train={len(X_train):,}, Val={len(X_val):,}")
        
        return X_train, X_val, y_train, y_val


def create_preprocessor(pair_name: str, config: dict, target_mode: str = None) -> PairSpecificPreprocessor:
    """Factory function to create enhanced preprocessor from configuration."""
    
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


# Quick integration test function
def test_enhanced_features():
    """Test function to validate enhanced features work correctly."""
    
    # Create sample data
    import pandas as pd
    np.random.seed(42)
    
    dates = pd.date_range('2024-01-01', periods=1000, freq='5T')
    sample_data = pd.DataFrame({
        'open': 1.1000 + np.random.randn(1000) * 0.001,
        'high': 1.1005 + np.random.randn(1000) * 0.001,
        'low': 0.9995 + np.random.randn(1000) * 0.001,
        'close': 1.1002 + np.random.randn(1000) * 0.001,
        'volume': np.random.randint(1000, 10000, 1000)
    }, index=dates)
    
    # Test preprocessor
    preprocessor = PairSpecificPreprocessor(
        pair_name='EUR_USD',
        target_mode='binary',
        sequence_length=64
    )
    
    # Process data
    features, target = preprocessor.prepare_pair_data(sample_data, horizon=64)
    X, y = preprocessor.create_sequences(features, target)
    
    print(f"Test Results:")
    print(f"✅ Features created: {len(features.columns)}")
    print(f"✅ Sequences shape: {X.shape}")
    print(f"✅ Target shape: {y.shape}")
    
    return len(features.columns) >= 50


__all__ = ['PairSpecificPreprocessor', 'create_preprocessor', 'test_enhanced_features']