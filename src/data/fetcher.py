# src/data/fetcher.py
"""Enhanced data fetching module with FIXED pagination support."""

import pandas as pd
import numpy as np
import requests
import time
import pickle
import os
from typing import Dict, Optional, List, Tuple
from datetime import datetime
from pathlib import Path

class MultiPairOANDAFetcher:
    """Enhanced OANDA API data fetcher with FIXED pagination and time alignment."""
    
    def __init__(self, api_key: str, account_id: str, environment: str = 'practice', 
                 cache_enabled: bool = False):
        self.api_key = api_key
        self.account_id = account_id
        self.environment = environment
        self.cache_enabled = cache_enabled
        
        # Create cache directory
        if self.cache_enabled:
            self.cache_dir = Path('data/raw')
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 Cache directory: {self.cache_dir}")
        
        if environment == 'practice':
            self.api_url = "https://api-fxpractice.oanda.com"
        else:
            self.api_url = "https://api-fxtrade.oanda.com"
        
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        # Major pairs + Commodities configuration
        self.pairs_config = {
            'EUR_USD': {
                'name': 'EUR/USD',
                'pip_value': 0.0001,
                'spread_typical': 0.00015,
                'volatility_factor': 1.0,
                'session_preference': ['London', 'NY'],
                'type': 'currency'
            },
            'GBP_USD': {
                'name': 'GBP/USD',
                'pip_value': 0.0001,
                'spread_typical': 0.00020,
                'volatility_factor': 1.3,
                'session_preference': ['London', 'NY'],
                'type': 'currency'
            },
            'USD_JPY': {
                'name': 'USD/JPY',
                'pip_value': 0.01,
                'spread_typical': 0.015,
                'volatility_factor': 1.1,
                'session_preference': ['Tokyo', 'London'],
                'type': 'currency'
            },
            'AUD_USD': {
                'name': 'AUD/USD',
                'pip_value': 0.0001,
                'spread_typical': 0.00025,
                'volatility_factor': 1.2,
                'session_preference': ['Sydney', 'London'],
                'type': 'currency'
            },
            'XAU_USD': {
                'name': 'GOLD/USD',
                'pip_value': 0.01,
                'spread_typical': 0.35,
                'volatility_factor': 2.5,
                'session_preference': ['London', 'NY'],
                'type': 'commodity'
            },
            'USD_CHF': {
                'name': 'USD/CHF',
                'pip_value': 0.0001,
                'spread_typical': 0.00020,
                'volatility_factor': 1.0,
                'session_preference': ['London', 'NY'],
                'type': 'currency'
            },
            'USD_CAD': {
                'name': 'USD/CAD',
                'pip_value': 0.0001,
                'spread_typical': 0.00025,
                'volatility_factor': 1.1,
                'session_preference': ['NY', 'London'],
                'type': 'currency'
            }
        }
    
    def _get_cache_path(self, instrument: str, granularity: str) -> Path:
        """Get cache file path for instrument and granularity."""
        return self.cache_dir / f"{instrument}_{granularity}.pkl"
    
    def _load_from_cache(self, instrument: str, granularity: str, 
                        lookback_candles: int) -> Optional[pd.DataFrame]:
        """Load data from cache if available and valid."""
        if not self.cache_enabled:
            return None
            
        cache_path = self._get_cache_path(instrument, granularity)
        
        if not cache_path.exists():
            return None
        
        try:
            # Check cache age with EXTENDED granularity-specific limits
            cache_age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
            
            # Extended cache limits
            cache_limits = {
                'M1': 12,   # 12 hours for M1
                'M5': 48,   # 48 hours for M5 (EXTENDED)  
                'M15': 48,  # 48 hours for M15 (EXTENDED)
                'H1': 72,   # 72 hours for H1
                'H4': 168,  # 1 week for H4
                'D': 504    # 3 weeks for daily
            }
            
            max_age = cache_limits.get(granularity, 48)
            
            if cache_age_hours > max_age:
                print(f"   💾 Cache expired for {instrument}_{granularity} ({cache_age_hours:.1f}h > {max_age}h)")
                # Auto-cleanup expired cache
                cache_path.unlink(missing_ok=True)
                return None
            
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Validate cache has enough data
            if len(cached_data) >= lookback_candles * 0.7:  # Allow 30% tolerance
                print(f"   💾 Cache HIT: {instrument}_{granularity} ({len(cached_data)} candles)")
                return cached_data.tail(lookback_candles) if len(cached_data) > lookback_candles else cached_data
            else:
                print(f"   💾 Cache insufficient: {instrument}_{granularity} ({len(cached_data)} < {lookback_candles})")
                return None
                
        except Exception as e:
            print(f"   ❌ Cache load error for {instrument}_{granularity}: {e}")
            return None
    
    def _save_to_cache(self, instrument: str, granularity: str, data: pd.DataFrame) -> None:
        """Save data to cache."""
        if not self.cache_enabled or data is None or len(data) == 0:
            return
        
        try:
            cache_path = self._get_cache_path(instrument, granularity)
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"   💾 Cached {len(data)} candles to {cache_path}")
        except Exception as e:
            print(f"   ⚠️ Cache save error: {e}")
    
    def fetch_pair_data(self, instrument: str, granularity: str = 'M15', 
                       count: int = 5000, lookback_candles: int = 50000,
                       target_candles: int = None) -> Optional[pd.DataFrame]:
        """FIXED fetch with proper pagination loop - REALISTIC limits."""
        
        # Use realistic target (not 180k which is too much)
        if target_candles is not None:
            actual_target = min(target_candles, 50000)  # Max 50k candles (~6 months M15)
        else:
            actual_target = min(lookback_candles, 50000)
        
        # Try cache first
        cached_data = self._load_from_cache(instrument, granularity, actual_target)
        if cached_data is not None:
            return cached_data
        
        print(f"   🌐 Fetching {instrument} from API (target: {actual_target:,} candles)...")
        
        url = f"{self.api_url}/v3/instruments/{instrument}/candles"
        
        # FIXED pagination loop
        all_data = []
        fetched = 0
        page_count = 0
        last_timestamp = None
        max_pages = 15  # Safety limit (15 * 5000 = 75k max)
        
        while fetched < actual_target and page_count < max_pages:
            page_count += 1
            chunk_size = min(5000, actual_target - fetched)
            
            params = {
                'granularity': granularity,
                'price': 'MBA',
                'count': chunk_size
            }
            
            # FIXED: Go backwards in time properly
            if last_timestamp:
                params['to'] = last_timestamp
                # Keep count with 'to' parameter
            
            try:
                response = requests.get(url, headers=self.headers, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    candles = data.get('candles', [])
                    
                    if not candles:
                        print(f"   ⚠️ No more candles available at page {page_count}")
                        break
                    
                    # Filter complete candles and avoid duplicates
                    existing_times = {c['time'] for c in all_data}
                    new_candles = [c for c in candles if c['complete'] and c['time'] not in existing_times]
                    
                    if not new_candles:
                        print(f"   ⚠️ No new candles found at page {page_count}")
                        break
                    
                    # Sort by time (oldest first)
                    new_candles.sort(key=lambda x: x['time'])
                    all_data.extend(new_candles)
                    fetched = len(all_data)
                    
                    # FIXED: Set last_timestamp to oldest candle for next iteration
                    last_timestamp = new_candles[0]['time']
                    
                    print(f"   📈 Page {page_count}: +{len(new_candles):,} candles, total: {fetched:,}/{actual_target:,}")
                    
                    # Check if we've reached our target
                    if fetched >= actual_target:
                        print(f"   🎯 Target reached: {fetched:,} candles")
                        break
                    
                    # Rate limiting
                    time.sleep(0.3)
                    
                elif response.status_code == 429:  # Rate limit
                    print(f"   ⏰ Rate limited, waiting 60 seconds...")
                    time.sleep(60)
                    continue
                else:
                    print(f"❌ {instrument} API error: {response.status_code}")
                    break
                    
            except Exception as e:
                print(f"❌ {instrument} fetch error: {e}")
                break
        
        if not all_data:
            return None
        
        # Sort all data by time (oldest first)
        all_data.sort(key=lambda x: x['time'])
        
        # Convert to DataFrame
        df = self._candles_to_dataframe(all_data)
        
        # Save to cache
        self._save_to_cache(instrument, granularity, df)
        
        # Log data info
        if len(df) > 0:
            days_covered = (df.index[-1] - df.index[0]).days
            print(f"   📅 {instrument}: {days_covered} days covered ({len(df):,} candles)")
        
        return df
    
    def _candles_to_dataframe(self, candles: List[Dict]) -> pd.DataFrame:
        """Convert OANDA candles to DataFrame."""
        ohlcv_data = []
        for candle in candles:
            # Calculate spread or use typical spread
            if 'ask' in candle and 'bid' in candle:
                spread = float(candle['ask']['c']) - float(candle['bid']['c'])
            else:
                # Use mid prices if ask/bid not available
                spread = 0.00015  # Default spread
            
            ohlcv_data.append({
                'datetime': pd.to_datetime(candle['time']),
                'open': float(candle['mid']['o']),
                'high': float(candle['mid']['h']),
                'low': float(candle['mid']['l']),
                'close': float(candle['mid']['c']),
                'volume': int(candle['volume']),
                'spread': spread
            })
        
        df = pd.DataFrame(ohlcv_data)
        df = df.sort_values('datetime')  # Ensure chronological order
        df.set_index('datetime', inplace=True)
        df.index = df.index.tz_localize(None)
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='last')]
        
        return df
    
    def fetch_multi_timeframe_data(self, instrument: str, signal_tf: str = 'M15', 
                                  exec_tf: str = 'M5', lookback_candles: int = 50000) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """FIXED multi-timeframe data fetching with proper time alignment."""
        
        print(f"   📊 Multi-timeframe fetch: {signal_tf} + {exec_tf}")
        
        # STEP 1: Fetch execution timeframe first (usually covers less time)
        exec_target = min(lookback_candles, 30000)  # M5: ~100 days
        exec_data = self.fetch_pair_data(instrument, exec_tf, target_candles=exec_target)
        
        if exec_data is None or len(exec_data) < 1000:
            print(f"   ❌ Insufficient execution data")
            return None, None
        
        # STEP 2: Get time range from execution data
        exec_start = exec_data.index.min()
        exec_end = exec_data.index.max()
        
        print(f"   ⚡ Exec range: {exec_start.strftime('%Y-%m-%d')} to {exec_end.strftime('%Y-%m-%d')}")
        
        # STEP 3: Fetch signal timeframe for same period
        signal_target = min(lookback_candles, 15000)  # M15: ~150 days
        signal_data = self.fetch_pair_data(instrument, signal_tf, target_candles=signal_target)
        
        if signal_data is None or len(signal_data) < 500:
            print(f"   ❌ Insufficient signal data")
            return None, None
        
        # STEP 4: Align time ranges
        # Find common time range
        common_start = max(signal_data.index.min(), exec_data.index.min())
        common_end = min(signal_data.index.max(), exec_data.index.max())
        
        print(f"   🎯 Common range: {common_start.strftime('%Y-%m-%d')} to {common_end.strftime('%Y-%m-%d')}")
        
        # Filter to common range
        signal_aligned = signal_data.loc[common_start:common_end]
        exec_aligned = exec_data.loc[common_start:common_end]
        
        print(f"   📊 Aligned: {signal_tf}={len(signal_aligned)}, {exec_tf}={len(exec_aligned)}")
        
        # Ensure minimum data requirements
        if len(signal_aligned) < 500 or len(exec_aligned) < 1000:
            print(f"   ❌ Insufficient aligned data")
            return None, None
        
        return signal_aligned, exec_aligned
    
    def fetch_all_pairs(self, granularity: str = 'M15', count: int = 3000, 
                       lookback_candles: int = 50000, 
                       pairs: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Fetch data for all configured pairs with REALISTIC limits."""
        
        # Use realistic limits
        lookback_candles = min(lookback_candles, 50000)
        
        cache_status = "enabled" if self.cache_enabled else "disabled"
        print(f"📊 Multi-pair data fetching... ({granularity}, max {lookback_candles:,} candles, cache {cache_status})")
        
        pair_data = {}
        pairs_to_fetch = pairs or list(self.pairs_config.keys())
        
        for instrument in pairs_to_fetch:
            if instrument not in self.pairs_config:
                print(f"⚠️ Unknown instrument: {instrument}")
                continue
                
            config = self.pairs_config[instrument]
            print(f"   📈 Fetching {config['name']}...")
            
            data = self.fetch_pair_data(instrument, granularity, count, lookback_candles)
            
            if data is not None and len(data) > 1000:
                pair_data[instrument] = data
                print(f"   ✅ {config['name']}: {len(data):,} records")
                time.sleep(0.2)  # Rate limiting between pairs
            else:
                print(f"   ❌ {config['name']}: Insufficient data")
        
        print(f"\n✅ {len(pair_data)} pairs fetched successfully")
        return pair_data
    
    def clear_cache(self, instrument: str = None, granularity: str = None) -> None:
        """Clear cache files."""
        if not self.cache_enabled:
            print("Cache is disabled")
            return
        
        if instrument and granularity:
            cache_path = self._get_cache_path(instrument, granularity)
            if cache_path.exists():
                cache_path.unlink()
                print(f"🗑️ Cleared cache: {cache_path}")
        else:
            # Clear all cache files
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
                print(f"🗑️ Cleared cache: {cache_file}")
    
    def get_cache_info(self) -> Dict[str, Dict]:
        """Get information about cached files."""
        if not self.cache_enabled:
            return {"cache_enabled": False}
        
        cache_info = {"cache_enabled": True, "files": {}}
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                
                cache_info["files"][cache_file.name] = {
                    "size": len(data),
                    "date_range": f"{data.index[0]} to {data.index[-1]}",
                    "file_size_mb": cache_file.stat().st_size / (1024 * 1024),
                    "last_modified": datetime.fromtimestamp(cache_file.stat().st_mtime)
                }
            except Exception as e:
                cache_info["files"][cache_file.name] = {"error": str(e)}
        
        return cache_info
    
    def get_pair_config(self, pair_name: str) -> Dict:
        """Get configuration for a specific pair."""
        return self.pairs_config.get(pair_name, {})
    
    def get_available_pairs(self) -> List[str]:
        """Get list of available currency pairs."""
        return list(self.pairs_config.keys())

def create_fetcher(config: dict, cache_enabled: bool = False) -> MultiPairOANDAFetcher:
    """Factory function to create fetcher from configuration."""
    api_config = config.get('api', {})
    
    return MultiPairOANDAFetcher(
        api_key=api_config.get('api_key', ''),
        account_id=api_config.get('account_id', ''),
        environment=api_config.get('environment', 'practice'),
        cache_enabled=cache_enabled
    )

__all__ = ['MultiPairOANDAFetcher', 'create_fetcher']
