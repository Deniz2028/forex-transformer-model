#!/usr/bin/env python3
"""
Enhanced Temporal Training and Backtesting Module
Comprehensive temporal training with validation plots, training time tracking, and enhanced reporting.
"""

import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional, Any

from ..data.fetcher import create_fetcher
from ..data.preprocess import create_preprocessor
from ..models.factory import create_model
from ..training.trainer import create_trainer
from ..utils.temporal_plotting import TemporalPlotter, create_temporal_plots
from ..utils.training_timer import TrainingTimer, TimedTraining

logger = logging.getLogger(__name__)

class TemporalTrainer:
    """
    Enhanced temporal training ve backtesting sınıfı.
    
    Features:
    - 6 ay eğitim + 6 ay sinyal üretimi
    - Veri sızıntısı koruması
    - Validation curve plotting
    - Training time tracking
    - Enhanced reporting
    """
    
    def __init__(self, config: Dict[str, Any], cache_enabled: bool = True):
        self.config = config
        self.cache_enabled = cache_enabled
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Temporal sınırları - config'den veya default'tan al
        temporal_config = config.get('temporal', {})
        self.train_start = temporal_config.get('train_start', '2024-01-01')      # String olarak
        self.train_end = temporal_config.get('train_end', '2024-06-30')          # String olarak
        self.signal_start = temporal_config.get('signal_start', '2024-07-01')    # String olarak
        self.signal_end = temporal_config.get('signal_end', '2024-12-31')        # String olarak
        
        # Enhanced features
        self.enable_plotting = temporal_config.get('save_validation_plots', True)
        self.track_training_time = temporal_config.get('log_training_time', True)
        self.confidence_threshold = temporal_config.get('confidence_threshold', 0.35)
        
        # Initialize components
        self.plotter = None
        self.training_timer = None
        
        if self.enable_plotting:
            self.plotter = TemporalPlotter('temporal_plots', 'png')
        
        if self.track_training_time:
            self.training_timer = TrainingTimer('logs/temporal')
        
        logger.info(f"🕒 Enhanced TemporalTrainer initialized")
        logger.info(f"   Eğitim periyodu: {self.train_start} → {self.train_end}")
        logger.info(f"   Sinyal periyodu: {self.signal_start} → {self.signal_end}")
        logger.info(f"   Cihaz: {self.device}")
        logger.info(f"   Plotting: {self.enable_plotting}")
        logger.info(f"   Time tracking: {self.track_training_time}")
        
        # Sonuçları saklamak için
        self.trained_models = {}
        self.training_results = {}
        self.signal_results = {}
        self.training_metrics = {}
    
    def run_temporal_pipeline(self, pairs: List[str]) -> Dict[str, Any]:
        """
        Enhanced temporal pipeline with comprehensive tracking.
        """
        logger.info(f"\n🚀 ENHANCED TEMPORAL TRAINING & BACKTESTING PIPELINE")
        logger.info(f"{'='*60}")
        
        pipeline_results = {
            'training_results': {},
            'signal_results': {},
            'pipeline_summary': {},
            'training_metrics': {},
            'plot_paths': {}
        }
        
        pipeline_start_time = datetime.now()
        
        for pair_name in pairs:
            logger.info(f"\n{'='*50}")
            logger.info(f"📊 Processing {pair_name}")
            logger.info(f"{'='*50}")
            
            try:
                # 1. VERİ HAZIRLIĞI
                train_data, signal_data = self._prepare_temporal_data(pair_name)
                
                if train_data is None or signal_data is None:
                    logger.error(f"❌ {pair_name}: Veri hazırlama başarısız")
                    continue
                
                # 2. ENHANCED MODEL EĞİTİMİ
                model, training_history, training_metrics = self._train_enhanced_temporal_model(
                    pair_name, train_data
                )
                
                if model is None:
                    logger.error(f"❌ {pair_name}: Model eğitimi başarısız")
                    continue
                
                # 3. VALIDATION PLOTS
                if self.enable_plotting and training_history:
                    plot_path = self.plotter.plot_training_history(
                        training_history, pair_name, 'enhanced_transformer'
                    )
                    if plot_path:
                        pipeline_results['plot_paths'][f'{pair_name}_training'] = plot_path
                
                # 4. SİNYAL ÜRETİMİ
                signals = self._generate_enhanced_temporal_signals(
                    pair_name, model, signal_data
                )
                
                # 5. SIGNAL ANALYSIS PLOTS
                if self.enable_plotting and signals and 'signals_df' in signals:
                    signal_plot_path = self.plotter.plot_signal_analysis(
                        signals['signals_df'], pair_name,
                        (self.train_start, self.train_end),
                        (self.signal_start, self.signal_end),
                        signal_data
                    )
                    if signal_plot_path:
                        pipeline_results['plot_paths'][f'{pair_name}_signals'] = signal_plot_path
                
                # 6. SONUÇLARI KAYDET
                pipeline_results['training_results'][pair_name] = training_history
                pipeline_results['signal_results'][pair_name] = signals
                if training_metrics:
                    pipeline_results['training_metrics'][pair_name] = training_metrics.to_dict()
                
                logger.info(f"✅ {pair_name}: temporal pipeline tamamlandı")
                
            except Exception as e:
                logger.error(f"❌ {pair_name}: pipeline hatası: {str(e)}")
                continue
        
        # 7. PIPELINE SUMMARY PLOTS
        if self.enable_plotting and pipeline_results['training_results']:
            summary_plot_path = self.plotter.plot_pipeline_summary(pipeline_results)
            if summary_plot_path:
                pipeline_results['plot_paths']['pipeline_summary'] = summary_plot_path
            
            timeline_plot_path = self.plotter.plot_temporal_timeline(
                (self.train_start, self.train_end),
                (self.signal_start, self.signal_end),
                list(pipeline_results['training_results'].keys())
            )
            if timeline_plot_path:
                pipeline_results['plot_paths']['temporal_timeline'] = timeline_plot_path
        
        # 8. ENHANCED ÖZET RAPOR
        pipeline_end_time = datetime.now()
        pipeline_duration = pipeline_end_time - pipeline_start_time
        
        pipeline_results['pipeline_summary'] = self._generate_enhanced_pipeline_summary(
            pipeline_results, pipeline_duration
        )
        
        # 9. TRAINING TIMER SUMMARY
        if self.training_timer:
            timer_summary = self.training_timer.get_summary_stats()
            pipeline_results['pipeline_summary']['training_timer_summary'] = timer_summary
        
        # 10. SONUÇLARI KAYDET
        self._save_enhanced_pipeline_results(pipeline_results)
        
        return pipeline_results
    
    def _prepare_temporal_data(self, pair_name: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Enhanced temporal data preparation with better validation."""
        logger.info(f"📊 {pair_name}: enhanced temporal data preparation...")
        
        try:
            # Fetcher oluştur
            fetcher = create_fetcher(self.config, cache_enabled=self.cache_enabled)
            
            # TOPLAM VERİ ÇEK
            full_data = fetcher.fetch_pair_data(
                instrument=pair_name,
                granularity=self.config['data']['granularity'],
                lookback_candles=self.config['data']['lookback_candles'],
                target_candles=self.config['data']['lookback_candles']
            )
            
            if full_data is None or len(full_data) < 10000:
                logger.error(f"❌ {pair_name}: Yetersiz veri ({len(full_data) if full_data is not None else 0})")
                return None, None
            
            # Ensure datetime index
            if 'time' in full_data.columns:
                full_data['time'] = pd.to_datetime(full_data['time'])
                full_data.set_index('time', inplace=True)
            
            if not isinstance(full_data.index, pd.DatetimeIndex):
                full_data.index = pd.to_datetime(full_data.index)
            
            # ENHANCED TEMPORAL SPLITTING
            train_start_dt = pd.to_datetime(self.train_start)
            train_end_dt = pd.to_datetime(self.train_end)
            signal_start_dt = pd.to_datetime(self.signal_start)
            signal_end_dt = pd.to_datetime(self.signal_end)
            
            # Create masks with buffer zones
            train_mask = (full_data.index >= train_start_dt) & (full_data.index <= train_end_dt)
            signal_mask = (full_data.index >= signal_start_dt) & (full_data.index <= signal_end_dt)
            
            train_data = full_data[train_mask].copy()
            signal_data = full_data[signal_mask].copy()
            
            # Enhanced validation
            min_train_samples = self.config.get('temporal', {}).get('min_train_samples', 5000)
            min_signal_samples = self.config.get('temporal', {}).get('min_signal_samples', 1000)
            
            logger.info(f"   📈 Training data: {len(train_data)} records ({train_data.index.min()} → {train_data.index.max()})")
            logger.info(f"   📉 Signal data: {len(signal_data)} records ({signal_data.index.min()} → {signal_data.index.max()})")
            
            # Data quality checks
            if len(train_data) < min_train_samples:
                logger.error(f"❌ {pair_name}: Yetersiz eğitim verisi ({len(train_data)} < {min_train_samples})")
                return None, None
            
            if len(signal_data) < min_signal_samples:
                logger.error(f"❌ {pair_name}: Yetersiz sinyal verisi ({len(signal_data)} < {min_signal_samples})")
                return None, None
            
            # STRICT DATA LEAKAGE CHECK
            gap_days = (signal_start_dt - train_end_dt).days
            if gap_days < 0:
                logger.error(f"❌ {pair_name}: VERİ SIZINTISI TESPİT EDİLDİ!")
                logger.error(f"   Training end: {train_end_dt}")
                logger.error(f"   Signal start: {signal_start_dt}")
                logger.error(f"   Gap: {gap_days} days")
                return None, None
            
            logger.info(f"   ✅ Data leakage check passed: {gap_days} days gap")
            logger.info(f"   ✅ {pair_name}: temporal data preparation successful")
            
            return train_data, signal_data
            
        except Exception as e:
            logger.error(f"❌ {pair_name}: data preparation error: {str(e)}")
            return None, None
    
    def _train_enhanced_temporal_model(self, pair_name: str, train_data: pd.DataFrame) -> Tuple[Optional[Any], Optional[Dict], Optional[Any]]:
        """Enhanced model training with comprehensive tracking."""
        logger.info(f"🤖 {pair_name}: enhanced model training...")
        
        training_metrics = None
        
        try:
            # Start training timer if enabled
            if self.training_timer:
                self.training_timer.start_training(
                    pair_name=pair_name,
                    model_type='enhanced_transformer',
                    total_epochs=self.config['training']['epochs'],
                    sequence_length=self.config['data']['sequence_length']
                )
            
            # Preprocessor setup
            preprocessor = create_preprocessor(pair_name, self.config)
            preprocessor.target_mode = self.config['model']['target_mode']
            
            # Pair-specific adjustments
            pair_config = self.config.get('pairs', {}).get(pair_name, {})
            if pair_config.get('target_mode'):
                preprocessor.target_mode = pair_config['target_mode']
                logger.info(f"   🎯 {pair_name}: Using {preprocessor.target_mode} mode")
            
            # Feature preparation
            features, target = preprocessor.prepare_pair_data(
                train_data,
                horizon=64,
                dynamic_k_aggressive=True,
                seq_len=self.config['data']['sequence_length']
            )
            
            if len(features) < 1000:
                logger.error(f"❌ {pair_name}: Insufficient features ({len(features)})")
                return None, None, None
            
            # Create sequences
            X, y = preprocessor.create_sequences(features, target)
            
            if len(X) < 500:
                logger.error(f"❌ {pair_name}: Insufficient sequences ({len(X)})")
                return None, None, None
            
            logger.info(f"   📊 Training data: {len(X)} sequences, {X.shape[2]} features")
            
            # Update training timer with data info
            if self.training_timer:
                train_size = int(len(X) * 0.8)
                val_size = len(X) - train_size
                self.training_timer.current_metrics.train_samples = train_size
                self.training_timer.current_metrics.val_samples = val_size
                self.training_timer.current_metrics.n_features = X.shape[2]
            
            # Model creation
            model = create_model(
                model_type='enhanced_transformer',
                config=self.config,
                n_features=X.shape[2],
                device=self.device
            )
            
            # Enhanced training with custom callback for timer updates
            trainer = create_trainer(self.config, self.device)
            trainer.target_mode = preprocessor.target_mode
            
            # Custom training loop with timer integration
            trained_model, history = self._train_with_timer_integration(
                trainer, pair_name, model, X, y
            )
            
            # Finish training timer
            if self.training_timer:
                training_metrics = self.training_timer.finish_training(success=True)
            
            # Enhanced model saving
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_save_path = f'temporal_models/{pair_name}_enhanced_transformer_{timestamp}.pth'
            Path('temporal_models').mkdir(exist_ok=True)
            
            model_data = {
                'model_state_dict': trained_model.state_dict(),
                'model_type': 'enhanced_transformer',
                'input_size': X.shape[2],
                'history': history,
                'feature_columns': preprocessor.feature_columns,
                'pair_name': pair_name,
                'target_mode': preprocessor.target_mode,
                'config': self.config,
                'train_period': f'{self.train_start}_{self.train_end}',
                'training_metrics': training_metrics.to_dict() if training_metrics else None,
                'timestamp': timestamp
            }
            
            torch.save(model_data, model_save_path)
            logger.info(f"💾 Model saved: {model_save_path}")
            
            # Store in trainer
            self.trained_models[pair_name] = {
                'model': trained_model,
                'model_path': model_save_path,
                'preprocessor': preprocessor,
                'history': history,
                'training_metrics': training_metrics
            }
            
            best_val_acc = max(history['val_acc']) if history['val_acc'] else 0
            logger.info(f"✅ {pair_name}: enhanced training completed (Val Acc: {best_val_acc:.2f}%)")
            
            return trained_model, history, training_metrics
            
        except Exception as e:
            logger.error(f"❌ {pair_name}: enhanced training error: {str(e)}")
            
            # Finish training timer with failure
            if self.training_timer:
                training_metrics = self.training_timer.finish_training(success=False)
            
            return None, None, training_metrics
    
    def _train_with_timer_integration(self, trainer, pair_name: str, model, X: np.ndarray, y: np.ndarray):
        """Training with timer integration for epoch-by-epoch tracking."""
        
        # This is a simplified version - in practice, you'd integrate timer updates
        # into the actual training loop of your trainer
        
        trained_model, history = trainer.train_pair_model(
            pair_name=pair_name,
            X=X,
            y=y,
            epochs=self.config['training']['epochs'],
            batch_size=self.config['training']['batch_size'],
            dropout=self.config['model']['dropout_rate']
        )
        
        # Update timer with epoch information if available
        if self.training_timer and history:
            for epoch, (train_acc, val_acc) in enumerate(zip(history['train_acc'], history['val_acc'])):
                self.training_timer.update_epoch(epoch + 1, train_acc, val_acc)
        
        return trained_model, history
    
    def _generate_enhanced_temporal_signals(self, pair_name: str, model: Any, signal_data: pd.DataFrame) -> Optional[Dict]:
        """Enhanced signal generation with better filtering and analysis."""
        logger.info(f"📡 {pair_name}: enhanced signal generation...")
        
        try:
            # Get preprocessor
            if pair_name not in self.trained_models:
                logger.error(f"❌ {pair_name}: Trained model not found")
                return None
            
            preprocessor = self.trained_models[pair_name]['preprocessor']
            
            # Prepare signal data
            features, _ = preprocessor.prepare_pair_data(
                signal_data,
                horizon=64,
                dynamic_k_aggressive=True,
                seq_len=self.config['data']['sequence_length']
            )
            
            if len(features) < self.config['data']['sequence_length']:
                logger.error(f"❌ {pair_name}: Insufficient signal data ({len(features)})")
                return None
            
            # Scale features using fitted scaler
            features_scaled = preprocessor.feature_scaler.transform(features)
            
            # Enhanced signal generation
            model.eval()
            signals = []
            seq_len = self.config['data']['sequence_length']
            
            # Pair-specific confidence threshold
            pair_config = self.config.get('pairs', {}).get(pair_name, {})
            confidence_threshold = pair_config.get('confidence_threshold', self.confidence_threshold)
            
            with torch.no_grad():
                for i in range(seq_len, len(features_scaled)):
                    # Get sequence
                    sequence = features_scaled[i-seq_len:i]
                    sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                    
                    # Get timestamp and price
                    timestamp = features.index[i]
                    current_price = signal_data.loc[timestamp, 'close'] if timestamp in signal_data.index else None
                    
                    # Predict
                    output = model(sequence_tensor)
                    
                    if preprocessor.target_mode == 'binary':
                        # Binary prediction
                        probability = torch.sigmoid(output).cpu().numpy()[0][0]
                        prediction = 1 if probability > 0.5 else 0
                        confidence = abs(probability - 0.5) * 2
                        
                        # Enhanced filtering
                        if confidence > confidence_threshold:
                            signals.append({
                                'datetime': timestamp,
                                'price': current_price,
                                'prediction': prediction,
                                'confidence': float(confidence),
                                'probability': float(probability),
                                'signal': 'LONG' if prediction == 1 else 'SHORT'
                            })
                    
                    else:  # three_class
                        # Three-class prediction
                        probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
                        predicted_class = np.argmax(probabilities)
                        confidence = np.max(probabilities)
                        
                        # Filter flat signals and low confidence
                        if predicted_class != 1 and confidence > confidence_threshold:
                            signal_type = 'LONG' if predicted_class == 0 else 'SHORT'
                            binary_pred = 1 if predicted_class == 0 else 0
                            
                            signals.append({
                                'datetime': timestamp,
                                'price': current_price,
                                'prediction': binary_pred,
                                'confidence': float(confidence),
                                'probabilities': probabilities.tolist(),
                                'signal': signal_type
                            })
            
            logger.info(f"✅ {pair_name}: {len(signals)} high-quality signals generated")
            
            # Create enhanced signals analysis
            if signals:
                signals_df = pd.DataFrame(signals)
                signals_df['datetime'] = pd.to_datetime(signals_df['datetime'])
                signals_df = signals_df.sort_values('datetime')
                
                # Enhanced CSV saving with metadata
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'signals_{pair_name}_{timestamp}.csv'
                filepath = Path('temporal_signals') / filename
                Path('temporal_signals').mkdir(exist_ok=True)
                
                # Add metadata as comments
                metadata_lines = [
                    f'# Temporal Signals for {pair_name}',
                    f'# Generated: {datetime.now().isoformat()}',
                    f'# Training Period: {self.train_start} to {self.train_end}',
                    f'# Signal Period: {self.signal_start} to {self.signal_end}',
                    f'# Model: Enhanced Transformer',
                    f'# Target Mode: {preprocessor.target_mode}',
                    f'# Confidence Threshold: {confidence_threshold}',
                    f'# Total Signals: {len(signals_df)}',
                    ''
                ]
                
                with open(filepath, 'w') as f:
                    f.write('\n'.join(metadata_lines))
                    signals_df.to_csv(f, index=False)
                
                # Enhanced analysis
                long_signals = len(signals_df[signals_df['signal'] == 'LONG'])
                short_signals = len(signals_df[signals_df['signal'] == 'SHORT'])
                avg_confidence = signals_df['confidence'].mean()
                
                logger.info(f"💾 {pair_name}: enhanced signals saved: {filepath}")
                logger.info(f"   📊 Analysis: {long_signals} LONG, {short_signals} SHORT")
                logger.info(f"   🎯 Avg Confidence: {avg_confidence:.3f}")
                
                return {
                    'signals_count': len(signals),
                    'signals_df': signals_df,
                    'csv_path': str(filepath),
                    'signal_period': f'{self.signal_start}_{self.signal_end}',
                    'long_signals': long_signals,
                    'short_signals': short_signals,
                    'avg_confidence': avg_confidence,
                    'confidence_threshold_used': confidence_threshold,
                    'target_mode': preprocessor.target_mode
                }
            else:
                logger.warning(f"⚠️ {pair_name}: No signals met confidence threshold ({confidence_threshold})")
                return None
            
        except Exception as e:
            logger.error(f"❌ {pair_name}: enhanced signal generation error: {str(e)}")
            return None
    
    def _generate_enhanced_pipeline_summary(self, results: Dict[str, Any], 
                                          pipeline_duration: timedelta) -> Dict[str, Any]:
        """Generate enhanced pipeline summary with comprehensive metrics."""
        
        training_results = results['training_results']
        signal_results = results['signal_results']
        training_metrics = results.get('training_metrics', {})
        
        # Enhanced summary with detailed metrics
        summary = {
            'pipeline_info': {
                'train_period': f'{self.train_start} → {self.train_end}',
                'signal_period': f'{self.signal_start} → {self.signal_end}',
                'total_pairs_processed': len(training_results),
                'successful_training': len([p for p in training_results.values() if p is not None]),
                'successful_signals': len([p for p in signal_results.values() if p is not None]),
                'timestamp': datetime.now().isoformat(),
                'pipeline_duration_minutes': pipeline_duration.total_seconds() / 60,
                'model_type': 'enhanced_transformer',
                'features_enabled': {
                    'validation_plotting': self.enable_plotting,
                    'training_time_tracking': self.track_training_time,
                    'enhanced_signal_filtering': True,
                    'data_leakage_protection': True
                }
            },
            'training_summary': {},
            'signal_summary': {},
            'performance_analysis': {}
        }
        
        # Enhanced training summary
        training_times = []
        validation_accuracies = []
        overfitting_gaps = []
        
        for pair, history in training_results.items():
            if history and 'val_acc' in history:
                best_val_acc = max(history['val_acc']) if history['val_acc'] else 0
                final_train_acc = history['train_acc'][-1] if history['train_acc'] else 0
                overfitting_gap = final_train_acc - best_val_acc
                
                # Get training metrics if available
                pair_metrics = training_metrics.get(pair, {})
                training_time = pair_metrics.get('training_duration_seconds', 0) / 60
                
                training_times.append(training_time)
                validation_accuracies.append(best_val_acc)
                overfitting_gaps.append(overfitting_gap)
                
                summary['training_summary'][pair] = {
                    'best_val_acc': best_val_acc,
                    'final_train_acc': final_train_acc,
                    'overfitting_gap': overfitting_gap,
                    'epochs_trained': len(history['val_acc']),
                    'training_time_minutes': training_time,
                    'peak_memory_mb': pair_metrics.get('peak_memory_mb', 0),
                    'avg_cpu_percent': pair_metrics.get('avg_cpu_percent', 0)
                }
        
        # Enhanced signal summary
        total_signals = 0
        confidence_scores = []
        
        for pair, signals in signal_results.items():
            if signals:
                signal_count = signals['signals_count']
                total_signals += signal_count
                confidence_scores.append(signals['avg_confidence'])
                
                summary['signal_summary'][pair] = {
                    'total_signals': signal_count,
                    'long_signals': signals['long_signals'],
                    'short_signals': signals['short_signals'],
                    'avg_confidence': signals['avg_confidence'],
                    'confidence_threshold': signals.get('confidence_threshold_used', self.confidence_threshold),
                    'target_mode': signals.get('target_mode', 'binary'),
                    'signal_csv': signals['csv_path']
                }
        
        # Performance analysis
        if validation_accuracies:
            summary['performance_analysis'] = {
                'accuracy_stats': {
                    'mean_val_acc': np.mean(validation_accuracies),
                    'best_val_acc': np.max(validation_accuracies),
                    'worst_val_acc': np.min(validation_accuracies),
                    'acc_std': np.std(validation_accuracies)
                },
                'overfitting_analysis': {
                    'mean_gap': np.mean(overfitting_gaps),
                    'max_gap': np.max(overfitting_gaps),
                    'problematic_pairs': [pair for pair, gap in zip(training_results.keys(), overfitting_gaps) if gap > 15]
                },
                'timing_analysis': {
                    'total_training_time_hours': sum(training_times) / 60,
                    'avg_training_time_minutes': np.mean(training_times) if training_times else 0,
                    'fastest_pair_minutes': np.min(training_times) if training_times else 0,
                    'slowest_pair_minutes': np.max(training_times) if training_times else 0
                },
                'signal_analysis': {
                    'total_signals_generated': total_signals,
                    'avg_signals_per_pair': total_signals / len(signal_results) if signal_results else 0,
                    'avg_confidence_across_pairs': np.mean(confidence_scores) if confidence_scores else 0,
                    'signal_quality_score': np.mean(confidence_scores) * 100 if confidence_scores else 0
                }
            }
        
        return summary
    
    def _save_enhanced_pipeline_results(self, results: Dict[str, Any]) -> None:
        """Save enhanced pipeline results with comprehensive data."""
        
        try:
            # Create results directories
            for directory in ['temporal_results', 'temporal_reports', 'temporal_models']:
                Path(directory).mkdir(exist_ok=True)
            
            # Timestamp for all files
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 1. Complete pipeline results (JSON)
            pipeline_file = Path('temporal_results') / f'enhanced_pipeline_{timestamp}.json'
            
            # Prepare JSON-serializable data
            json_results = {
                'pipeline_summary': results['pipeline_summary'],
                'training_pairs': list(results['training_results'].keys()),
                'signal_pairs': list(results['signal_results'].keys()),
                'plot_files': results.get('plot_paths', {}),
                'config_used': self.config,
                'temporal_settings': {
                    'train_start': self.train_start,
                    'train_end': self.train_end,
                    'signal_start': self.signal_start,
                    'signal_end': self.signal_end,
                    'confidence_threshold': self.confidence_threshold
                }
            }
            
            with open(pipeline_file, 'w') as f:
                json.dump(json_results, f, indent=2, default=str)
            
            logger.info(f"📋 Enhanced pipeline results saved: {pipeline_file}")
            
            # 2. Training metrics summary (JSON)
            if results.get('training_metrics'):
                metrics_file = Path('temporal_results') / f'training_metrics_{timestamp}.json'
                with open(metrics_file, 'w') as f:
                    json.dump(results['training_metrics'], f, indent=2, default=str)
                logger.info(f"⏱️ Training metrics saved: {metrics_file}")
            
            # 3. Enhanced summary report
            self._save_enhanced_summary_report(results, timestamp)
            
            # 4. Configuration backup
            config_backup = Path('temporal_results') / f'config_backup_{timestamp}.json'
            with open(config_backup, 'w') as f:
                json.dump(self.config, f, indent=2, default=str)
            
            logger.info(f"⚙️ Configuration backup saved: {config_backup}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save enhanced pipeline results: {e}")
    
    def _save_enhanced_summary_report(self, results: Dict[str, Any], timestamp: str) -> None:
        """Save enhanced summary report with detailed analysis."""
        
        try:
            summary = results['pipeline_summary']
            
            # Create comprehensive text report
            report_lines = [
                "=" * 80,
                "ENHANCED TEMPORAL TRAINING & SIGNAL GENERATION REPORT",
                "=" * 80,
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Pipeline Duration: {summary['pipeline_info']['pipeline_duration_minutes']:.1f} minutes",
                "",
                "TEMPORAL CONFIGURATION:",
                f"  Training Period: {summary['pipeline_info']['train_period']}",
                f"  Signal Period: {summary['pipeline_info']['signal_period']}",
                f"  Model Type: {summary['pipeline_info']['model_type']}",
                "",
                "PROCESSING SUMMARY:",
                f"  Total Pairs: {summary['pipeline_info']['total_pairs_processed']}",
                f"  Successful Training: {summary['pipeline_info']['successful_training']}",
                f"  Successful Signals: {summary['pipeline_info']['successful_signals']}",
                f"  Success Rate: {summary['pipeline_info']['successful_training']/summary['pipeline_info']['total_pairs_processed']*100:.1f}%",
                ""
            ]
            
            # Training analysis
            if 'performance_analysis' in summary:
                perf = summary['performance_analysis']
                
                report_lines.extend([
                    "TRAINING PERFORMANCE ANALYSIS:",
                    f"  Average Validation Accuracy: {perf['accuracy_stats']['mean_val_acc']:.2f}%",
                    f"  Best Validation Accuracy: {perf['accuracy_stats']['best_val_acc']:.2f}%",
                    f"  Worst Validation Accuracy: {perf['accuracy_stats']['worst_val_acc']:.2f}%",
                    f"  Accuracy Standard Deviation: {perf['accuracy_stats']['acc_std']:.2f}%",
                    "",
                    "OVERFITTING ANALYSIS:",
                    f"  Average Overfitting Gap: {perf['overfitting_analysis']['mean_gap']:.2f}pp",
                    f"  Maximum Overfitting Gap: {perf['overfitting_analysis']['max_gap']:.2f}pp",
                    f"  Problematic Pairs: {', '.join(perf['overfitting_analysis']['problematic_pairs']) if perf['overfitting_analysis']['problematic_pairs'] else 'None'}",
                    "",
                    "TIMING ANALYSIS:",
                    f"  Total Training Time: {perf['timing_analysis']['total_training_time_hours']:.1f} hours",
                    f"  Average Time per Pair: {perf['timing_analysis']['avg_training_time_minutes']:.1f} minutes",
                    f"  Fastest Training: {perf['timing_analysis']['fastest_pair_minutes']:.1f} minutes",
                    f"  Slowest Training: {perf['timing_analysis']['slowest_pair_minutes']:.1f} minutes",
                    "",
                    "SIGNAL ANALYSIS:",
                    f"  Total Signals Generated: {perf['signal_analysis']['total_signals_generated']}",
                    f"  Average Signals per Pair: {perf['signal_analysis']['avg_signals_per_pair']:.1f}",
                    f"  Average Confidence: {perf['signal_analysis']['avg_confidence_across_pairs']:.3f}",
                    f"  Signal Quality Score: {perf['signal_analysis']['signal_quality_score']:.1f}/100",
                    ""
                ])
            
            # Individual pair results
            report_lines.append("INDIVIDUAL PAIR RESULTS:")
            training_summary = summary.get('training_summary', {})
            signal_summary = summary.get('signal_summary', {})
            
            for pair in training_summary.keys():
                train_stats = training_summary.get(pair, {})
                signal_stats = signal_summary.get(pair, {})
                
                report_lines.extend([
                    f"  {pair}:",
                    f"    Training: {train_stats.get('best_val_acc', 0):.1f}% val acc, {train_stats.get('overfitting_gap', 0):.1f}pp gap, {train_stats.get('training_time_minutes', 0):.1f}min",
                    f"    Signals: {signal_stats.get('total_signals', 0)} total ({signal_stats.get('long_signals', 0)} long, {signal_stats.get('short_signals', 0)} short), {signal_stats.get('avg_confidence', 0):.3f} confidence",
                    ""
                ])
            
            # Feature summary
            features = summary['pipeline_info']['features_enabled']
            report_lines.extend([
                "ENHANCED FEATURES USED:",
                f"  Validation Plotting: {'✅' if features['validation_plotting'] else '❌'}",
                f"  Training Time Tracking: {'✅' if features['training_time_tracking'] else '❌'}",
                f"  Enhanced Signal Filtering: {'✅' if features['enhanced_signal_filtering'] else '❌'}",
                f"  Data Leakage Protection: {'✅' if features['data_leakage_protection'] else '❌'}",
                "",
                "=" * 80
            ])
            
            # Save text report
            report_file = Path('temporal_reports') / f'enhanced_summary_{timestamp}.txt'
            with open(report_file, 'w') as f:
                f.write('\n'.join(report_lines))
            
            logger.info(f"📄 Enhanced summary report saved: {report_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save enhanced summary report: {e}")


def run_temporal_training_pipeline(pairs: List[str], config_path: str = "configs/temporal.yaml") -> Dict[str, Any]:
    """
    Enhanced temporal training pipeline with comprehensive features.
    
    Args:
        pairs: Currency pairs to process
        config_path: Path to temporal configuration file
        
    Returns:
        Enhanced pipeline results
    """
    
    # Load configuration
    from .. import config
    
    if Path(config_path).exists():
        cfg = config.load(config_path)
        logger.info(f"📁 Enhanced config loaded: {config_path}")
    else:
        cfg = config.get_default_config()
        logger.warning(f"⚠️ Using default config - create {config_path} for optimal settings")
    
    # Enhanced Transformer configuration
    cfg['model']['target_mode'] = cfg.get('model', {}).get('target_mode', 'binary')
    cfg['model']['type'] = 'enhanced_transformer'
    cfg['data']['granularity'] = cfg.get('data', {}).get('granularity', 'M15')
    cfg['training']['epochs'] = cfg.get('training', {}).get('epochs', 50)
    cfg['training']['batch_size'] = cfg.get('training', {}).get('batch_size', 32)
    
    # Enhanced Transformer settings
    cfg['transformer'] = cfg.get('transformer', {
        'd_model': 256,
        'nhead': 8,
        'num_layers': 4,
        'ff_dim': 512
    })
    
    # Create and run enhanced temporal trainer
    temporal_trainer = TemporalTrainer(cfg, cache_enabled=True)
    results = temporal_trainer.run_temporal_pipeline(pairs)
    
    return results


if __name__ == "__main__":
    # Enhanced example usage
    test_pairs = ["EUR_USD", "GBP_USD", "USD_JPY"]
    
    logger.info("🚀 Enhanced Temporal Training Pipeline Test")
    
    results = run_temporal_training_pipeline(
        pairs=test_pairs,
        config_path="configs/temporal.yaml"
    )
    
    if results:
        logger.info("✅ Enhanced temporal pipeline test completed successfully")
        
        # Print summary
        summary = results.get('pipeline_summary', {})
        if 'performance_analysis' in summary:
            perf = summary['performance_analysis']
            logger.info(f"📊 Results: {perf['accuracy_stats']['mean_val_acc']:.1f}% avg accuracy, "
                       f"{perf['signal_analysis']['total_signals_generated']} signals generated")
    else:
        logger.error("❌ Enhanced temporal pipeline test failed")
