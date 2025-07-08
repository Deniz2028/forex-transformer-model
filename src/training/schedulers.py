# src/training/schedulers.py
import torch
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
import numpy as np

class OneCycleLRConfig:
    """OneCycleLR için forex-specific konfigürasyon"""
    def __init__(self, max_lr=0.01, pct_start=0.25, div_factor=10.0, 
                 final_div_factor=1000.0, three_phase=True):
        self.max_lr = max_lr
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.three_phase = three_phase
        self.anneal_strategy = 'cos'
        self.cycle_momentum = True
        self.base_momentum = 0.85
        self.max_momentum = 0.95

class ForexOneCycleLR:
    """
    Forex prediction için optimize edilmiş OneCycleLR wrapper
    """
    def __init__(self, optimizer, config, total_steps):
        self.optimizer = optimizer
        self.config = config
        self.total_steps = total_steps
        
        self.scheduler = OneCycleLR(
            optimizer,
            max_lr=config.max_lr,
            total_steps=total_steps,
            pct_start=config.pct_start,
            div_factor=config.div_factor,
            final_div_factor=config.final_div_factor,
            three_phase=config.three_phase,
            anneal_strategy=config.anneal_strategy,
            cycle_momentum=config.cycle_momentum,
            base_momentum=config.base_momentum,
            max_momentum=config.max_momentum
        )
        
        self.step_count = 0
        self.lr_history = []
        
    def step(self):
        """Scheduler step with logging"""
        self.scheduler.step()
        self.step_count += 1
        current_lr = self.scheduler.get_last_lr()[0]
        self.lr_history.append(current_lr)
        
    def get_lr(self):
        """Current learning rate"""
        return self.scheduler.get_last_lr()[0]
        
    def plot_lr_schedule(self, save_path=None):
        """Learning rate schedule görselleştirmesi"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.lr_history)
        plt.title('OneCycleLR Schedule')
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

def find_optimal_lr(model, dataloader, optimizer, start_lr=1e-8, end_lr=10, num_iter=100):
    """
    LR Range Test - optimal learning rate bulma
    """
    model.train()
    lr_mult = (end_lr / start_lr) ** (1 / num_iter)
    lr = start_lr
    optimizer.param_groups[0]['lr'] = lr
    
    avg_loss = 0.0
    best_loss = 0.0
    losses = []
    lrs = []
    
    for i, batch in enumerate(dataloader):
        if i >= num_iter:
            break
            
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch['features'])
        loss = torch.nn.functional.cross_entropy(outputs, batch['targets'])
        
        # Compute the smoothed loss
        avg_loss = 0.98 * avg_loss + 0.02 * loss.item()
        smoothed_loss = avg_loss / (1 - 0.98 ** (i + 1))
        
        # Stop if loss is exploding
        if i > 10 and smoothed_loss > 4 * best_loss:
            break
            
        # Record the best loss
        if smoothed_loss < best_loss or i == 0:
            best_loss = smoothed_loss
            
        # Store the values
        losses.append(smoothed_loss)
        lrs.append(lr)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update learning rate
        lr *= lr_mult
        optimizer.param_groups[0]['lr'] = lr
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('LR Range Test')
    plt.grid(True)
    plt.show()
    
    return lrs, losses

def get_forex_scheduler_config(model_size='medium', training_regime='standard'):
    """
    Model boyutu ve training rejimina göre optimal scheduler config
    """
    configs = {
        'small': {
            'standard': OneCycleLRConfig(max_lr=0.01, pct_start=0.3, div_factor=25),
            'aggressive': OneCycleLRConfig(max_lr=0.02, pct_start=0.25, div_factor=10)
        },
        'medium': {
            'standard': OneCycleLRConfig(max_lr=0.005, pct_start=0.25, div_factor=10),
            'aggressive': OneCycleLRConfig(max_lr=0.01, pct_start=0.2, div_factor=5)
        },
        'large': {
            'standard': OneCycleLRConfig(max_lr=0.003, pct_start=0.3, div_factor=15),
            'aggressive': OneCycleLRConfig(max_lr=0.005, pct_start=0.25, div_factor=8)
        }
    }
    
    return configs[model_size][training_regime]