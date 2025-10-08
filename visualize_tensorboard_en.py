#!/usr/bin/env python3
"""
TensorBoard Data Visualization Script
Reads and visualizes training metrics from TensorBoard logs
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob

def load_tensorboard_data(log_dir):
    """Load TensorBoard data"""
    print(f"Loading TensorBoard data from: {log_dir}")
    
    # Find all event files
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not event_files:
        print(f"No event files found in {log_dir}")
        return None
    
    print(f"Found event file: {event_files[0]}")
    
    # Create EventAccumulator
    ea = EventAccumulator(event_files[0])
    ea.Reload()
    
    print(f"Available scalars: {ea.Tags()['scalars']}")
    return ea

def plot_metrics(ea, save_path=None):
    """Plot training metrics"""
    scalars = ea.Tags()['scalars']
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics Visualization', fontsize=16)
    
    # Plot loss
    if 'VLA Train/Loss' in scalars:
        loss_data = ea.Scalars('VLA Train/Loss')
        steps = [x.step for x in loss_data]
        values = [x.value for x in loss_data]
        
        axes[0, 0].plot(steps, values, 'b-', linewidth=2)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Training Steps')
        axes[0, 0].set_ylabel('Loss Value')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot learning rate
    if 'VLA Train/Learning Rate' in scalars:
        lr_data = ea.Scalars('VLA Train/Learning Rate')
        steps = [x.step for x in lr_data]
        values = [x.value for x in lr_data]
        
        axes[0, 1].plot(steps, values, 'r-', linewidth=2)
        axes[0, 1].set_title('Learning Rate')
        axes[0, 1].set_xlabel('Training Steps')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot current action L1 loss
    if 'VLA Train/Curr Action L1 Loss' in scalars:
        curr_l1_data = ea.Scalars('VLA Train/Curr Action L1 Loss')
        steps = [x.step for x in curr_l1_data]
        values = [x.value for x in curr_l1_data]
        
        axes[1, 0].plot(steps, values, 'g-', linewidth=2)
        axes[1, 0].set_title('Current Action L1 Loss')
        axes[1, 0].set_xlabel('Training Steps')
        axes[1, 0].set_ylabel('L1 Loss')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot next actions L1 loss
    if 'VLA Train/Next Actions L1 Loss' in scalars:
        next_l1_data = ea.Scalars('VLA Train/Next Actions L1 Loss')
        steps = [x.step for x in next_l1_data]
        values = [x.value for x in next_l1_data]
        
        axes[1, 1].plot(steps, values, 'm-', linewidth=2)
        axes[1, 1].set_title('Next Actions L1 Loss')
        axes[1, 1].set_xlabel('Training Steps')
        axes[1, 1].set_ylabel('L1 Loss')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {save_path}")
    
    plt.show()
    
    return fig

def print_metrics_summary(ea):
    """Print metrics summary"""
    scalars = ea.Tags()['scalars']
    
    print("\n=== Training Metrics Summary ===")
    
    for metric_name in scalars:
        data = ea.Scalars(metric_name)
        if data:
            values = [x.value for x in data]
            steps = [x.step for x in data]
            
            print(f"\n{metric_name}:")
            print(f"  Step range: {min(steps)} - {max(steps)}")
            print(f"  Value range: {min(values):.6f} - {max(values):.6f}")
            print(f"  Final value: {values[-1]:.6f}")
            print(f"  Data points: {len(values)}")

def main():
    # Latest TensorBoard log directory
    # log_dir = "logs/tensorboard/openvla-libero-spatial+libero_spatial_no_noops_mini+b2+lr-0.001+lora-r16+dropout-0.02025-10-07 21:06:27.684839"
    log_dir = "logs/tensorboard/openvla-libero-spatial+libero_spatial_no_noops_mini+b2+lr-0.0003+lora-r16+dropout-0.02025-10-07 22:02:38.907409"
    
    if not os.path.exists(log_dir):
        print(f"Log directory does not exist: {log_dir}")
        return
    
    # Load data
    ea = load_tensorboard_data(log_dir)
    if ea is None:
        return
    
    # Print metrics summary
    print_metrics_summary(ea)
    
    # Plot charts
    plot_metrics(ea, save_path="training_metrics_en.png")
    
    print("\n=== TensorBoard Access Information ===")
    print("TensorBoard is running in the background. You can access it via:")
    print("1. Browser: http://localhost:6006")
    print("2. SSH port forwarding: ssh -L 6006:localhost:6006 your_server")
    print("3. Chart saved as: training_metrics_en.png")

if __name__ == "__main__":
    main()
