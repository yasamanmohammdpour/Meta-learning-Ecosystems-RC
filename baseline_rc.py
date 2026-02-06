# baseline_rc.py
# Train RC from scratch (NO meta-learning baseline)
# Tests both SHORT-TERM and LONG-TERM predictions

import torch
import torch.nn as nn
import numpy as np
from rc_model import Reservoir
from load_tasks import ReadTasks, ExtractTasks
from utils import loss_long_term
import matplotlib.pyplot as plt
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


def train_and_test_rc_baseline(train_data, test_data, train_length=20000, 
                                test_length=50000, ridge_beta=1e-4, 
                                noise_level=0.003):
    """
    Train RC from scratch (no meta-learning).
    
    Args:
        train_data: Training trajectory
        test_data: Testing trajectory
        train_length: How many samples to train on
        test_length: How many steps to predict
        ridge_beta: Regularization
        noise_level: Noise
    
    Returns:
        dv: Divergence metric
        train_loss: Training loss
        real: Ground truth
        pred: Predictions
    """
    
    print(f"  Training RC from scratch (random initialization)...")
    
    # Create RANDOM RC (no meta-learning!)
    model = Reservoir(
        input_dim=3,
        reservoir_size=1000,
        output_dim=3,
        spectral_radius=0.95,
        sparsity=0.1,
        alpha=0.3,
        w_in_scale=0.1,
        device=DEVICE
    )
    
    # === TRAINING ===
    model.eval()
    model.reset_state()
    
    noise = np.random.normal(0., noise_level, size=train_data.shape)
    data_noisy = np.clip(train_data + noise, 0, 1)
    
    available_pairs = len(data_noisy) - 1
    train_len = min(train_length, available_pairs)
    
    X = torch.tensor(data_noisy[:train_len], dtype=DTYPE, device=DEVICE)
    Y = torch.tensor(data_noisy[1:train_len+1], dtype=DTYPE, device=DEVICE)
    
    with torch.no_grad():
        states = torch.empty((X.shape[0], model.n), device=DEVICE, dtype=DTYPE)
        
        for t in range(X.shape[0]):
            states[t] = model.step(X[t])
        
        # Ridge regression
        n_samples = states.shape[0]
        n_features = states.shape[1]
        
        sqrt_beta = torch.sqrt(torch.tensor(ridge_beta, device=DEVICE, dtype=DTYPE))
        
        states_aug = torch.cat([
            states,
            sqrt_beta * torch.eye(n_features, device=DEVICE, dtype=DTYPE)
        ], dim=0)
        
        Y_aug = torch.cat([
            Y,
            torch.zeros(n_features, Y.shape[1], device=DEVICE, dtype=DTYPE)
        ], dim=0)
        
        W_out_T = torch.linalg.lstsq(states_aug, Y_aug, rcond=None).solution
        model.W_out.weight.data = W_out_T.T
    
    train_loss = nn.functional.mse_loss(model.readout(states), Y).item()
    print(f"    Train loss: {train_loss:.6f}")
    
    # === TESTING ===
    model.reset_state()
    
    clean_test = test_data.copy()
    noise = np.random.normal(0., noise_level, size=test_data.shape)
    data_noisy = np.clip(test_data + noise, 0, 1)
    
    available_pairs = len(data_noisy) - 1
    test_len = min(test_length, available_pairs)
    
    current_state = torch.tensor(data_noisy[0], dtype=DTYPE, device=DEVICE)
    target = torch.tensor(clean_test[1:test_len+1], dtype=DTYPE)
    
    preds = []
    
    with torch.no_grad():
        for step_idx in range(target.shape[0]):
            r = model.step(current_state)
            y = model.readout(r)
            
            y_clipped = torch.clamp(y, 0.0, 1.0)
            preds.append(y_clipped.cpu().numpy())
            
            current_state = y_clipped
    
    real = target.cpu().numpy()
    pred = np.array(preds)
    
    dv = loss_long_term(real, pred, dt=0.04)
    print(f"    DV: {dv:.4f}")
    
    return dv, train_loss, real, pred


def save_plots(real, pred, name, test_type, output_dir="plots_rc_baseline"):
    """
    Save 3D attractors and time series plots.
    
    Args:
        real: Ground truth trajectory
        pred: Predicted trajectory
        name: System name
        test_type: "short_term" or "long_term"
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 3D Real attractor
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")
    ax.plot(real[:, 0], real[:, 1], real[:, 2], color="orange", label="Real", linewidth=0.5, alpha=0.8)
    ax.set_title(f"{name} - Real Attractor ({test_type})", fontsize=14, weight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{name}_{test_type}_real_3d.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3D Predicted attractor
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")
    ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], color="blue", label="Predicted", linewidth=0.5, alpha=0.8)
    ax.set_title(f"{name} - Predicted Attractor ({test_type})", fontsize=14, weight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{name}_{test_type}_pred_3d.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Time series comparison
    fig, ax = plt.subplots(3, 1, figsize=(12, 8))
    pred_length = min(2000, len(pred))
    
    for i in range(3):
        ax[i].plot(real[:pred_length, i], label="Real", color='orange', linewidth=1.0, alpha=0.8)
        ax[i].plot(pred[:pred_length, i], label="Predicted", color='blue', linewidth=1.0, alpha=0.8)
        ax[i].set_ylabel(f"Dim {i+1}", fontsize=11)
        ax[i].grid(True, alpha=0.3)
        if i == 0:
            ax[i].set_title(f"{name} - Time Series Comparison ({test_type})", fontsize=14, weight='bold')
            ax[i].legend()
        if i == 2:
            ax[i].set_xlabel("Time steps", fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{name}_{test_type}_timeseries.png", dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Test RC-only baseline on all 3 test systems with both short-term and long-term predictions."""
    
    print("="*60)
    print("BASELINE: RC ONLY (No Meta-Learning)")
    print("="*60 + "\n")
    
    # Load test data
    read_tasks = ReadTasks("data")
    
    train_set = ['aizawa']  # Dummy train set
    test_set = ['foodchain', 'hastings', 'lotka']
    
    read_tasks.preprocess(
        random_tasks=False,
        train_set=train_set,
        val_set=[],
        test_set=test_set
    )
    
    train_tasks, _, test_tasks, train_names, _, test_names = read_tasks.get_tasks()
    
    # Test configurations
    test_configs = [
        {"name": "short_term", "test_length": 1000, "label": "Short-Term (1k steps)"},
        {"name": "long_term", "test_length": 50000, "label": "Long-Term (50k steps)"}
    ]
    
    all_results = {}
    
    for config in test_configs:
        print("\n" + "="*60)
        print(f"TESTING: {config['label']}")
        print("="*60)
        
        extract_test = ExtractTasks(
            test_tasks, test_names,
            train_length=20000,
            test_length=config['test_length']
        )
        
        results = []
        test_reals = []
        test_preds = []
        test_names_results = []
        
        for idx in range(len(extract_test)):
            train_data, test_data, name = extract_test.get_specific_data(idx)
            
            print(f"\nTesting system {idx+1}/3: {name}")
            print("-"*60)
            
            dv, train_loss, real, pred = train_and_test_rc_baseline(
                train_data, test_data,
                train_length=20000,
                test_length=config['test_length'],
                ridge_beta=1e-4
            )
            
            results.append({
                'name': name,
                'dv': dv,
                'train_loss': train_loss
            })
            
            test_reals.append(real)
            test_preds.append(pred)
            test_names_results.append(name)
        
        # Save results for this configuration
        all_results[config['name']] = results
        
        # Summary for this test type
        print("\n" + "-"*60)
        print(f"RESULTS: {config['label']}")
        print("-"*60)
        for r in results:
            print(f"{r['name']:20s}: DV = {r['dv']:.4f}, Train Loss = {r['train_loss']:.6f}")
        
        avg_dv = np.mean([r['dv'] for r in results])
        print(f"\nAverage DV: {avg_dv:.4f}")
        print("-"*60)
        
        # Generate plots
        print(f"\nGenerating plots for {config['label']}...")
        for real, pred, name in zip(test_reals, test_preds, test_names_results):
            save_plots(real, pred, name, config['name'])
        print(f"✓ All {config['label']} plots saved to ./plots_rc_baseline/")
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY - RC ONLY BASELINE")
    print("="*60)
    
    print("\nSHORT-TERM (1,000 steps):")
    print("-"*40)
    for r in all_results['short_term']:
        print(f"  {r['name']:15s}: DV = {r['dv']:.4f}")
    avg_short = np.mean([r['dv'] for r in all_results['short_term']])
    print(f"  {'Average':15s}: DV = {avg_short:.4f}")
    
    print("\nLONG-TERM (50,000 steps):")
    print("-"*40)
    for r in all_results['long_term']:
        print(f"  {r['name']:15s}: DV = {r['dv']:.4f}")
    avg_long = np.mean([r['dv'] for r in all_results['long_term']])
    print(f"  {'Average':15s}: DV = {avg_long:.4f}")
    
    print("\n" + "="*60)
    print(f"✓ All plots saved to ./plots_rc_baseline/")
    print("="*60)
    
    return all_results


if __name__ == "__main__":
    results = main()