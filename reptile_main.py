# reptile_main.py
# CORRECT RC: No embedding, ridge regression with proper beta

import os
import torch
import torch.nn as nn
import numpy as np
from rc_model import Reservoir
import argparse
import copy
from load_tasks import ReadTasks, ExtractTasks
import matplotlib.pyplot as plt
from utils import *
from tqdm import trange
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ======================================================
# Device handling
# ======================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

torch.set_float32_matmul_precision("high")

print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# ======================================================
# Argument parsing - NO EMBEDDING!
# ======================================================
parser = argparse.ArgumentParser("Fast Reptile RC - NO EMBEDDING")

parser.add_argument("--logdir", default="logdir")
parser.add_argument("--meta-iterations", type=int, default=30)  # Paper quality
parser.add_argument("--start-meta-iteration", type=int, default=0)
parser.add_argument("--meta-lr", type=float, default=1.0)
parser.add_argument("--train-length", type=int, default=20000)  # Paper quality
parser.add_argument("--test-length", type=int, default=50000)   # Paper quality
parser.add_argument("--noise-level", type=float, default=0.003)
parser.add_argument("--ridge-beta", type=float, default=1e-2)  # was 1e-6

# Parallelization
parser.add_argument("--parallel-tasks", action="store_true", default=True)
parser.add_argument("--num-workers", type=int, default=4)

args = parser.parse_args()
print("\n" + "="*60)
print("CORRECT RC: NO EMBEDDING (Reservoir provides memory)")
print("="*60)
print(args)
print("="*60 + "\n")

# ======================================================
# Load tasks
# ======================================================
print("Loading tasks...")
read_tasks = ReadTasks("data")

train_set = ['aizawa', 'bouali', 'chua', 'sprott_03', 'sprott_14']
test_set  = ['foodchain', 'hastings', 'lotka']

read_tasks.preprocess(
    random_tasks=False,
    train_set=train_set,
    val_set=[],
    test_set=test_set
)

train_tasks, val_tasks, test_tasks, train_names, _, test_names = read_tasks.get_tasks()

extract_train_task = ExtractTasks(
    train_tasks, train_names,
    train_length=args.train_length,
    test_length=args.test_length
)

extract_test_task = ExtractTasks(
    test_tasks, test_names,
    train_length=args.train_length,
    test_length=args.test_length
)

print(f"Training tasks: {train_names}")
print(f"Testing tasks: {test_names}\n")

# ======================================================
# Model - NO EMBEDDING: input is just 3D!
# ======================================================
input_size = 3
output_size = 3

print(f"Creating reservoir with input_dim={input_size} (NO embedding)...")
meta_model = Reservoir(
    input_dim=input_size,  # 3D input
    reservoir_size=1000,
    output_dim=output_size,
    spectral_radius=0.95,
    sparsity=0.1,
    alpha=0.3,
    w_in_scale=0.1,
    device=DEVICE
)

loss_fn = nn.MSELoss()

# ======================================================
# Ridge Regression - NO EMBEDDING!
# ======================================================
def learning_process_ridge(model, data):
    """
    RC with ridge regression - NO EMBEDDING.
    """
    model.eval()
    model.reset_state()

    noise = np.random.normal(0., args.noise_level, size=data.shape)
    data = np.clip(data + noise, 0, 1)

    # We need train_length + 1 samples to create train_length pairs
    available_pairs = len(data) - 1  # Max number of (x(t), x(t+1)) pairs
    train_length = min(args.train_length, available_pairs)

    # Now this is safe:
    X = torch.tensor(
        data[:train_length],  # Indices 0 to train_length-1
        dtype=DTYPE,
        device=DEVICE
    )

    Y = torch.tensor(
        data[1:train_length + 1],  # Indices 1 to train_length
        dtype=DTYPE,
        device=DEVICE
    )

    with torch.no_grad():
        # Collect reservoir states
        states = torch.empty(
            (X.shape[0], model.n),
            device=DEVICE,
            dtype=DTYPE
        )

        for t in range(X.shape[0]):
            states[t] = model.step(X[t])

        # Ridge regression
        n_samples = states.shape[0]
        n_features = states.shape[1]
        
        beta = args.ridge_beta
        sqrt_beta = torch.sqrt(torch.tensor(beta, device=DEVICE, dtype=DTYPE))
        
        states_aug = torch.cat([
            states,
            sqrt_beta * torch.eye(n_features, device=DEVICE, dtype=DTYPE)
        ], dim=0)
        
        Y_aug = torch.cat([
            Y,
            torch.zeros(n_features, Y.shape[1], device=DEVICE, dtype=DTYPE)
        ], dim=0)
        
        W_out_T = torch.linalg.lstsq(states_aug, Y_aug, rcond=None).solution
        W_out = W_out_T.T
        
        model.W_out.weight.data = W_out
    
    final_preds = model.readout(states)
    final_loss = nn.functional.mse_loss(final_preds, Y).item()
    
    return model, final_loss


# ======================================================
# Long-term prediction - NO EMBEDDING!
# ======================================================
# def long_term_validation(model, data):
#     """
#     Closed-loop prediction with SOFT clipping and noise injection.
#     """
#     model.eval()
#     model.reset_state()

#     clean_data = data.copy()
#     noise = np.random.normal(0., args.noise_level, size=data.shape)
#     data = np.clip(data + noise, 0, 1)

#     available_pairs = len(data) - 1
#     test_length = min(args.test_length, available_pairs)

#     current_state = torch.tensor(data[0], dtype=DTYPE, device=DEVICE)
    
#     target_test = torch.tensor(
#         clean_data[1:test_length + 1],
#         dtype=DTYPE
#     )

#     preds = []

#     with torch.no_grad():
#         for step_idx in range(target_test.shape[0]):
#             r = model.step(current_state)
#             y = model.readout(r)
            
#             # SOFT clipping: Allow some overshoot
#             y_clipped = torch.clamp(y, -0.1, 1.1)  # ← Changed from [0, 1] to [-0.1, 1.1]
            
#             # Add tiny noise to prevent convergence
#             if step_idx % 10 == 0:  # Every 10 steps
#                 noise = 0.001 * torch.randn_like(y_clipped)
#                 y_clipped = y_clipped + noise
            
#             # Save clipped to [0, 1] for comparison
#             preds.append(torch.clamp(y_clipped, 0.0, 1.0).cpu().numpy())
            
#             # Use slightly-outside-bounds prediction for feedback
#             current_state = y_clipped

#     return target_test.cpu().numpy(), np.array(preds)

def long_term_validation(model, data):
    """Closed-loop prediction - STABLE VERSION."""
    model.eval()
    model.reset_state()

    clean_data = data.copy()
    noise = np.random.normal(0., args.noise_level, size=data.shape)
    data = np.clip(data + noise, 0, 1)

    available_pairs = len(data) - 1
    test_length = min(args.test_length, available_pairs)

    current_state = torch.tensor(data[0], dtype=DTYPE, device=DEVICE)
    target_test = torch.tensor(clean_data[1:test_length + 1], dtype=DTYPE)

    preds = []

    with torch.no_grad():
        for step_idx in range(target_test.shape[0]):
            r = model.step(current_state)
            y = model.readout(r)
            
            # SIMPLE clipping - no noise injection!
            y_clipped = torch.clamp(y, 0.0, 1.0)
            preds.append(y_clipped.cpu().numpy())
            
            # Feed back clipped prediction
            current_state = y_clipped

    return target_test.cpu().numpy(), np.array(preds)
    
# ======================================================
# Parallel task processing
# ======================================================
def process_single_task(task_idx, task_data, task_name, meta_model_state):
    """Process single task in parallel"""
    model = Reservoir(
        input_dim=3,  # ✅ FIXED: was 3*embedding_time
        reservoir_size=1000,
        output_dim=3,
        spectral_radius=0.95,
        sparsity=0.1,
        alpha=0.3,
        w_in_scale=0.1,
        device=DEVICE
    )
    model.load_state_dict(meta_model_state)
    
    model_copy, loss = learning_process_ridge(model, task_data)
    diff = (model_copy.W_out.weight.data - meta_model_state['W_out.weight']).cpu()
    
    return diff, task_name, loss

# ======================================================
# Reptile meta-training
# ======================================================
task_sample_num = 5

print("="*60)
print(f"Starting meta-training: {args.meta_iterations} iterations")
print(f"Parallel processing: {args.parallel_tasks}")
print("="*60 + "\n")

meta_start_time = time.time()

for meta_iter in trange(args.start_meta_iteration, args.meta_iterations, desc="Meta-iterations"):
    iter_start = time.time()
    
    meta_lr = args.meta_lr * (1 - meta_iter / args.meta_iterations)

    # Sample tasks
    task_data_list = []
    task_names_list = []
    used = set()
    
    while len(used) < task_sample_num:
        train_data, _, name = extract_train_task.get_random_data()
        if name in used:
            continue
        task_data_list.append(train_data)
        task_names_list.append(name)
        used.add(name)
    
    meta_state = meta_model.state_dict()
    
    if args.parallel_tasks and args.num_workers > 1:
        # PARALLEL
        with ThreadPoolExecutor(max_workers=min(args.num_workers, task_sample_num)) as executor:
            futures = []
            for idx in range(task_sample_num):
                future = executor.submit(
                    process_single_task,
                    idx,
                    task_data_list[idx],
                    task_names_list[idx],
                    meta_state
                )
                futures.append(future)
            
            task_diffs = []
            for future in as_completed(futures):
                diff, name, loss = future.result()
                task_diffs.append(diff.to(DEVICE))
                print(f"  ✓ Task: {name}, Loss: {loss:.6f}")
        
        sum_gradients = torch.stack(task_diffs).sum(dim=0)
    
    else:
        # SEQUENTIAL
        sum_gradients = None
        for idx in range(task_sample_num):
            model = copy.deepcopy(meta_model)
            name = task_names_list[idx]
            train_data = task_data_list[idx]
            
            print(f'  Task {idx+1}/{task_sample_num}: {name}')
            model_copy, loss = learning_process_ridge(model, train_data)

            diff = model_copy.W_out.weight.data - meta_model.W_out.weight.data
            sum_gradients = diff if sum_gradients is None else sum_gradients + diff
    
    # Update meta-model
    meta_model.W_out.weight.data += (meta_lr / task_sample_num) * sum_gradients
    
    # Progress
    iter_time = time.time() - iter_start
    elapsed_total = time.time() - meta_start_time
    avg_time = elapsed_total / (meta_iter - args.start_meta_iteration + 1)
    remaining_iters = args.meta_iterations - meta_iter - 1
    eta = remaining_iters * avg_time
    
    print(f"\n{'─'*60}")
    print(f"Iter {meta_iter+1}/{args.meta_iterations} in {iter_time:.1f}s | Elapsed: {elapsed_total/60:.1f}m | ETA: {eta/60:.1f}m")
    print(f"{'─'*60}\n")
    
    if (meta_iter + 1) % 10 == 0:
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            'meta_iter': meta_iter,
            'model_state_dict': meta_model.state_dict(),
            'args': args,
        }, f"checkpoints/meta_iter_{meta_iter+1}.pt")
        print(f"✓ Checkpoint saved\n")

total_train_time = time.time() - meta_start_time
print("\n" + "="*60)
print(f"Meta-training completed! Time: {total_train_time/60:.1f} min")
print("="*60 + "\n")

# ======================================================
# Testing
# ======================================================
print("="*60)
print("Testing on unseen systems...")
print("="*60 + "\n")

test_start_time = time.time()
test_reals, test_preds, test_names_results, test_dv = [], [], [], []

for idx in range(len(extract_test_task)):
    task_start = time.time()
    
    model = copy.deepcopy(meta_model)
    model.reset_state()
    
    train_data, test_data, name = extract_test_task.get_specific_data(idx)

    print(f"Testing task {idx+1}/{len(extract_test_task)}: {name}")
    
    model_copy, train_loss = learning_process_ridge(model, train_data)
    print(f"  Train loss: {train_loss:.6f}")
    
    model_copy.reset_state()
    real, pred = long_term_validation(model_copy, test_data)

    dv = loss_long_term(real, pred, dt=0.04)
    
    task_time = time.time() - task_start  # ✅ Now task_start is defined!
    print(f"  DV: {dv:.4f} (Time: {task_time:.1f}s)\n")

    test_reals.append(real)
    test_preds.append(pred)
    test_names_results.append(name)
    test_dv.append(dv)

test_total_time = time.time() - test_start_time
print(f"Testing completed in {test_total_time:.1f}s\n")

# ======================================================
# Summary
# ======================================================
print("="*60)
print("RESULTS SUMMARY")
print("="*60)
for name, dv in zip(test_names_results, test_dv):
    print(f"{name:20s}: DV = {dv:.4f}")
print(f"\nAverage DV: {np.mean(test_dv):.4f}")
print("="*60 + "\n")

# ======================================================
# Plotting
# ======================================================
print("Generating plots...")
for real, pred, name in zip(test_reals, test_preds, test_names_results):

    os.makedirs("plots", exist_ok=True)

    # 3D Real
    ax = plt.figure(figsize=(10, 8)).add_subplot(projection="3d")
    ax.plot(real[:, 0], real[:, 1], real[:, 2], color="orange", label="Real", linewidth=0.5, alpha=0.8)
    ax.set_title(f"{name} - Real Attractor", fontsize=14, weight='bold')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{name}_real_3d.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 3D Predicted
    ax = plt.figure(figsize=(10, 8)).add_subplot(projection="3d")
    ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], color="blue", label="Predicted", linewidth=0.5, alpha=0.8)
    ax.set_title(f"{name} - Predicted Attractor", fontsize=14, weight='bold')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{name}_pred_3d.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Time series
    fig, ax = plt.subplots(3, 1, figsize=(12, 8))
    pred_length = min(2000, len(pred))

    for i in range(3):
        ax[i].plot(real[:pred_length, i], label="Real", color='orange', linewidth=1.0, alpha=0.8)
        ax[i].plot(pred[:pred_length, i], label="Predicted", color='blue', linewidth=1.0, alpha=0.8)
        ax[i].set_ylabel(f"Dim {i+1}", fontsize=11)
        ax[i].grid(True, alpha=0.3)
        if i == 0:
            ax[i].set_title(f"{name} - Time Series", fontsize=14, weight='bold')
            ax[i].legend()
        if i == 2:
            ax[i].set_xlabel("Time steps", fontsize=11)

    plt.tight_layout()
    plt.savefig(f"plots/{name}_timeseries.png", dpi=150, bbox_inches='tight')
    plt.close()

print("✓ All plots saved to ./plots/\n")

# ======================================================
# Timing
# ======================================================
total_time = time.time() - meta_start_time
print("="*60)
print("TIMING SUMMARY")
print("="*60)
print(f"Meta-training: {total_train_time/60:.1f} min")
print(f"Testing:       {test_total_time:.1f}s")
print(f"Total:         {total_time/60:.1f} min")
print("="*60)
print("\n✓ Complete!")