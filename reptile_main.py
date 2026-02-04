# reptile_main.py

import os
import torch
import torch.nn as nn
import numpy as np
from rc_model import Reservoir
from torch.utils.data import DataLoader, TensorDataset
import argparse
import copy
from load_tasks import ReadTasks, ExtractTasks
import matplotlib.pyplot as plt
from utils import *
from tqdm import trange

# ======================================================
# Device handling (GPU first, CPU fallback)
# ======================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

torch.set_float32_matmul_precision("high")

# ======================================================
# Argument parsing
# ======================================================
parser = argparse.ArgumentParser("Train reptile on chaos")

parser.add_argument("--logdir", default="logdir")
parser.add_argument("--meta-iterations", type=int, default=50)
parser.add_argument("--start-meta-iteration", type=int, default=0)
parser.add_argument("--iterations", type=int, default=20)
parser.add_argument("--test-iterations", type=int, default=10)
parser.add_argument("--batch", type=int, default=128)
parser.add_argument("--meta-lr", type=float, default=1.0)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--validate-every", type=int, default=5)
parser.add_argument("--train-length", type=int, default=20000)
parser.add_argument("--test-length", type=int, default=51000)
parser.add_argument("--train-scale", type=float, default=1.0)
parser.add_argument("--noise-level", type=float, default=0.003)

parser.add_argument("--d-obs", type=int, default=3)
parser.add_argument("--embedding-time", type=int, default=1000)

args = parser.parse_args()
print(args)

# ======================================================
# Dataset helper (unchanged logic, GPU-safe)
# ======================================================
def create_dataset(data, N):
    data_N = []
    for i in range(len(data) - N):
        data_N.append(np.squeeze(np.reshape(data[i : N + i], (-1, 1))))
    return np.array(data_N)

# ======================================================
# Load tasks
# ======================================================
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

# ======================================================
# Model
# ======================================================
input_size = 3 * args.embedding_time
output_size = 3

meta_model = Reservoir(
    input_dim=input_size,
    reservoir_size=1000,
    output_dim=output_size,
    spectral_radius=0.95,
    sparsity=0.1,
    alpha=0.3,
    w_in_scale=0.1,
    device=DEVICE
)

meta_optimizer = torch.optim.SGD(meta_model.parameters(), lr=args.meta_lr)
loss_fn = nn.MSELoss()

# ======================================================
# Inner-loop learning (GPU-safe)
# ======================================================
def learning_process(model, data, optimizer, iterations):
    model.train()
    model.reset_state()

    noise = np.random.normal(0., args.noise_level, size=data.shape)
    data = np.clip(data + noise, 0, 1)

    data_N = create_dataset(data, args.embedding_time)

    X = torch.tensor(
        data_N[:args.train_length],
        dtype=DTYPE,
        device=DEVICE
    )

    Y = torch.tensor(
        data[args.embedding_time : args.train_length + args.embedding_time],
        dtype=DTYPE,
        device=DEVICE
    )

    for _ in range(iterations):
        optimizer.zero_grad()
        model.reset_state()

        states = torch.empty(
            (X.shape[0], model.n),
            device=DEVICE,
            dtype=DTYPE
        )

        for t in range(X.shape[0]):
            states[t] = model.step(X[t])

        preds = model.readout(states)
        loss = loss_fn(preds, Y)

        loss.backward()
        optimizer.step()

    return model, loss.item()

# ======================================================
# Validation (unchanged behavior)
# ======================================================
def validation_process(model, data):
    model.eval()

    noise = np.random.normal(0., args.noise_level, size=data.shape)
    data = np.clip(data + noise, 0, 1)

    data_N = create_dataset(data, args.embedding_time)

    data_train = torch.tensor(
        data_N[:args.train_length],
        dtype=DTYPE,
        device=DEVICE
    )

    target_train = torch.tensor(
        data[args.embedding_time : args.train_length + args.embedding_time],
        dtype=DTYPE,
        device=DEVICE
    )

    dataset = TensorDataset(data_train, target_train)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True)

    losses = []

    with torch.no_grad():
        for Xb, Yb in loader:
            model.reset_state()

            states = torch.empty(
                (Xb.shape[0], model.n),
                device=DEVICE,
                dtype=DTYPE
            )

            for t in range(Xb.shape[0]):
                states[t] = model.step(Xb[t])

            preds = model.readout(states)
            loss = loss_fn(preds, Yb)
            losses.append(loss.item())

    return np.mean(losses), losses

# ======================================================
# Long-term prediction (unchanged logic)
# ======================================================
def long_term_validation(model, data):
    model.eval()
    model.reset_state()

    clean_data = data.copy()

    noise = np.random.normal(0., args.noise_level, size=data.shape)
    data = np.clip(data + noise, 0, 1)

    data_N = create_dataset(data, args.embedding_time)

    data_test = torch.tensor(
        data_N[:args.test_length],
        dtype=DTYPE,
        device=DEVICE
    )

    target_test = torch.tensor(
        clean_data[args.embedding_time : args.test_length + args.embedding_time],
        dtype=DTYPE
    )

    preds = []
    init_seq = data_test[0]

    with torch.no_grad():
        for _ in range(target_test.shape[0]):
            r = model.step(init_seq)
            y = model.readout(r)
            preds.append(y.cpu().numpy())
            init_seq = torch.cat([init_seq, y], dim=0)[output_size:]

    return target_test.cpu().numpy(), np.array(preds)

# ======================================================
# Reptile meta-training
# ======================================================
task_sample_num = 5

for meta_iter in trange(args.start_meta_iteration, args.meta_iterations):
    meta_lr = args.meta_lr * (1 - meta_iter / args.meta_iterations)
    for g in meta_optimizer.param_groups:
        g["lr"] = meta_lr

    sum_gradients = None
    used = set()

    while len(used) < task_sample_num:
        model = copy.deepcopy(meta_model)
        optimizer = torch.optim.Adam(model.W_out.parameters(), lr=args.lr)

        train_data, _, name = extract_train_task.get_random_data()
        if name in used:
            continue

        print('\ntrain task: ', name)
        model_copy, _ = learning_process(model, train_data, optimizer, args.iterations)

        diff = model_copy.W_out.weight.data - meta_model.W_out.weight.data
        sum_gradients = diff if sum_gradients is None else sum_gradients + diff
        used.add(name)

    meta_model.W_out.weight.data += (meta_lr / task_sample_num) * sum_gradients

# ======================================================
# Testing + plotting (FULLY preserved)
# ======================================================
test_reals, test_preds, test_names, test_dv = [], [], [], []

for idx in range(len(extract_test_task)):
    model = copy.deepcopy(meta_model)
    optimizer = torch.optim.Adam(model.W_out.parameters(), lr=args.lr)

    train_data, test_data, name = extract_test_task.get_specific_data(idx)

    model_copy, _ = learning_process(model, train_data, optimizer, args.iterations)
    real, pred = long_term_validation(model_copy, test_data)

    dv = loss_long_term(real, pred, dt=0.04)
    print(f"task: {name}, dv: {dv}")

    test_reals.append(real)
    test_preds.append(pred)
    test_names.append(name)
    test_dv.append(dv)

# ======================================================
# Plotting (unchanged)
# ======================================================
for real, pred, name in zip(test_reals, test_preds, test_names):

    os.makedirs("plots", exist_ok=True)

    ax = plt.figure().add_subplot(projection="3d")
    ax.plot(real[:, 0], real[:, 1], real[:, 2], color="orange", label="real")
    ax.set_title(name)
    ax.legend()
    # plt.show()
    plt.savefig(f"plots/{name}_real_3d.png", dpi=150)
    plt.close()

    ax = plt.figure().add_subplot(projection="3d")
    ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], color="blue", label="pred")
    ax.set_title(name)
    ax.legend()
    # plt.show()
    plt.savefig(f"plots/{name}_pred_3d.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(3, 1, figsize=(8, 6))
    pred_length = 2000

    for i in range(3):
        ax[i].plot(pred[:pred_length, i], label="pred")
        ax[i].plot(real[:pred_length, i], label="real")

    plt.legend()
    # plt.show()
    plt.savefig(f"plots/{name}_timeseries.png", dpi=150)
    plt.close()
