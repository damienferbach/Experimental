#!/usr/bin/env python
"""
Modified version of nanogpt_log_losses_adana.py to use fineweb dataset
with validation evaluation and plotting.
"""

import os, yaml, socket, pathlib
import wandb
import tiktoken
import signal
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.stats as stats
import argparse
import logging
import glob
import pandas as pd
from typing import Dict, List, Any
from tqdm import tqdm
from dataclasses import dataclass
from huggingface_hub import snapshot_download
import shutil
import sys
import functools

# Try to import directly
from nanogpt_minimal import ModelConfig, TextDataset, init_train_state, train_step, count_params, GPT
from fineweb_dataset import FineWebDataset, create_fineweb_datasets
import jax
import jax.numpy as jnp
import optimizers
import optax
from flax import linen as nn
from flax.core import FrozenDict
from flax.training.train_state import TrainState

from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

#recommended flags for faster training
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_async_collectives=true '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
    '--xla_gpu_enable_highest_priority_async_stream=true '
)

LOG_STEPS_BASE = 1.01

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    vocab_size: int = 50257
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
    n_layer: int = 12
    dropout_rate: float = 0.1

# GPT-2 model size configurations
GPT2_CONFIGS = {
    'GPT2-nano': ModelConfig(
        vocab_size=50304,
        n_head=12,
        n_embd=768,
        block_size=1024,
        n_layer=12,
        dropout_rate=0.1
    ),
    'GPT2-medium': ModelConfig(
        vocab_size=50304,
        n_head=16,
        n_embd=1024,
        block_size=1024,
        n_layer=24,
        dropout_rate=0.1
    ),
    'GPT2-large': ModelConfig(
        vocab_size=50304,
        n_head=20,
        n_embd=1280,
        block_size=1024,
        n_layer=36,
        dropout_rate=0.1
    ),
    'GPT2-jumbo': ModelConfig(
        vocab_size=50304,
        n_head=25,
        n_embd=1600,
        block_size=1024,
        n_layer=48,
        dropout_rate=0.1
    )
}


def get_model_config(model_name: str) -> ModelConfig:
    """Get model configuration by name.
    
    Args:
        model_name: Name of the model ('GPT2-nano', 'GPT2-medium', 'GPT2-large', 'GPT2-jumbo')
        
    Returns:
        ModelConfig: Configuration for the specified model
        
    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name not in GPT2_CONFIGS:
        available_models = ', '.join(GPT2_CONFIGS.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available_models}")
    return GPT2_CONFIGS[model_name]

#From adamW_multi_gpu.py
def _init_train_state_sharded(config, model, key, mesh):
    """Creates a sharded training state for multi-GPU training."""
    inputs = jax.ShapeDtypeStruct(shape=(1, config["seq_len"]), dtype=jnp.int32)
    
    def init(rng, inputs):
        params = model.init(rng)
        # Initialize DANA optimizer
        g1 = optimizers.powerlaw_schedule(1.0, 0.0, 0.0, 1)
        g2 = optimizers.powerlaw_schedule(config["dana_g2"], 0.0, 0.0, 1)
        g3 = optimizers.powerlaw_schedule(config["dana_g3_iv"] * config["dana_g2"], 0.0, config["dana_g3_p"], 1)
        Delta = optimizers.powerlaw_schedule(1.0, 0.0, -1.0, config["dana_delta"])
        dana = optimizers.dana_optimizer(g1=g1, g2=g2, g3=g3, Delta=Delta)
        if config["optimizer"] == "dana":
            #sched = optax.schedules.warmup_constant_schedule(init_value=0, peak_value=config['learning_rate'], warmup_steps=config['warmup'])
            optimizer = optax.chain(
                optax.clip_by_global_norm(config['grad_clip']),
                dana,
                optax.add_decayed_weights(config['weight_decay'] * config['dana_g2'])
                #optax.scale_by_learning_rate(sched)
            )
        
        return TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optimizer)
    
    params_shape = jax.eval_shape(init, key, inputs)
    shardings = nn.get_sharding(params_shape, mesh)
    state = jax.jit(init, out_shardings=shardings)(key, inputs)
    return shardings, state

def train_step_sharded(state: TrainState, x: jnp.ndarray, y: jnp.ndarray, mesh: Mesh):
    """Sharded training step for multi-GPU data parallelism."""
    # Add sharding constraints for input data
    x = jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P("data")))
    y = jax.lax.with_sharding_constraint(y, NamedSharding(mesh, P("data")))
    
    def loss_fn(params: FrozenDict) -> jnp.ndarray:
        logits = state.apply_fn(params, x, False)
        # Loss computation in float32 for numerical stability
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        return loss

    loss, grads = jax.value_and_grad(loss_fn, has_aux=False)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return loss, new_state

def eval_step_sharded(state: TrainState, x: jnp.ndarray, y: jnp.ndarray, mesh: Mesh):
    """Sharded evaluation step for multi-GPU data parallelism (no gradient computation)."""
    # Add sharding constraints for input data
    x = jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P("data")))
    y = jax.lax.with_sharding_constraint(y, NamedSharding(mesh, P("data")))
    
    # Forward pass only - no gradients
    logits = state.apply_fn(state.params, x, False)
    # Loss computation in float32 for numerical stability
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    return loss

def parse_args():
    # First define all arguments with hardcoded defaults
    parser = argparse.ArgumentParser(description="Train nanogpt with fineweb dataset")
    parser.add_argument(
        "--train_steps", type=int, default=10000,
        help="Number of training steps"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Training batch size"
    )
    parser.add_argument(
        "--seq_len", type=int, default=1024,
        help="Sequence length for training"
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=64,
        help="Validation batch size (comparable to Karpathy's)"
    )
    parser.add_argument(
        "--val_max_tokens", type=int, default=None,
        help="Maximum tokens to load for validation (default: val_batch_size * seq_len * 30)"
    )
    parser.add_argument(
        "--grad_clip", type=float, default=2.0,
        help="Gradient clipping value"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0,
        help="Weight decay value"
    )
    parser.add_argument(
        "--init_std", type=float, default=0.02,
        help="Weight initialization standard deviation"
    )
    parser.add_argument(
        "--results_dir", type=str, default="results",
        help="Directory to store results"
    )
    # Add Dana hyperparameters
    parser.add_argument(
        "--dana_g2", type=float, default=1.0,
        help="DANA G2 parameter"
    )
    parser.add_argument(
        "--dana_delta", type=float, default=8.0,
        help="DANA Delta parameter"
    )
    parser.add_argument(
        "--dana_g3_iv", type=float, default=0.2,
        help="(dana_g3_iv) * (dana_g2) represents DANA G3 initial value to keep the ratio of g3 to g2 constant"
    )
    parser.add_argument(
        "--dana_g3_sv", type=float, default=0.0,
        help="DANA G3 saturation value"
    )
    parser.add_argument(
        "--dana_g3_p", type=float, default=-0.8,
        help="DANA G3 power"
    )
    parser.add_argument(
        "--dana_g3_ts", type=float, default=1.0,
        help="DANA G3 time scale"
    )
    parser.add_argument(
        "--optimizer", type=str, default="dana",
        choices=["dana", "rmsprop", "rmsprop_dana", "adam"],
        help="Optimizer to use: dana, rmsprop, rmsprop_dana or adam"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1.0,
        help="Learning rate for RMSprop"
    )
    parser.add_argument(
        "--beta_2", type=float, default=0.999,
        help="beta_2 for RMSprop"
    )
    parser.add_argument(
        "--wandb", type=bool, default=False,
        help="Whether to use wandb"
    )
    parser.add_argument(
        "--data_root", type=str, default="~/scratch/fineweb/sample/10BT",
        help="Data root directory"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="~/scratch/checkpoints",
        help="Checkpoint directory"
    )
    parser.add_argument(
        "--bias_correction", type=bool, default=False,
        help="Whether to add bias correction in rms prop update"
    )
    parser.add_argument(
        "--warmup", type=int, default=0,
        help="Warmup steps for learning rate (linear increase from 0 to learning_rate)"
    )
    parser.add_argument(
        "--model", type=str, default="GPT2-nano",
        help="Model to use: GPT2-nano, GPT2-medium, GPT2-large, GPT2-jumbo"
    )
    # Parse command line args first
    args = parser.parse_args()
    
    # Then load YAML config and override defaults if not specified in command line
    machines = os.getenv("CLUSTER") or socket.gethostname().split('.')
    if 'mila' in machines:
        cluster = 'mila'
    elif 'tamia' in machines:
        cluster = 'tamia'
    else:
        print('Unknown cluster')
        raise ValueError('Unknown cluster')
    cfg_file = pathlib.Path(__file__).parent / "configs" / f"{cluster}.yaml"
    with open(cfg_file) as f:
        print(f"Loading config from {cfg_file}")
        yaml_cfg = yaml.safe_load(f)
        
    # Override defaults with YAML values if they exist and weren't specified in command line
    for arg in vars(args):
        if arg in yaml_cfg and arg not in sys.argv:
            setattr(args, arg, yaml_cfg[arg])
            
    return args

def evaluate_validation_loss(state, val_dataset, config, eval_step_fn, val_steps=20):
    """Evaluate validation loss with multi-GPU support"""
    total_loss = 0.0
    steps_taken = 0
    
    # Create a fresh iterator each time we evaluate validation loss, which will be sharded across devices
    val_iterator = val_dataset.iterate_once(config["val_batch_size"], config["seq_len"])
    
    for x, y, w in val_iterator:
        if steps_taken >= val_steps:
            break
            
        loss = eval_step_fn(state, x, y)  # Forward pass only for validation
        total_loss += loss
        steps_taken += 1
    
    if steps_taken == 0:
        return float('inf')  # Return inf if no validation data
    
    return total_loss / steps_taken

def create_learning_curve_plot(step_numbers, train_losses, val_losses, num_params, config):
    """Create a log-log plot of the learning curve with both training and validation losses."""
    # Convert to numpy arrays
    steps = np.array(step_numbers)
    train_loss_values = np.array(train_losses)
    val_loss_values = np.array(val_losses)
    
    # Calculate tokens processed
    tokens_per_step = config["batch_size"] * config["seq_len"]
    tokens = steps * tokens_per_step
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot both training and validation curves
    ax.loglog(tokens, train_loss_values, 'o-', color='blue', label='Training Loss', markersize=4)
    ax.loglog(tokens, val_loss_values, 's-', color='red', label='Validation Loss', markersize=4)
    
    # Fit power law to training data if we have enough points
    if len(tokens) > 4 and tokens[-1] >= 1e6:
        # Find index where tokens first exceeds 1M
        start_idx = np.where(tokens >= 1e6)[0][0]
        # Linear fit in log-log space from 1M tokens onwards
        log_tokens = np.log(tokens + 1)
        log_train_loss = np.log(train_loss_values)
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_tokens[start_idx:], log_train_loss[start_idx:])
        # Power law parameters: loss = A * tokens^beta
        A = np.exp(intercept)
        beta = slope
        fit_loss = A * ((tokens + 1) ** beta)
        ax.loglog(tokens, fit_loss, '--', color='green', label=f'Training Fit: {A:.4f} t^({beta:.4f})', alpha=0.7)
    
    # Set axis labels and title
    ax.set_xlabel('Training Tokens')
    ax.set_ylabel('Loss')
    ax.set_ylim(min(min(train_loss_values), min(val_loss_values)) * 0.9, 12)
    ax.set_title(f'NanoGPT Language Model Learning Curve (FineWeb)\n{num_params:,} parameters')
    
    # Convert x-axis to better format
    def format_tokens(x, pos):
        if x >= 1e6:
            return f'{x/1e6:.1f}M'
        elif x >= 1e3:
            return f'{x/1e3:.1f}K'
        else:
            return f'{x:.0f}'
    
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_tokens))
    
    # Add grid
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Add legend
    ax.legend()
    
    # Generate a more descriptive filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = (
        f"{config['results_dir']}/nanogpt_dana_fineweb_learning_curve_{timestamp}_"
        f"steps_{config['train_steps']}_bs_{config['batch_size']}_"
        f"seq_{config['seq_len']}_"
        f"g2_{config['dana_g2']}_delta_{config['dana_delta']}_g3iv_{config['dana_g3_iv']}_"
        f"g3sv_{config['dana_g3_sv']}_g3p_{config['dana_g3_p']}_g3ts_{config['dana_g3_ts']}.pdf"
    )
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Learning curve plot saved as {output_filename}")

def modify_nanogpt_for_fineweb():
    """
    Train NanoGPT with fineweb dataset and validation evaluation.
    """
    args = parse_args()

    # Log JAX device information
    logger.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
    logger.info('JAX local devices: %r', jax.local_devices())
    logger.info('Total devices available: %d', jax.device_count())
    
    # Validate batch size is divisible by device count
    if args.batch_size % jax.device_count() != 0:
        raise ValueError(f"Batch size ({args.batch_size}) must be divisible by the number of devices ({jax.device_count()})")
    
    if args.val_batch_size % jax.device_count() != 0:
        raise ValueError(f"Validation batch size ({args.val_batch_size}) must be divisible by the number of devices ({jax.device_count()})")
    
    # Calculate per-device batch sizes
    per_device_batch_size = args.batch_size // jax.device_count()
    per_device_val_batch_size = args.val_batch_size // jax.device_count()
    
    logger.info(f"Total batch size: {args.batch_size}, per-device batch size: {per_device_batch_size}")
    logger.info(f"Total validation batch size: {args.val_batch_size}, per-device validation batch size: {per_device_val_batch_size}")
    
    # Create device mesh for data parallelism
    mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), ("data",))
    logger.info(f"Created device mesh: {mesh}")
    
    # Create JIT-compiled train step function with mesh frozen
    train_step_fn = jax.jit(functools.partial(train_step_sharded, mesh=mesh))
    # Create JIT-compiled eval step function with mesh frozen
    eval_step_fn = jax.jit(functools.partial(eval_step_sharded, mesh=mesh))

    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Configuration dictionary
    # Set default val_max_tokens if not specified
    val_max_tokens = args.val_max_tokens
    if val_max_tokens is None:
        # Default to enough tokens for 30 validation batches (with some buffer)
        val_max_tokens = args.val_batch_size * args.seq_len * 30
    
    config = {
        "train_steps": args.train_steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "val_batch_size": args.val_batch_size,
        "val_max_tokens": val_max_tokens,
        "grad_clip": args.grad_clip,
        "weight_decay": args.weight_decay,
        "init_std": args.init_std,
        "results_dir": args.results_dir,
        "dana_g2": args.dana_g2,
        "dana_delta": args.dana_delta,
        "dana_g3_iv": args.dana_g3_iv,
        "dana_g3_sv": args.dana_g3_sv,
        "dana_g3_p": args.dana_g3_p,
        "dana_g3_ts": args.dana_g3_ts,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "beta_2": args.beta_2,
        "wandb": args.wandb,
        "data_root": args.data_root,
        "checkpoint_dir": args.checkpoint_dir,
        "bias_correction": args.bias_correction,
        "warmup": args.warmup,
        "model": args.model,
        "num_devices": jax.device_count()
    }

    DATA_ROOT     = pathlib.Path(config["data_root"])
    CHECKPOINT_DIR = pathlib.Path(config["checkpoint_dir"])
    
    # Create LOG_STEPS
    LOG_STEPS = jnp.unique(jnp.concatenate([
        jnp.array([0]),
        jnp.int32(LOG_STEPS_BASE**jnp.arange(1, jnp.ceil(jnp.log(config["train_steps"])/jnp.log(LOG_STEPS_BASE)))),
        jnp.array([config["train_steps"]])
    ]))
    
    # Initialize model
    key = jax.random.PRNGKey(0)
    model = GPT(get_model_config(config["model"]))
    params = model.init(key)
    num_params = count_params(params)
    print(f"Training model: {config['model']}, Number of parameters: {num_params:,}")
    
    # Initialize sharded train state
    shardings, state = _init_train_state_sharded(config, model, key, mesh)
    num_params = count_params(state.params)
    
    # Get parquet files and split for train/val
    #TAMIA/MILA CLUSTER
    data_root = os.path.expanduser(DATA_ROOT)

    # parquet_files = sorted(glob.glob(os.path.join(data_root, "*_00000.parquet")))
    
    # if not parquet_files:
    #     raise ValueError(f"No parquet files found in {data_root}")
        
    # logger.info(f"Found {len(parquet_files)} parquet files")
    
    # # Reserve the last file for validation, rest for training
    # val_files = parquet_files[-1:]
    # train_files = parquet_files[:-1]
    
    # logger.info(f"Using {len(train_files)} files for training")
    # logger.info(f"Using {len(val_files)} files for validation")
    
    # # Initialize datasets
    # train_dataset = FineWebDataset(train_files)
    # val_dataset = FineWebDataset(val_files, max_tokens=config["val_max_tokens"], is_validation=True)
    train_dataset, val_dataset = create_fineweb_datasets(
            data_root, 
            val_max_tokens=config["val_max_tokens"],
            val_files_count=1
        )
    logger.info(f"Validation dataset limited to {config['val_max_tokens']:,} tokens")
    
    # Create training iterator
    train_iterator = train_dataset.iterate_once(config["batch_size"], config["seq_len"])
    
    # Storage for losses and metrics
    metrics_history = {
        'step': [],
        'train_loss': [],
        'val_loss': [],
        'tokens_processed': [],
        'time_elapsed': []
    }
    
    # Training loop with loss logging
    pbar = tqdm(range(config["train_steps"]), desc="Training")
    start_time = time.time()

    run_name = f"gpt2_dana_fineweb_steps_{config['train_steps']}_bs_{config['batch_size']}_seq_{config['seq_len']}_g2_{config['dana_g2']}_g3iv_{config['dana_g3_iv']}_g3p_{config['dana_g3_p']}_wd_{config['weight_decay']}"
    if config["wandb"]:
        wandb.init(project="gpt2-fineweb", 
                    name = run_name, 
                    config=config)
        config = wandb.config
    
    for step in pbar:
        # Get next batch
        x, y, w = next(train_iterator)
        
        # Forward and backward pass
        loss, state = train_step_fn(state, x, y)
        
        # Update progress bar
        pbar.set_postfix(loss=f"{loss:.4f}")
        
        if config["wandb"]:
            wandb.log({
                "step": step,
                "train_loss": float(loss)
            })
        
        # Log metrics at specified steps
        if step in LOG_STEPS:
            # Evaluate validation loss
            val_loss = evaluate_validation_loss(state, val_dataset, config, eval_step_fn)

            total_tokens = step * config["batch_size"] * config["seq_len"]
            metrics_history['step'].append(step)
            metrics_history['train_loss'].append(float(loss))
            metrics_history['val_loss'].append(float(val_loss))
            metrics_history['tokens_processed'].append(total_tokens)
            metrics_history['time_elapsed'].append(time.time() - start_time)
            
            if config["wandb"]:
                wandb.log({
                    "val_loss": float(val_loss),
                    "tokens_processed": total_tokens,
                    "time_elapsed": time.time() - start_time,
                    "tokens_per_second": total_tokens / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
                })
            
            # Print detailed metrics
            elapsed = time.time() - start_time
            average_tokens_per_second = total_tokens / elapsed
            tqdm.write(f"\nStep: {step}/{config['train_steps']} ({100.0 * step / config['train_steps']:.1f}%)")
            tqdm.write(f"  Train Loss: {loss:.6f}")
            tqdm.write(f"  Val Loss: {val_loss:.6f}")
            tqdm.write(f"  Time: {elapsed:.2f}s ({elapsed/60:.2f}m)")
            tqdm.write(f"  Tokens: {total_tokens:,} ({average_tokens_per_second:.1f} tokens/s)")
            tqdm.write(f"  G2: {config['dana_g2']}, G3_iv: {config['dana_g3_iv']}, g3p: {config['dana_g3_p']}\n")
    
    if config["wandb"]:
        wandb.finish()
    
    # Create CHECKPOINTS directory in scratch
    #TAMIA/MILA CLUSTER
    checkpoint_dir = os.path.expanduser(CHECKPOINT_DIR)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save model parameters
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"model_dana_fineweb_step_{config['train_steps']}_"
        f"bs_{config['batch_size']}_"
        f"seq_{config['seq_len']}_"
        f"wd_{config['weight_decay']}_"
        f"g2_{config['dana_g2']}_g3iv_{config['dana_g3_iv']}_g3p_{config['dana_g3_p']}.pkl"
    )

    checkpoint_data = {
            'params': state.params,
            'config': config,
            'num_params': num_params,
            'optimizer_type': 'adamw',
            'precision': 'mixed_bfloat16_rope',
            'multi_gpu': True,
            'num_devices': jax.device_count(),
            'final_train_loss': float(loss),
            'final_val_loss': float(val_loss) if 'val_loss' in locals() and not config["disable_validation"] else None
        }
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print(f"Saved checkpoint to {checkpoint_path}")
    
    # Create log-log plot of losses
    create_learning_curve_plot(
        metrics_history['step'],
        metrics_history['train_loss'],
        metrics_history['val_loss'],
        num_params,
        config
    )
    
    # Save metrics history with configuration
    metrics_data = {
        'metrics': metrics_history,
        'config': config,
        'num_params': num_params
    }
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    metrics_filename = os.path.join(
        config['results_dir'],
        f"nanogpt_dana_fineweb_metrics_{timestamp}_"
        f"steps_{config['train_steps']}_bs_{config['batch_size']}_"
        f"seq_{config['seq_len']}_"
        f"wd_{config['weight_decay']}_"
        f"g2_{config['dana_g2']}_delta_{config['dana_delta']}_g3iv_{config['dana_g3_iv']}_"
        f"g3sv_{config['dana_g3_sv']}_g3p_{config['dana_g3_p']}_g3ts_{config['dana_g3_ts']}.pkl"
    )
    
    with open(metrics_filename, 'wb') as f:
        pickle.dump(metrics_data, f)
    
    print(f"Saved metrics data to {metrics_filename}")
    
    return metrics_history, config, num_params

if __name__ == "__main__":
    modify_nanogpt_for_fineweb()
