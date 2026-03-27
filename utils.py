import time
import torch
import psutil
import os
import json
import matplotlib.pyplot as plt
from typing import Dict, Any

def count_parameters(model) -> Dict[str, int]:
    total = sum((p.numel() for p in model.parameters()))
    trainable = sum((p.numel() for p in model.parameters() if p.requires_grad))
    return {'total': total, 'trainable': trainable}

def print_parameter_info(model, model_name: str):
    info = count_parameters(model)
    pct = 100.0 * info['trainable'] / info['total']
    print(f'\n[{model_name}] Parameters:')
    print(f"  Total      : {info['total']:,}")
    print(f"  Trainable  : {info['trainable']:,}")
    print(f"  Frozen     : {info['total'] - info['trainable']:,}")
    print(f'  Trainable% : {pct:.2f}%\n')

class Timer:

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start

    def formatted(self) -> str:
        mins, secs = divmod(int(self.elapsed), 60)
        return f'{mins}m {secs}s'

def get_device() -> torch.device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    return device

def save_results(results: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to {path}')

def load_results(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)

def print_comparison_table(results: Dict[str, Dict[str, Any]]):
    header = f"{'Model':<20} {'Exact Match':>12} {'F1 Score':>10} {'Train Time':>12} {'Trainable Params':>18}"
    sep = '-' * len(header)
    print('\n' + sep)
    print(header)
    print(sep)
    for model_name, m in results.items():
        em = f"{m.get('exact_match', 0.0):.2f}"
        f1 = f"{m.get('f1', 0.0):.2f}"
        tt = f"{m.get('training_time', '?')}"
        tp = f"{m.get('trainable_params', 0):,}"
        print(f'{model_name:<20} {em:>12} {f1:>10} {tt:>12} {tp:>18}')
    print(sep + '\n')

def plot_comparison(results: Dict[str, Dict[str, Any]], save_path: str='results/comparison.png'):
    models = list(results.keys())
    ems = [results[m].get('exact_match', 0.0) for m in models]
    f1s = [results[m].get('f1', 0.0) for m in models]
    times = [results[m].get('training_time_sec', 0.0) for m in models]
    x = range(len(models))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].bar(x, ems, color=['#4C72B0', '#DD8452', '#55A868'])
    axes[0].set_title('Exact Match Score')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=15)
    axes[0].set_ylabel('Score')
    axes[1].bar(x, f1s, color=['#4C72B0', '#DD8452', '#55A868'])
    axes[1].set_title('F1 Score')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=15)
    axes[1].set_ylabel('Score')
    axes[2].bar(x, times, color=['#4C72B0', '#DD8452', '#55A868'])
    axes[2].set_title('Training Time (sec)')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=15)
    axes[2].set_ylabel('Seconds')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f'Comparison chart saved to {save_path}')