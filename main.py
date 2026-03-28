import argparse
import os
from data_loader import get_tokenized_datasets, get_dataloaders
from utils import print_comparison_table, plot_comparison, load_results, save_results

def parse_args():
    parser = argparse.ArgumentParser(description='BERT QA: Baseline vs LoRA vs Adapter comparison')
    parser.add_argument('--model', choices=['baseline', 'lora', 'adapter', 'all'], default='all', help='Which model(s) to train (default: all)')
    parser.add_argument('--train_size', type=int, default=8000, help='Number of training examples (default: 8000)')
    parser.add_argument('--val_size', type=int, default=1000, help='Number of validation examples (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (default: 16)')
    parser.add_argument('--epochs', type=int, default=2, help='Training epochs (default: 2)')
    parser.add_argument('--no_plot', action='store_true', help='Skip generating the comparison chart')
    return parser.parse_args()

def main():
    args = parse_args()
    print('=' * 62)
    print('  BERT Question Answering - Parameter Efficiency Comparison')
    print('=' * 62)
    print(f'  Training samples : {args.train_size}')
    print(f'  Val samples      : {args.val_size}')
    print(f'  Batch size       : {args.batch_size}')
    print(f'  Epochs           : {args.epochs}')
    print(f'  Model(s)         : {args.model}')
    print('=' * 62 + '\n')
    tokenizer, train_ds, val_ds, val_raw = get_tokenized_datasets(train_size=args.train_size, val_size=args.val_size)
    train_loader, _ = get_dataloaders(train_ds, val_ds, batch_size=args.batch_size)
    all_results = {}
    run_all = args.model == 'all'
    run_baseline = run_all or args.model == 'baseline'
    run_lora = run_all or args.model == 'lora'
    run_adapter = run_all or args.model == 'adapter'
    if run_baseline:
        print('\n' + '-' * 60)
        print('  [1/3] Baseline - Full Fine-Tuning')
        print('-' * 60)
        from train_baseline import train_baseline
        _, results = train_baseline(train_loader, val_ds, val_raw, epochs=args.epochs)
        all_results['Baseline BERT'] = results
    if run_lora:
        print('\n' + '-' * 60)
        print('  [2/3] LoRA - Low-Rank Adaptation')
        print('-' * 60)
        from train_lora import train_lora
        _, results = train_lora(train_loader, val_ds, val_raw, epochs=args.epochs)
        all_results['LoRA'] = results
    if run_adapter:
        print('\n' + '-' * 60)
        print('  [3/3] Adapter - Bottleneck Adapter Layers')
        print('-' * 60)
        from train_adapter import train_adapter
        _, results = train_adapter(train_loader, val_ds, val_raw, epochs=args.epochs)
        all_results['Adapter'] = results
    result_files = {'Baseline BERT': 'results/baseline_results.json', 'LoRA': 'results/lora_results.json', 'Adapter': 'results/adapter_results.json'}
    for name, path in result_files.items():
        if name not in all_results and os.path.exists(path):
            try:
                all_results[name] = load_results(path)
                print(f'Loaded cached results for {name} from {path}')
            except Exception:
                pass
    if all_results:
        display_results = {}
        for name, r in all_results.items():
            display_results[name] = {'exact_match': r.get('exact_match', 0.0), 'f1': r.get('f1', 0.0), 'training_time': r.get('training_time', 'N/A'), 'training_time_sec': r.get('training_time_sec', 0.0), 'trainable_params': r.get('trainable_params', 0)}
        print_comparison_table(display_results)
        save_results(all_results, 'results/all_results.json')
        if not args.no_plot:
            try:
                plot_comparison(display_results, save_path='results/comparison.png')
            except Exception as e:
                print(f'Could not generate chart: {e}')
    else:
        print('\nNo results to display. Run at least one model.')
    print('\nDone!')
if __name__ == '__main__':
    main()