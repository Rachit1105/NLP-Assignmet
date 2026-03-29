import os
import torch
from torch.optim import AdamW
from transformers import AutoModelForQuestionAnswering, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
from utils import Timer, get_device, count_parameters, print_parameter_info, save_results
from squad_evaluate import evaluate_model
MODEL_NAME = 'bert-base-uncased'
LEARNING_RATE = 0.0003
EPOCHS = 2
BATCH_SIZE = 16
WARMUP_RATIO = 0.1
SAVE_DIR = 'saved_models/lora'
RESULTS_PATH = 'results/lora_results.json'
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ['query', 'value']

def build_lora_model(model_name: str=MODEL_NAME):
    base_model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    lora_config = LoraConfig(task_type=TaskType.QUESTION_ANS, r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, target_modules=LORA_TARGET_MODULES, bias='none')
    model = get_peft_model(base_model, lora_config)
    return model

def train_lora(train_loader, val_dataset, val_raw, epochs: int=EPOCHS):
    device = get_device()
    print(f'\nLoading {MODEL_NAME} with LoRA adapters...')
    model = build_lora_model(MODEL_NAME)
    model.to(device)
    print_parameter_info(model, 'LoRA BERT')
    param_info = count_parameters(model)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f'\nStarting LoRA fine-tuning for {epochs} epoch(s)...')
    with Timer() as timer:
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
            for step, batch in enumerate(progress):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total_loss += loss.item()
                progress.set_postfix(loss=f'{total_loss / (step + 1):.4f}')
            avg_loss = total_loss / len(train_loader)
            print(f'  Epoch {epoch + 1} avg loss: {avg_loss:.4f}')
    training_time_sec = timer.elapsed
    mins, secs = divmod(int(training_time_sec), 60)
    training_time_str = f'{mins}m {secs}s'
    print(f'\nLoRA training complete in {training_time_str}')
    print('\nEvaluating LoRA model...')
    metrics = evaluate_model(model, val_dataset, val_raw, device, batch_size=BATCH_SIZE)
    results = {'model': 'LoRA', 'exact_match': metrics['exact_match'], 'f1': metrics['f1'], 'training_time': training_time_str, 'training_time_sec': training_time_sec, 'total_params': param_info['total'], 'trainable_params': param_info['trainable']}
    print(f"\n[LoRA] Exact Match : {metrics['exact_match']:.4f}")
    print(f"[LoRA] F1 Score    : {metrics['f1']:.4f}")
    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    print(f'LoRA adapters saved to {SAVE_DIR}')
    save_results(results, RESULTS_PATH)
    return (model, results)
if __name__ == '__main__':
    from data_loader import get_tokenized_datasets, get_dataloaders
    tokenizer, train_ds, val_ds, val_raw = get_tokenized_datasets()
    train_loader, _ = get_dataloaders(train_ds, val_ds, batch_size=BATCH_SIZE)
    train_lora(train_loader, val_ds, val_raw)