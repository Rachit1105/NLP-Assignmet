import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoModelForQuestionAnswering, get_linear_schedule_with_warmup
from tqdm import tqdm
from utils import Timer, get_device, count_parameters, print_parameter_info, save_results
from squad_evaluate import evaluate_model
MODEL_NAME = 'bert-base-uncased'
LEARNING_RATE = 0.0001
EPOCHS = 2
BATCH_SIZE = 16
WARMUP_RATIO = 0.1
ADAPTER_DIM = 64
SAVE_DIR = 'saved_models/adapter'
RESULTS_PATH = 'results/adapter_results.json'

class AdapterLayer(nn.Module):

    def __init__(self, hidden_size: int, bottleneck_dim: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.down_proj = nn.Linear(hidden_size, bottleneck_dim)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(bottleneck_dim, hidden_size)
        nn.init.normal_(self.down_proj.weight, std=0.001)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.normal_(self.up_proj.weight, std=0.001)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)
        return x + residual

class BertLayerWithAdapter(nn.Module):

    def __init__(self, bert_layer, hidden_size: int, bottleneck_dim: int):
        super().__init__()
        self.bert_layer = bert_layer
        self.adapter_attn = AdapterLayer(hidden_size, bottleneck_dim)
        self.adapter_ffn = AdapterLayer(hidden_size, bottleneck_dim)

    def forward(self, *args, **kwargs):
        outputs = self.bert_layer(*args, **kwargs)
        if isinstance(outputs, tuple):
            hidden = outputs[0]
            hidden = self.adapter_attn(hidden)
            hidden = self.adapter_ffn(hidden)
            return (hidden,) + outputs[1:]
        else:
            hidden = outputs
            hidden = self.adapter_attn(hidden)
            hidden = self.adapter_ffn(hidden)
            return hidden

def insert_adapters(model, bottleneck_dim: int=ADAPTER_DIM):
    hidden_size = model.config.hidden_size
    for param in model.parameters():
        param.requires_grad = False
    for i, layer in enumerate(model.bert.encoder.layer):
        wrapped = BertLayerWithAdapter(layer, hidden_size, bottleneck_dim)
        model.bert.encoder.layer[i] = wrapped
    for param in model.qa_outputs.parameters():
        param.requires_grad = True
    return model

def train_adapter(train_loader, val_dataset, val_raw, epochs: int=EPOCHS):
    device = get_device()
    print(f'\nLoading {MODEL_NAME} with adapter layers...')
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    model = insert_adapters(model, bottleneck_dim=ADAPTER_DIM)
    model.to(device)
    print_parameter_info(model, 'Adapter BERT')
    param_info = count_parameters(model)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=LEARNING_RATE)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f'\nStarting adapter fine-tuning for {epochs} epoch(s)...')
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
    print(f'\nAdapter training complete in {training_time_str}')
    print('\nEvaluating Adapter model...')
    metrics = evaluate_model(model, val_dataset, val_raw, device, batch_size=BATCH_SIZE)
    results = {'model': 'Adapter', 'exact_match': metrics['exact_match'], 'f1': metrics['f1'], 'training_time': training_time_str, 'training_time_sec': training_time_sec, 'total_params': param_info['total'], 'trainable_params': param_info['trainable']}
    print(f"\n[Adapter] Exact Match : {metrics['exact_match']:.4f}")
    print(f"[Adapter] F1 Score    : {metrics['f1']:.4f}")
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'adapter_model.pt'))
    model.config.save_pretrained(SAVE_DIR)
    print(f'Adapter model saved to {SAVE_DIR}')
    save_results(results, RESULTS_PATH)
    return (model, results)
if __name__ == '__main__':
    from data_loader import get_tokenized_datasets, get_dataloaders
    tokenizer, train_ds, val_ds, val_raw = get_tokenized_datasets()
    train_loader, _ = get_dataloaders(train_ds, val_ds, batch_size=BATCH_SIZE)
    train_adapter(train_loader, val_ds, val_raw)