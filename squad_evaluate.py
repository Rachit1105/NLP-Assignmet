import collections
import numpy as np
import torch
from tqdm import tqdm
import evaluate as hf_evaluate
squad_metric = hf_evaluate.load('squad')

def compute_predictions(start_logits_all: np.ndarray, end_logits_all: np.ndarray, val_dataset, val_raw, n_best: int=20, max_answer_len: int=30):
    example_to_features = collections.defaultdict(list)
    for idx, example_id in enumerate(val_dataset['example_id']):
        example_to_features[example_id].append(idx)
    id_to_example = {ex['id']: ex for ex in val_raw}
    predictions = []
    for example_id, feature_indices in example_to_features.items():
        example = id_to_example[example_id]
        context = example['context']
        answers = []
        for feat_idx in feature_indices:
            start_logits = start_logits_all[feat_idx]
            end_logits = end_logits_all[feat_idx]
            offsets = val_dataset['offset_mapping'][feat_idx]
            start_indices = np.argsort(start_logits)[-1:-n_best - 1:-1].tolist()
            end_indices = np.argsort(end_logits)[-1:-n_best - 1:-1].tolist()
            for start_idx in start_indices:
                for end_idx in end_indices:
                    if offsets[start_idx] is None or offsets[end_idx] is None:
                        continue
                    if end_idx < start_idx:
                        continue
                    if end_idx - start_idx + 1 > max_answer_len:
                        continue
                    char_start = offsets[start_idx][0]
                    char_end = offsets[end_idx][1]
                    answers.append({'score': start_logits[start_idx] + end_logits[end_idx], 'text': context[char_start:char_end]})
        if answers:
            best = max(answers, key=lambda a: a['score'])
            predictions.append({'id': example_id, 'prediction_text': best['text']})
        else:
            predictions.append({'id': example_id, 'prediction_text': ''})
    return predictions

def evaluate_model(model, val_dataset, val_raw, device, batch_size: int=16):
    model.eval()
    model.to(device)
    cols_to_keep = ['input_ids', 'attention_mask', 'token_type_ids']
    cols_to_keep = [c for c in cols_to_keep if c in val_dataset.column_names]
    tensor_dataset = val_dataset.remove_columns([c for c in val_dataset.column_names if c not in cols_to_keep])
    tensor_dataset.set_format('torch')
    from torch.utils.data import DataLoader
    loader = DataLoader(tensor_dataset, batch_size=batch_size)
    all_start_logits = []
    all_end_logits = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            all_start_logits.append(outputs.start_logits.cpu().numpy())
            all_end_logits.append(outputs.end_logits.cpu().numpy())
    start_logits = np.concatenate(all_start_logits, axis=0)
    end_logits = np.concatenate(all_end_logits, axis=0)
    predictions = compute_predictions(start_logits, end_logits, val_dataset, val_raw)
    references = [{'id': ex['id'], 'answers': ex['answers']} for ex in val_raw]
    results = squad_metric.compute(predictions=predictions, references=references)
    return {'exact_match': round(results['exact_match'], 4), 'f1': round(results['f1'], 4)}