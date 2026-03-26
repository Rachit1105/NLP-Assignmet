from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch
MODEL_NAME = 'bert-base-uncased'
MAX_LENGTH = 384
DOC_STRIDE = 128
TRAIN_SUBSET = 8000
VAL_SUBSET = 1000

def load_squad_raw(train_size: int=TRAIN_SUBSET, val_size: int=VAL_SUBSET):
    print('Loading SQuAD dataset...')
    dataset = load_dataset('squad')
    train_raw = dataset['train']
    val_raw = dataset['validation']
    if train_size:
        train_raw = train_raw.select(range(min(train_size, len(train_raw))))
    if val_size:
        val_raw = val_raw.select(range(min(val_size, len(val_raw))))
    print(f'  Train samples : {len(train_raw)}')
    print(f'  Val   samples : {len(val_raw)}')
    return (train_raw, val_raw)

def get_tokenizer(model_name: str=MODEL_NAME):
    return AutoTokenizer.from_pretrained(model_name)

def preprocess_training_examples(examples, tokenizer, max_length=MAX_LENGTH, stride=DOC_STRIDE):
    questions = [q.strip() for q in examples['question']]
    tokenized = tokenizer(questions, examples['context'], max_length=max_length, truncation='only_second', stride=stride, return_overflowing_tokens=True, return_offsets_mapping=True, padding='max_length')
    sample_map = tokenized.pop('overflow_to_sample_mapping')
    offset_map = tokenized.pop('offset_mapping')
    start_positions = []
    end_positions = []
    for i, offsets in enumerate(offset_map):
        sample_idx = sample_map[i]
        answers = examples['answers'][sample_idx]
        if len(answers['answer_start']) == 0:
            start_positions.append(0)
            end_positions.append(0)
            continue
        ans_start_char = answers['answer_start'][0]
        ans_end_char = ans_start_char + len(answers['text'][0])
        sequence_ids = tokenized.sequence_ids(i)
        ctx_start = 0
        while ctx_start < len(sequence_ids) and sequence_ids[ctx_start] != 1:
            ctx_start += 1
        ctx_end = len(sequence_ids) - 1
        while ctx_end >= 0 and sequence_ids[ctx_end] != 1:
            ctx_end -= 1
        if offsets[ctx_start][0] > ans_end_char or offsets[ctx_end][1] < ans_start_char:
            start_positions.append(0)
            end_positions.append(0)
            continue
        token_start = ctx_start
        while token_start <= ctx_end and offsets[token_start][0] <= ans_start_char:
            token_start += 1
        start_positions.append(token_start - 1)
        token_end = ctx_end
        while token_end >= ctx_start and offsets[token_end][1] >= ans_end_char:
            token_end -= 1
        end_positions.append(token_end + 1)
    tokenized['start_positions'] = start_positions
    tokenized['end_positions'] = end_positions
    return tokenized

def preprocess_validation_examples(examples, tokenizer, max_length=MAX_LENGTH, stride=DOC_STRIDE):
    questions = [q.strip() for q in examples['question']]
    tokenized = tokenizer(questions, examples['context'], max_length=max_length, truncation='only_second', stride=stride, return_overflowing_tokens=True, return_offsets_mapping=True, padding='max_length')
    sample_map = tokenized.pop('overflow_to_sample_mapping')
    example_ids = []
    for i in range(len(tokenized['input_ids'])):
        example_ids.append(examples['id'][sample_map[i]])
        sequence_ids = tokenized.sequence_ids(i)
        tokenized['offset_mapping'][i] = [o if sequence_ids[k] == 1 else None for k, o in enumerate(tokenized['offset_mapping'][i])]
    tokenized['example_id'] = example_ids
    return tokenized

def get_tokenized_datasets(model_name: str=MODEL_NAME, train_size: int=TRAIN_SUBSET, val_size: int=VAL_SUBSET):
    tokenizer = get_tokenizer(model_name)
    train_raw, val_raw = load_squad_raw(train_size, val_size)
    print('Tokenizing training set...')
    train_dataset = train_raw.map(lambda ex: preprocess_training_examples(ex, tokenizer), batched=True, remove_columns=train_raw.column_names)
    train_dataset.set_format('torch')
    print('Tokenizing validation set...')
    val_dataset = val_raw.map(lambda ex: preprocess_validation_examples(ex, tokenizer), batched=True, remove_columns=val_raw.column_names)
    return (tokenizer, train_dataset, val_dataset, val_raw)

def get_dataloaders(train_dataset, val_dataset, batch_size: int=16):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_for_loader = val_dataset.remove_columns(['example_id', 'offset_mapping'])
    val_for_loader.set_format('torch')
    val_loader = DataLoader(val_for_loader, batch_size=batch_size)
    return (train_loader, val_loader)