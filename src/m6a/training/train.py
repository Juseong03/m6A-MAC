"""Train the MAC head while keeping the RiNALMo encoder frozen."""
import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from m6a.models import M6AMAC
from m6a.models.sequence_model import normalize_sequences, validate_length


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_split(path, length):
    frame = pd.read_csv(path, sep='\t', dtype={'id': str})
    required = {'id', 'label', f'seq{length}'}
    if length in (41, 401) and f'seq{length}' not in frame and 'seq801' in frame:
        # 41- and 401-nt windows are the centred slices of the stored 801-nt window.
        long = normalize_sequences(frame.seq801.tolist(), 801)
        start = (801 - length) // 2
        frame[f'seq{length}'] = [sequence[start:start + length] for sequence in long]
    if not required.issubset(frame):
        raise ValueError('Required TSV columns: ' + ', '.join(sorted(required)))
    if frame.empty or frame.id.isna().any() or frame.id.duplicated().any():
        raise ValueError('Each split requires nonempty, unique candidate IDs')
    if set(frame.label) != {0, 1}:
        raise ValueError('Each split must contain both binary labels, 0 and 1')
    frame[f'seq{length}'] = normalize_sequences(frame[f'seq{length}'].tolist(), length)
    return frame


def validate_splits(train, validation, length):
    for column in ('id', f'seq{length}', 'gene'):
        if column in train and column in validation:
            if set(train[column].dropna()) & set(validation[column].dropna()):
                raise ValueError('Train/validation overlap in ' + column)


def validation_scores(model, frame, length, batch_size):
    model.eval()
    probabilities = []
    with torch.no_grad():
        sequences = frame[f'seq{length}'].tolist()
        for start in range(0, len(sequences), batch_size):
            probabilities.extend(model(sequences[start:start + batch_size]).cpu().tolist())
    labels = frame.label.to_numpy()
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    threshold = float(thresholds[np.flatnonzero(f1 == f1.max())[-1]])
    return probabilities, {'average_precision': float(average_precision_score(labels, probabilities)),
                           'auroc': float(roc_auc_score(labels, probabilities)),
                           'threshold': threshold}


def train_seed(config, train, validation, output, seed, model_factory=M6AMAC):
    # mkdir without exist_ok protects completed and interrupted runs alike.
    output.mkdir(parents=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = model_factory(length=config['length'], dimension=config['dimension'],
                          encoder_checkpoint=config['encoder_checkpoint'], device=config['device'])
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=.5,
                                                          patience=2, min_lr=1e-6)
    protocol = {key: value for key, value in config.items()
                if key not in ('train', 'validation', 'encoder_checkpoint', 'output', 'device')}
    protocol.update(seed=seed, split_sha256={name: sha256(config[name]) for name in ('train', 'validation')},
                    encoder_sha256=sha256(config['encoder_checkpoint']),
                    selection='validation average precision', effective_batch_size=256,
                    feature_precision='float32 extraction -> float16 -> float32 head input')
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    best, stale, history = -float('inf'), 0, []
    sequences = train[f"seq{config['length']}"].tolist()
    labels = train.label.to_numpy()
    device = torch.device(config['device'])
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        order = np.random.permutation(len(train))
        total_loss = 0.0
        for start in range(0, len(order), 256):
            group = order[start:start + 256]
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0, len(group), config['micro_batch']):
                indices = group[offset:offset + config['micro_batch']]
                logits = model.forward_logits([sequences[index] for index in indices])
                target = torch.tensor(labels[indices], dtype=torch.float32, device=device)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction='sum')
                (loss / len(group)).backward()
                total_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.0)
            optimizer.step()
        probabilities, scores = validation_scores(model, validation, config['length'], config['micro_batch'])
        history.append({'epoch': epoch, 'train_loss': total_loss / len(train), **scores})
        scheduler.step(scores['average_precision'])
        if scores['average_precision'] > best:
            best, stale = scores['average_precision'], 0
            temporary = output / 'best.pt.part'
            torch.save({'state_dict': model.head.state_dict(), 'length': config['length'],
                        'dimension': config['dimension'], 'epoch': epoch, **scores}, temporary)
            temporary.replace(output / 'best.pt')
            pd.DataFrame({'id': validation.id, 'label': validation.label,
                          'probability': probabilities}).to_csv(output / 'validation_predictions.tsv', sep='\t', index=False)
        else:
            stale += 1
        (output / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
        print(json.dumps({'seed': seed, **history[-1]}), flush=True)
        if stale >= 5:
            break
    (output / 'metrics.json').write_text(json.dumps({'status': 'complete', 'best_validation_ap': best}, indent=2) + '\n')


DEFAULTS = {'length': 201, 'dimension': 128, 'seeds': [17, 29, 43, 61, 79],
            'device': 'cuda:0', 'epochs': 120, 'micro_batch': 8,
            'learning_rate': 0.0003, 'weight_decay': 0.01}
REQUIRED = {'train', 'validation', 'encoder_checkpoint', 'output'}


def read_config(path):
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict) or set(raw) - (set(DEFAULTS) | REQUIRED) or not REQUIRED.issubset(raw):
        raise ValueError('Config must contain train, validation, encoder_checkpoint, output and supported training settings')
    config = {**DEFAULTS, **raw}
    validate_length(config['length'])
    for key in ('dimension', 'epochs', 'micro_batch'):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(key + ' must be a positive integer')
    if config['dimension'] % 8 or 256 % config['micro_batch']:
        raise ValueError('dimension must divide into eight heads; micro_batch must divide 256')
    if not isinstance(config['seeds'], list) or not config['seeds'] or any(type(s) is not int or not 0 <= s < 2**32 for s in config['seeds']) or len(set(config['seeds'])) != len(config['seeds']):
        raise ValueError('seeds must be distinct nonnegative 32-bit integers')
    for key in ('learning_rate', 'weight_decay'):
        value = config[key]
        if type(value) not in (int, float) or not np.isfinite(value) or value < 0 or (key == 'learning_rate' and value == 0):
            raise ValueError('Invalid ' + key)
    for key in REQUIRED:
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(key + ' must be a nonempty path string')
    torch.device(config['device'])
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true', help='Validate inputs without loading the encoder or starting training')
    args = parser.parse_args(argv)
    config = read_config(args.config)
    train = load_split(config['train'], config['length'])
    validation = load_split(config['validation'], config['length'])
    validate_splits(train, validation, config['length'])
    if not Path(config['encoder_checkpoint']).is_file():
        raise FileNotFoundError('Missing encoder checkpoint')
    destinations = [Path(config['output']) / f'seed{seed}' for seed in config['seeds']]
    if any(path.exists() or path.is_symlink() for path in destinations):
        raise FileExistsError('Choose a fresh output directory; existing seed outputs are protected')
    if args.dry_run:
        print(json.dumps({'status': 'valid', 'train_rows': len(train), 'validation_rows': len(validation),
                          'seeds': config['seeds'], 'launches_training': False}))
        return
    for seed, output in zip(config['seeds'], destinations):
        train_seed(config, train, validation, output, seed)


if __name__ == '__main__':
    main()
