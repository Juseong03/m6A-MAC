"""Input validation, checkpoint loading and inference CLI for m6A-MAC."""
from pathlib import Path
import argparse
import json
import torch
from torch import nn

def validate_length(length):
    if type(length) is not int or length not in (101, 201, 401, 801):
        raise ValueError('length must be one of 101, 201, 401, 801')
    return length


def normalize_sequences(sequences, length):
    if isinstance(sequences, str):
        sequences = [sequences]
    if not isinstance(sequences, (list, tuple)) or not sequences:
        raise ValueError('Provide a sequence string or a nonempty list/tuple of strings')
    normalized = []
    for sequence in sequences:
        if not isinstance(sequence, str):
            raise ValueError('Every sequence must be a string')
        sequence = sequence.upper().replace('U', 'T')
        if len(sequence) != length or set(sequence) - set('ACGT'):
            raise ValueError(f'Each sequence must contain exactly {length} A/C/G/T/U bases')
        if sequence[length // 2] != 'A':
            raise ValueError('The central target nucleotide must be A')
        normalized.append(sequence)
    return normalized


def load_checkpoint(module, path, expected=None):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    for key, value in (expected or {}).items():
        if key in checkpoint and checkpoint[key] != value:
            raise ValueError('Checkpoint ' + key + ' differs from the model configuration')
    state = checkpoint.get('state_dict', checkpoint)
    module.load_state_dict(state, strict=True)


def sequence_codes(sequences, device=None):
    """Encode normalized strings in A/C/G/T order without legacy factories."""
    lookup = {base: index for index, base in enumerate('ACGT')}
    return torch.tensor([[lookup[base] for base in sequence] for sequence in sequences],
                        dtype=torch.long, device=device)



def run_cli(model_class):
    """Execute a model file directly; report task-weight status explicitly."""
    parser = argparse.ArgumentParser(description=model_class.__doc__)
    parser.add_argument('--encoder-checkpoint', type=Path, required=True)
    parser.add_argument('--dimension', type=int, default=128)
    parser.add_argument('--length', type=int, choices=(101, 201, 401, 801), default=101)
    parser.add_argument('--sequence', nargs='+', help='Exact-length central-A sequences')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--backward', action='store_true', help='Check one backward pass without an optimizer step')
    args = parser.parse_args()
    sequences = args.sequence or ['C' * (args.length // 2) + 'A' + 'G' * (args.length // 2)]
    if args.backward and len(sequences) == 1:
        sequences = sequences * 2
    model = model_class(length=args.length, device=args.device, dimension=args.dimension, encoder_checkpoint=args.encoder_checkpoint)
    if args.checkpoint is not None:
        model.load_checkpoint(args.checkpoint)
    model.eval()
    with torch.no_grad():
        predictions = model(sequences).cpu().tolist()
    result = {'model': model_class.__name__, 'length': args.length,
              'device': args.device, 'checkpoint': str(args.checkpoint) if args.checkpoint else None,
              'task_weights': 'loaded' if args.checkpoint else 'random_initialization',
              'y_hat': predictions}
    if args.backward:
        model.train()
        logits = model.forward_logits(sequences)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
        loss.backward()
        result['backward_loss'] = loss.item()
        result['gradient_parameters'] = sum(p.grad is not None for p in model.parameters())
    print(json.dumps(result, ensure_ascii=False))
