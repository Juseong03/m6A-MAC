"""Contracts that do not require pretrained weights or native CUDA kernels."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import pandas as pd
import torch
from torch import nn
from m6a.models import m6a_mac as mac
from m6a.training.train import load_split, validate_splits, train_seed, read_config


class FakeEncoder:
    def __init__(self, identity, device):
        self.device = device
        self.model = nn.Embedding(4, 8).to(device)

    @torch.inference_mode()
    def __call__(self, sequences):
        return self.model(mac.sequence_codes(sequences, self.device))


def fake_model(**kwargs):
    with patch.object(mac, 'backbone_identity', return_value={'dimension': 8}), \
         patch.object(mac, 'FrozenEncoder', FakeEncoder), \
         patch.object(mac, 'reference_block', side_effect=lambda width: nn.Identity()):
        return mac.M6AMAC(**kwargs)


def frame(prefix):
    return pd.DataFrame({'id': [prefix + '0', prefix + '1'], 'label': [0, 1],
                         'seq101': [prefix * 50 + 'A' + 'G' * 50,
                                    prefix * 50 + 'A' + 'T' * 50]})


class StandaloneTests(unittest.TestCase):
    def test_frozen_encoder_and_head_checkpoint(self):
        model = fake_model(length=101, dimension=8, device='cpu', encoder_checkpoint='unused')
        model.train()
        self.assertFalse(model.encoder.training)
        logits = model.forward_logits(frame('C').seq101.tolist())
        logits.sum().backward()
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.encoder.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.head.parameters()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'head.pt'
            torch.save({'state_dict': model.head.state_dict()}, path)
            restored = fake_model(length=101, dimension=8, device='cpu', encoder_checkpoint='unused')
            restored.load_checkpoint(path)
            torch.save({'state_dict': model.head.state_dict(), 'length': 201}, path)
            with self.assertRaises(ValueError):
                restored.load_checkpoint(path)
            model.eval(); restored.eval()
            # Fake encoder weights are intentionally different; compare the head directly.
            for left, right in zip(model.head.parameters(), restored.head.parameters()):
                torch.testing.assert_close(left, right)
        for sequences in ([], ['N' * 101], ['C' * 101], ['A' * 100]):
            with self.assertRaises(ValueError):
                model(sequences)

    def test_data_contract_and_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'split.tsv'
            data = frame('C')
            data.to_csv(path, sep='\t', index=False)
            loaded = load_split(path, 101)
            validate_splits(loaded, frame('T'), 101)
            with self.assertRaises(ValueError):
                validate_splits(loaded, loaded, 101)
            for change in ({'label': [0, 2]}, {'id': ['same', 'same']},
                           {'seq101': ['N' * 101, 'A' * 101]}):
                data.assign(**change).to_csv(path, sep='\t', index=False)
                with self.assertRaises(ValueError):
                    load_split(path, 101)

    def test_training_updates_head_and_protects_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, validation = frame('C'), frame('T')
            for name, data in [('train', train), ('validation', validation)]:
                data.to_csv(root / (name + '.tsv'), sep='\t', index=False)
            (root / 'encoder.pt').write_bytes(b'test fixture')
            config = dict(length=101, dimension=8, device='cpu', epochs=2, micro_batch=2,
                          learning_rate=.001, weight_decay=.01, seeds=[17],
                          train=str(root / 'train.tsv'), validation=str(root / 'validation.tsv'),
                          encoder_checkpoint=str(root / 'encoder.pt'), output=str(root / 'output'))
            initial = {}
            def factory(**kwargs):
                model = fake_model(**kwargs)
                initial.update({name: value.clone() for name, value in model.head.state_dict().items()})
                return model
            with contextlib.redirect_stdout(io.StringIO()):
                train_seed(config, train, validation, root / 'output', 17, model_factory=factory)
            saved = torch.load(root / 'output/best.pt', weights_only=True)
            self.assertTrue(any(not torch.equal(value, saved['state_dict'][name]) for name, value in initial.items()))
            self.assertEqual(json.loads((root / 'output/metrics.json').read_text())['status'], 'complete')
            protocol = json.loads((root / 'output/protocol.json').read_text())
            self.assertNotIn(str(root), json.dumps(protocol))
            with self.assertRaises(FileExistsError):
                train_seed(config, train, validation, root / 'output', 17, model_factory=factory)

    def test_relocated_repository_runs_without_parent_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'relocated'
            shutil.copytree(ROOT, root, ignore=shutil.ignore_patterns('.git', '__pycache__', '*.egg-info'))
            (root / 'data').mkdir(exist_ok=True)
            frame('C').to_csv(root / 'data/train.tsv.gz', sep='\t', index=False)
            frame('T').to_csv(root / 'data/validation.tsv.gz', sep='\t', index=False)
            (root / 'encoder.pt').write_bytes(b'fixture')
            config = json.loads((root / 'configs/train.json').read_text())
            config.update(length=101, encoder_checkpoint='encoder.pt', seeds=[17], device='cpu')
            (root / 'config.json').write_text(json.dumps(config))
            result = subprocess.run([sys.executable, 'run.py', 'train', '--config', 'config.json', '--dry-run'],
                                    cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['status'], 'valid')
            self.assertFalse((root / 'runs').exists())
            for command in ('train', 'predict'):
                result = subprocess.run([sys.executable, 'run.py', command, '--help'], cwd=root,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_config_rejects_invalid_hyperparameters(self):
        base = json.loads((ROOT / 'configs/train.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            for change in ({'epochs': 0}, {'length': 401.0}, {'micro_batch': 3},
                           {'seeds': [17, 17]}, {'learning_rate': 0}, {'unknown': 1}):
                path.write_text(json.dumps({**base, **change}))
                with self.assertRaises(ValueError):
                    read_config(path)


if __name__ == '__main__':
    unittest.main()
