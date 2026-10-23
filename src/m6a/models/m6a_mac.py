"""m6A-MAC: complete sequence -> frozen RiNALMo-Mega -> MAC U-Net -> y_hat.

MAC combines attention, Mamba and convolution with an aligned U-Net head.
RiNALMo is an external dependency; its pretrained weights are supplied by the user.
"""
from pathlib import Path
import sys

import math
import types
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F

from .sequence_model import (normalize_sequences, validate_length,
                             sequence_codes, load_checkpoint, run_cli)



def Mamba(*args, **kwargs):
    from mamba_ssm import Mamba as NativeMamba
    return NativeMamba(*args, **kwargs)

class HymbaRMSNorm(nn.Module):
    """Root Mean Square Normalization."""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.variance_epsilon)
            return (self.weight.to(input_dtype) * x.to(input_dtype))
        else:
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * x


def circular_rel_bias(L: int, device: torch.device, slope: float = 1.0) -> torch.Tensor:
    """
    Circular relative position bias for attention.
    d_circular(i,j) = min(|i-j|, L-|i-j|)
    """
    idx = torch.arange(L, device=device)
    d = (idx[None, :] - idx[:, None]).abs()
    d = torch.minimum(d, L - d).float()
    return -slope * d


class AttentionBranch(nn.Module):
    """
    Multi-head attention with circular relative bias.
    """
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_size: int,
        circular: bool = True
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size
        self.circular = circular

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query_states: [B, L, H*D]
            key_states: [B, L, H*D]
            value_states: [B, L, H*D]
            mask: [B, L]
        Returns:
            [B, L, H*D]
        """
        bsz, q_len, _ = query_states.size()

        # Reshape to multi-head: [B, L, H, D]
        query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.attention_head_size)
        key_states = key_states.view(bsz, q_len, self.num_attention_heads, self.attention_head_size)
        value_states = value_states.view(bsz, q_len, self.num_attention_heads, self.attention_head_size)

        # [B, H, L, D]
        query_states = query_states.transpose(1, 2).contiguous()
        key_states = key_states.transpose(1, 2).contiguous()
        value_states = value_states.transpose(1, 2).contiguous()

        # Attention scores: [B, H, L, L]
        attn_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(self.attention_head_size)

        # Add circular relative bias
        if self.circular:
            bias = circular_rel_bias(q_len, query_states.device)
            attn_scores = attn_scores + bias.unsqueeze(0).unsqueeze(0)

        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill((mask == 0).view(bsz, 1, 1, q_len), -1e9)

        # Softmax + weighted sum
        attn_weights = attn_scores.softmax(dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)  # [B, H, L, D]

        # Reshape back: [B, L, H*D]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        return attn_output


class CircularCNNBranch(nn.Module):
    """
    Depthwise CNN with circular padding for local pattern extraction.
    """
    def __init__(self, d_model: int, kernel_size: int = 7, circular: bool = True):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.circular = circular

        # Depthwise convolution
        self.conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=0,
            groups=d_model
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]
        Returns:
            [B, L, D]
        """
        pad_size = (self.kernel_size - 1) // 2

        # [B, L, D] → [B, D, L]
        x = x.transpose(1, 2)

        # Apply circular or zero padding
        if self.circular:
            x = F.pad(x, (pad_size, pad_size), mode='circular')
        else:
            x = F.pad(x, (pad_size, pad_size), mode='constant', value=0)

        # Convolution
        x = self.conv(x)

        # [B, D, L] → [B, L, D]
        x = x.transpose(1, 2)

        return x


class CircMACBlock_v3(nn.Module):
    """
    CircMAC Block v3: HyMBA architecture + Circular CNN branch.

    Based on proven ThymbaBlock (HyMBA), adds circular CNN for local patterns.

    Architecture:
        Input → in_proj → split to (Q, K, V, base)
        ├─ Attention(Q, K, V) - global with circular bias
        ├─ Mamba(base) - sequential
        └─ CircularCNN(base) - local patterns
        └─ Router fusion → output

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        mamba_expand: Mamba expansion factor
        circular: Use circular features
        d_state: Mamba state dimension
        d_conv: Mamba convolution kernel
        conv_kernel_size: CNN kernel size
        use_attn: Enable attention branch (ablation)
        use_mamba: Enable mamba branch (ablation)
        use_conv: Enable conv branch (ablation)
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        mamba_expand: int = 2,
        circular: bool = True,
        d_state: int = 16,
        d_conv: int = 4,
        conv_kernel_size: int = 7,
        use_attn: bool = True,
        use_mamba: bool = True,
        use_conv: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.circular = circular

        # Ablation flags
        self.use_attn = use_attn
        self.use_mamba = use_mamba
        self.use_conv = use_conv
        self.n_active_branches = sum([use_attn, use_mamba, use_conv])

        if self.n_active_branches == 0:
            raise ValueError("At least one branch must be enabled!")

        # Input projection: QKV + base
        self.qkv_dim = d_model * 3  # Q, K, V each d_model
        self.in_proj = nn.Linear(d_model, self.qkv_dim + d_model, bias=True)

        # Branch 1: Attention
        if use_attn:
            self.self_attn = AttentionBranch(
                num_attention_heads=n_heads,
                attention_head_size=self.head_dim,
                circular=circular
            )
            self.norm_attn = HymbaRMSNorm(d_model)

        # Branch 2: Mamba
        if use_mamba:
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=mamba_expand
            )
            self.norm_mamba = HymbaRMSNorm(d_model)

        # Branch 3: Circular CNN
        if use_conv:
            self.cnn = CircularCNNBranch(
                d_model=d_model,
                kernel_size=conv_kernel_size,
                circular=circular
            )
            self.norm_cnn = HymbaRMSNorm(d_model)

        # Router: n-branch fusion (dynamic based on active branches)
        if self.n_active_branches > 1:
            self.router = nn.Sequential(
                nn.Linear(d_model * self.n_active_branches, d_model),
                nn.GELU(),
                nn.Linear(d_model, self.n_active_branches),
                nn.Softmax(dim=-1)
            )

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with ablation support.

        Args:
            x: [B, L, D]
            mask: [B, L]

        Returns:
            [B, L, D]
        """
        B, L, D = x.shape

        # Input projection
        full_proj = self.in_proj(x)  # [B, L, qkv_dim + d_model]

        # Split into Q, K, V, base
        qkv, base = torch.split(full_proj, [self.qkv_dim, self.d_model], dim=-1)
        q, k, v = torch.split(qkv, [self.d_model, self.d_model, self.d_model], dim=-1)

        # Collect active branch outputs
        branch_outputs = []

        # Branch 1: Attention
        if self.use_attn:
            attn_out = self.self_attn(q, k, v, mask)
            attn_out = self.norm_attn(attn_out)
            branch_outputs.append(attn_out)

        # Branch 2: Mamba
        if self.use_mamba:
            mamba_out = self.mamba(base)
            mamba_out = self.norm_mamba(mamba_out)
            branch_outputs.append(mamba_out)

        # Branch 3: Circular CNN
        if self.use_conv:
            cnn_out = self.cnn(base)
            cnn_out = self.norm_cnn(cnn_out)
            branch_outputs.append(cnn_out)

        # Fusion
        if self.n_active_branches == 1:
            # Single branch: no routing needed
            fused = branch_outputs[0]
        else:
            # Multi-branch: use router
            router_input = torch.cat([
                out.mean(dim=1, keepdim=True).expand(-1, L, -1)
                for out in branch_outputs
            ], dim=-1)  # [B, L, n_branches * D]

            gates = self.router(router_input)  # [B, L, n_branches]

            # Weighted fusion
            fused = sum(
                out * gates[..., i:i+1]
                for i, out in enumerate(branch_outputs)
            )

        return self.out_proj(fused)



def reference_block(dimension):
    block = CircMACBlock_v3(d_model=dimension, n_heads=8, circular=False, conv_kernel_size=7)
    block.mamba.use_fast_path = False
    return block

def signed_distance_buckets(length,device):
 relative=torch.arange(length,device=device)-length//2
 magnitude=(torch.floor(torch.log2(relative.abs().clamp(min=1).float())).long()+1).clamp(max=9)
 return 9+relative.sign()*magnitude


def unet_geometry(length, levels=2):
    """Odd-length grids produced by the aligned stride-2 U-Net path."""
    if length < 3 or length % 2 != 1:
        raise ValueError('An odd input with a central residue is required')
    sizes, padding = [length], []
    for _ in range(levels):
        pad = (sizes[-1] // 2) % 2
        padding.append(pad)
        sizes.append((sizes[-1] + 2 * pad + 1) // 2)
    return {'levels': sizes, 'symmetric_padding_before_down': padding,
            'full_path': sizes + sizes[-2::-1]}

class AlignedDown(nn.Module):
 def __init__(self,dimension):
  super().__init__();self.conv=nn.Conv1d(dimension,dimension,3,stride=2,padding=1)
 def forward(self,x):
  if x.shape[1]%2!=1:raise ValueError('Downsampling requires an odd grid')
  pad=(x.shape[1]//2)%2
  return self.conv(F.pad(x.transpose(1,2),(pad,pad))).transpose(1,2),pad


class AlignedUp(nn.Module):
 def __init__(self,dimension):
  super().__init__();self.conv=nn.ConvTranspose1d(dimension,dimension,3,stride=2,padding=1)
 def forward(self,x,pad,expected_length):
  y=self.conv(x.transpose(1,2)).transpose(1,2)
  if pad:y=y[:,pad:-pad]
  if y.shape[1]!=expected_length:raise ValueError('Decoder/skip grid mismatch')
  return y



class MACFeedForward(nn.Module):
    def __init__(self, dimension, moe, block_factory):
        super().__init__()
        if moe:
            raise ValueError('m6A-MAC uses dense FFN, without MoE')
        self.mac_norm = nn.LayerNorm(dimension)
        self.mac = block_factory(dimension)
        self.ffn_norm = nn.LayerNorm(dimension)
        self.ffn = nn.Sequential(nn.Linear(dimension, 8 * dimension), nn.GELU(),
                                 nn.Linear(8 * dimension, dimension))

    def forward(self, x):
        x = x + self.mac(self.mac_norm(x))
        return x + self.ffn(self.ffn_norm(x)), x.new_zeros(())

class MACReadout(nn.Module):
    def __init__(self, dimension, local=True, pooling=('center','local','global')):
        super().__init__()
        self.local = local
        if not pooling or len(set(pooling))!=len(pooling) or set(pooling)-{'center','local','global'}:
            raise ValueError('Poolings must be distinct center/local/global features')
        self.pooling=tuple(pooling)
        if 'global' in pooling:self.attention = nn.Linear(dimension, 1)
        if local:
            self.convolutions = nn.ModuleList([nn.Sequential(nn.Conv1d(4, 64, k, padding=k//2),
                nn.GroupNorm(8, 64), nn.GELU(), nn.Dropout(.15)) for k in [5, 11, 21]])
        width = dimension*len(pooling) + (384 if local else 0)
        self.classifier = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 256),
                                        nn.GELU(), nn.Dropout(.2), nn.Linear(256, 1))

    def forward(self, residues, codes):
        center = residues.shape[1]//2
        features=[]
        for pooling in self.pooling:
            if pooling=='center':features.append(residues[:,center])
            elif pooling=='local':features.append(residues[:,center-20:center+21].mean(1))
            else:
                weights=self.attention(residues).softmax(1)
                features.append((residues*weights).sum(1))
        if self.local:
            onehot = F.one_hot(codes.long(), 4).float().transpose(1, 2)
            for convolution in self.convolutions:
                x = convolution(onehot)
                features.extend([x[:, :, center], x[:, :, center-20:center+21].amax(-1)])
        return self.classifier(torch.cat(features, dim=-1)).flatten()

class FrozenMACHead(nn.Module):
    def __init__(self, length, input_dim=640, dimension=128, topology='unet',
                 use_skips=True, block_factory=None):
        super().__init__()
        if length not in (41, 101, 201, 401, 801) or dimension <= 0 or dimension % 8:
            raise ValueError('Supported odd length and width divisible by eight required')
        if topology not in ('unet', 'flat', 'pooling') or type(use_skips) is not bool:
            raise ValueError('Invalid topology or skip policy')
        self.length, self.input_dim = length, input_dim
        self.topology, self.use_skips = topology, use_skips
        self.projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, dimension))
        self.relative_embedding = nn.Embedding(19, dimension)
        self.head = MACReadout(dimension, local=False)
        factory = block_factory or reference_block
        def block():
            return MACFeedForward(dimension, False, factory)
        if topology == 'unet':
            self.encoder = nn.ModuleList([block(), block()])
            self.down = nn.ModuleList([AlignedDown(dimension), AlignedDown(dimension)])
            self.down_norm = nn.ModuleList([nn.LayerNorm(dimension), nn.LayerNorm(dimension)])
            self.bottleneck = nn.ModuleList([block(), block()])
            self.up = nn.ModuleList([AlignedUp(dimension), AlignedUp(dimension)])
            self.up_norm = nn.ModuleList([nn.LayerNorm(dimension), nn.LayerNorm(dimension)])
            self.decoder = nn.ModuleList([block(), block()])
        elif topology == 'flat':
            self.flat = nn.ModuleList([block() for _ in range(6)])

    def forward(self, inputs):
        h, codes = inputs['embeddings'], inputs['codes']
        if (h.ndim != 3 or h.shape[1:] != (self.length, self.input_dim)
                or codes.shape != h.shape[:2]):
            raise ValueError('Exact residue embedding/code alignment required; exclude BOS/EOS')
        if not torch.all(codes[:, self.length // 2] == 0):
            raise ValueError('Central target must be A')
        # Cached inputs must never form part of an encoder adaptation graph.
        x = self.projection(h.detach().float())
        x = x + self.relative_embedding(signed_distance_buckets(self.length, x.device))[None]
        if self.topology == 'unet':
            skips, pads = [], []
            for stage, down, norm in zip(self.encoder, self.down, self.down_norm):
                x, _ = stage(x)
                skips.append(x)
                x, pad = down(x)
                x = norm(x)
                pads.append(pad)
            for stage in self.bottleneck:
                x, _ = stage(x)
            for up, norm, stage, skip, pad in zip(self.up, self.up_norm, self.decoder,
                                                  reversed(skips), reversed(pads)):
                x = up(x, pad, skip.shape[1])
                x = norm(x + skip if self.use_skips else x)
                x, _ = stage(x)
        elif self.topology == 'flat':
            for stage in self.flat:
                x, _ = stage(x)
        return self.head(x, codes)

def import_official():
    import importlib.util
    spec = importlib.util.find_spec('rinalmo')
    if spec is None:
        raise ImportError('Install the official RiNALMo package before loading the encoder.')
    package = Path(next(iter(spec.submodule_search_locations)))
    if 'rinalmo.model.attention' not in sys.modules:
        # Keep the official non-flash implementation usable without flash-attn.
        attention = package / 'model' / 'attention.py'
        source = attention.read_text()
        source = '\n'.join(line for line in source.splitlines()
                           if not line.startswith('from flash_attn'))
        module = types.ModuleType('rinalmo.model.attention')
        module.__file__ = str(attention)
        sys.modules[module.__name__] = module
        exec(compile(source, str(attention), 'exec'), module.__dict__)
    from rinalmo.config import model_config
    from rinalmo.model.model import RiNALMo
    from rinalmo.data.alphabet import Alphabet
    return model_config,RiNALMo,Alphabet


def sdpa_forward(self,q,k,v,attn_mask=None,key_pad_mask=None):
    b=q.shape[0]
    q=self.to_q(q).view(b,-1,self.num_heads,self.c_head).transpose(1,2)
    k=self.to_k(k).view(b,-1,self.num_heads,self.c_head).transpose(1,2)
    v=self.to_v(v).view(b,-1,self.num_heads,self.c_head).transpose(1,2)
    if self.use_rot_emb:q,k=self.rotary_emb(q,k)
    mask=None
    if key_pad_mask is not None and key_pad_mask.any():mask=~key_pad_mask[:,None,None,:]
    if attn_mask is not None:raise ValueError('Unexpected attention mask')
    out=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=self.attention_dropout.p if self.training else 0.)
    return self.out_proj(out.transpose(1,2).reshape(b,-1,self.c_in)),None


def convert_state(state,model):
    converted={}
    for key,value in state.items():
        if '.mh_attn.Wqkv.' in key:
            for name,part in zip(['to_q','to_k','to_v'],value.chunk(3,0)):
                converted[key.replace('.mh_attn.Wqkv.',f'.mh_attn.mh_attn.{name}.')]=part
        elif '.mh_attn.' in key:
            converted[key.replace('.mh_attn.','.mh_attn.mh_attn.')]=value
        else:converted[key]=value
    # Original non-flash RoPE stores deterministic frequencies; flash uses nonpersistent buffer.
    for key,value in model.state_dict().items():
        if key.endswith('rotary_emb.inv_freq') and key not in converted:converted[key]=value
    return converted



def backbone_identity(length, checkpoint):
    validate_length(length)
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError('RiNALMo checkpoint does not exist: ' + str(checkpoint))
    return {'backbone': 'RiNALMo-Mega', 'dimension': 640, 'length': length,
            'checkpoint_file': str(checkpoint)}


class FrozenEncoder:
    def __init__(self, identity, device):
        self.identity, self.device = dict(identity), device
        configure, Model, Alphabet = import_official()
        config = configure('mega')
        config.model.transformer.use_flash_attn = False
        self.model = Model(config)
        self.alphabet = Alphabet(**config.alphabet)
        state = torch.load(identity['checkpoint_file'], map_location='cpu', weights_only=True)
        self.model.load_state_dict(convert_state(state, self.model), strict=True)
        for block in self.model.transformer.blocks:
            attention = block.mh_attn.mh_attn
            attention.forward = types.MethodType(sdpa_forward, attention)
        self.model.eval().requires_grad_(False).to(device)

    @torch.inference_mode()
    def __call__(self, sequences):
        tokens = torch.tensor(self.alphabet.batch_tokenize(sequences), device=self.device)
        length = self.identity['length']
        if tokens.shape[1] != length + 2:
            raise ValueError('Exactly one BOS/EOS pair required')
        if not torch.all(tokens[:, 1 + length // 2] == self.alphabet.get_idx('A')):
            raise ValueError('RiNALMo central target token mismatch')
        old_matmul = torch.backends.cuda.matmul.allow_tf32
        old_cudnn = torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            return self.model(tokens)['representation'][:, 1:1 + length].contiguous()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_matmul
            torch.backends.cudnn.allow_tf32 = old_cudnn

class M6AMAC(nn.Module):
    """Accept a string or batch of strings and return central-site probabilities.

    ``forward_logits`` supplies unnormalized scores for BCEWithLogitsLoss.
    ``forward`` supplies sigmoid probabilities, shape ``(batch_size,)``.
    Only the MAC head receives gradients. Inputs accept A/C/G/T/U in either case;
    ambiguous bases, padding, automatic cropping, and empty batches are rejected.
    """

    def __init__(self, length=801, dimension=128, device=None, encoder_checkpoint=None):
        super().__init__()
        validate_length(length)
        if type(dimension) is not int or dimension <= 0 or dimension % 8:
            raise ValueError('dimension must be a positive multiple of eight')
        self.length = length
        self.dimension = dimension
        device = torch.device(device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
        if encoder_checkpoint is None:
            raise ValueError('encoder_checkpoint is required')
        identity = backbone_identity(length, encoder_checkpoint)
        self._extractor = FrozenEncoder(identity, device)
        # Register the backbone so .to(), state_dict(), and parameter inspection work.
        self.encoder = self._extractor.model
        self.head = FrozenMACHead(length, identity['dimension'], dimension,
                                  topology='unet', use_skips=True).to(device)
        self.encoder.eval().requires_grad_(False)

    def train(self, mode=True):
        """Train the downstream head while keeping RiNALMo in evaluation mode."""
        super().train(mode)
        self.encoder.eval().requires_grad_(False)
        return self

    def _normalize_sequences(self, sequences):
        return normalize_sequences(sequences, self.length)

    def forward_logits(self, sequences):
        """Sequence -> tokens -> residue embeddings -> MAC U-Net -> logits."""
        sequences = self._normalize_sequences(sequences)
        device = next(self.head.parameters()).device
        self._extractor.device = device  # Follow subsequent model.to(device) calls.
        self.encoder.eval().requires_grad_(False)
        # Keep the FP32 extraction and FP16 representation boundary explicit.
        # Clone outside inference_mode so head autograd can save these tensors.
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            embeddings = self._extractor(sequences).clone().to(torch.float16).float()
        codes = sequence_codes(sequences, device)
        return self.head({'embeddings': embeddings, 'codes': codes})

    def forward(self, sequences):
        """Return y_hat: per-sequence central-A m6A probability in [0, 1]."""
        return torch.sigmoid(self.forward_logits(sequences))

    def load_head_checkpoint(self, path):
        """Strictly load a training checkpoint or a plain MAC-head state dict."""
        load_checkpoint(self.head, path, expected={'length': self.length, 'dimension': self.dimension})
        return self

    def load_checkpoint(self, path):
        """Common model API: load the downstream head checkpoint."""
        return self.load_head_checkpoint(path)



if __name__ == '__main__':
    run_cli(M6AMAC)
