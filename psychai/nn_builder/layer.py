from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Callable, Union, Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

LAYER_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}

def register_layer(name: str):
    def decorator(cls):
        LAYER_REGISTRY[name] = cls
        return cls
    return decorator


def build_layer(spec: Dict[str, Any]):
    layer_type = spec.get("type")
    if layer_type not in LAYER_REGISTRY:
        raise ValueError(f"Unknown layer type: {layer_type}")
    cls = LAYER_REGISTRY[layer_type]
    return cls(**{k: v for k, v in spec.items() if k != "type"})


class Layer(ABC, nn.Module):
    @property
    def requires(self) -> Tuple[str, ...]:
        raise NotImplementedError

    @property
    def provides(self) -> Tuple[str, ...]:
        raise NotImplementedError

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


@register_layer("linear")
class Linear(Layer):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    @property
    def requires(self): return ("hidden",)

    @property
    def provides(self): return ("hidden",)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        outputs = {}
        outputs["hidden"] = self.linear(inputs.get("hidden", inputs.get("embeddings")))
        return outputs


@register_layer("layer_norm")
class LayerNorm(Layer):
    def __init__(self, normalized_shape: Union[int, Tuple[int, ...]], eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape, eps=eps)

    @property
    def requires(self): return ("hidden", "embeddings")

    @property
    def provides(self): return ("hidden",)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        outputs = {}
        outputs["hidden"] = self.ln(inputs.get("hidden", inputs.get("embeddings")))
        return outputs


@register_layer("embedding")
class Embedding(Layer):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        kind: str = "learned",   # 'learned' or 'one_hot'
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        assert kind in ("learned", "one_hot"), "kind must be 'learned' or 'one_hot'"
        self.vocab_size = int(vocab_size)
        self.embed_size = int(embed_size)
        self.kind = kind
        self.dtype = dtype
        if self.kind == "learned":
            self.emb = nn.Embedding(self.vocab_size, self.embed_size)
        else:
            embedding_weights = torch.eye(self.vocab_size)
            self.emb = nn.Embedding.from_pretrained(embedding_weights, freeze=True)

    @property
    def requires(self): 
        return ("input_ids",)

    @property
    def provides(self): 
        return ("embeddings",)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        ids = inputs["input_ids"]
        if self.kind == "learned":
            out = self.emb(ids)
        else:
            if self.vocab_size != self.embed_size:
                raise ValueError(
                    f"EmbeddingLayer(kind='one_hot') requires embedding_dim == num_embeddings "
                    f"(got embedding_dim={self.embed_size}, num_embeddings={self.vocab_size}). "
                    f"If you want a projection, follow with a LinearLayer."
                )
            out = self.emb(ids)
            if inputs.get("attention_mask", None) is not None:
                mask = inputs["attention_mask"].to(dtype=out.dtype, device=out.device)
                mask = mask.unsqueeze(-1)
                out = out * mask
        return {"embeddings": out}


@register_layer("position_embedding")
class PositionEmbedding(Layer):
    def __init__(
        self,
        embed_size: int,
        block_size: int = 2048
    ):
        super().__init__()
        self.embed_size = embed_size
        self.block_size = block_size
        self.emb = nn.Embedding(block_size, embed_size)

    @property
    def requires(self):
        return ("embeddings",)

    @property
    def provides(self):
        return ("embeddings",)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        h = inputs["embeddings"]

        B, T, H = h.shape
        if H != self.embed_size:
            raise ValueError(f"PositionalEmbeddingLayer: hidden size mismatch (got {H}, expected {self.embed_size})")
        if T > self.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds block size {self.block_size}."
            )

        pos_ids = torch.arange(0, T, dtype=torch.long, device=h.device)
        
        pos = self.emb(pos_ids)

        return {"embeddings": h + pos}


@register_layer("causal_self_attention")
class CausalSelfAttention(Layer):
    def __init__(
        self,
        block_size: int,
        embed_size: int,
        num_heads: int,
        head_dim: Optional[int] = None
    ):
        super().__init__()
        if head_dim is None:
            if embed_size % num_heads != 0:
                raise ValueError(f"embed_size ({embed_size}) must be divisible by num_heads ({num_heads}) "
                                 "when head_dim is not provided.")
            head_dim = embed_size // num_heads
        if embed_size != num_heads * head_dim:
            raise ValueError(f"embed_size ({embed_size}) must equal num_heads * head_dim "
                             f"({num_heads} * {head_dim} = {num_heads * head_dim}).")

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.block_size = block_size
        self.head_dim = head_dim

        self.c_attn = nn.Linear(embed_size, 3 * embed_size)

        self.c_proj = nn.Linear(embed_size, embed_size)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.flash = hasattr(nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                        .view(1, 1, block_size, block_size))

    @property
    def requires(self):
        return ("embeddings", "hidden")

    @property
    def provides(self):
        return ("hidden",)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        h = inputs.get("embeddings", inputs.get("hidden"))

        B, T, H = h.shape
        if H != self.embed_size:
            raise ValueError(f"QKVProjectionLayer: last dim of hidden ({H}) != hidden_size ({self.embed_size}).")

        qkv = self.c_attn(h)

        q, k, v = qkv.split(self.embed_size, dim=2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        if self.flash:
            y = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, H)
        y = self.c_proj(y)

        return {"hidden": y}


@register_layer("mlp")
class MLP(Layer):
    def __init__(
        self,
        embed_size: int,
        expansion: int = 4,
        nonlinearity: str = "gelu",
        approximate: str = "tanh"
    ):
        super().__init__()
        inter = embed_size * int(expansion)
        # Linear layers (use your LinearLayer primitive)
        self.c_fc = nn.Linear(embed_size, inter)
        self.c_proj = nn.Linear(inter, embed_size)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        # Activation layer (use your registered activation layers)
        if nonlinearity == "relu":
            self.act = nn.ReLU()
        elif nonlinearity == "gelu":
            self.act = nn.GELU(approximate="tanh")
        elif nonlinearity == "tanh":
            self.act = nn.Tanh()
        elif nonlinearity == "sigmoid":
            self.act = nn.Sigmoid()

    @property
    def requires(self): 
        return ("hidden",)

    @property
    def provides(self): 
        return ("hidden",)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        x = inputs["hidden"]
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return {"hidden": x}


@register_layer("decoder_block")
class DecoderBlock(Layer):
    def __init__(self, 
                 block_size: int,
                 embed_size: int,
                 num_heads: int = 8,
                 activation: str = "gelu",
                 expansion: int = 4):
        super().__init__()
        self.ln_1 = nn.LayerNorm(normalized_shape=embed_size)
        self.ln_2 = nn.LayerNorm(normalized_shape=embed_size)
        self.attn = CausalSelfAttention(embed_size=embed_size, 
                                        num_heads=num_heads, 
                                        block_size=block_size)
        self.mlp = MLP(embed_size=embed_size, 
                        expansion=expansion, 
                        nonlinearity=activation)
    
    @property
    def requires(self): 
        return ("embeddings",)

    @property
    def provides(self): 
        return ("hidden",)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        x = inputs.get("embeddings", inputs.get("hidden"))
        x_norm = self.ln_1(x)
        att_out = self.attn({"embeddings": x_norm})["hidden"]
        x = x + att_out

        # Pre-norm MLP: x = x + MLP(LN(x))
        x_norm2 = self.ln_2(x)
        mlp_out = self.mlp({"hidden": x_norm2})["hidden"]
        x = x + mlp_out
        return {"hidden": x}


@register_layer("lstm_cell")
class LSTMCell(Layer):
    def __init__(self, input_size: int, embed_size: int):
        super().__init__()
        self.input_size = input_size
        self.embed_size = embed_size
        self.gates = nn.Linear(input_size+embed_size, embed_size*4, bias=True)
        with torch.no_grad():
            self.gates.bias[self.embed_size:2*self.embed_size].fill_(1.0)

    @property
    def requires(self): return ("embeddings", "last_hidden")

    @property
    def provides(self): return ("hidden", "last_hidden")

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        (h_t, c_t) = inputs["last_hidden"]
        x = inputs["embeddings"]
        z = self.gates(torch.cat([x, h_t], dim=-1))
        i, f, g, o = z.chunk(4, dim=-1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)

        c_t = f * c_t + i * g
        h_t = o * torch.tanh(c_t)
        return {"hidden": h_t, "last_hidden": (h_t, c_t)}


@register_layer("lstm")
class LSTM(Layer):
    def __init__(self, input_size: int, embed_size: int):
        super().__init__()
        self.input_size = input_size
        self.embed_size = embed_size
        self.cell = LSTMCell(input_size=input_size, embed_size=embed_size)

    @property
    def requires(self): return ("embeddings",)

    @property
    def provides(self): return ("hidden", "last_hidden")

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        x = inputs["embeddings"]

        B, T, _ = x.shape
        hc_t = inputs.get("last_hidden", None)
        if hc_t is None:
            h_t = torch.zeros(B, self.embed_size, device=x.device, dtype=x.dtype)
            c_t = torch.zeros(B, self.embed_size, device=x.device, dtype=x.dtype)
        else:
            h_t, c_t = hc_t
            if h_t.dim() >= 2 and h_t.shape[0] != B:
                h_t = h_t[:B, ...].contiguous()
            if c_t.dim() >= 2 and c_t.shape[0] != B:
                c_t = c_t[:B, ...].contiguous()
        hcs = []
        for t in range(T):
            x_t = x[:, t, :]
            out = self.cell({"embeddings": x_t, "last_hidden": (h_t, c_t)})
            hcs.append(out["hidden"].unsqueeze(1))
            h_t, c_t = out["last_hidden"]

        h = torch.cat(hcs, dim=1)

        return {"hidden": h, "last_hidden": (h_t, c_t)}


@register_layer("elman")
class Elman(Layer):
    def __init__(self, embed_size: int, input_size: int,
                 nonlinearity: str = "tanh"):
        super().__init__()
        self.input_size = input_size
        self.embed_size = embed_size
        self.hx = nn.Linear(input_size, embed_size, bias=True)
        self.hh = nn.Linear(embed_size, embed_size, bias=True)

        if nonlinearity.lower() == "tanh":
            self.act = nn.Tanh()
        elif nonlinearity.lower() == "relu":
            self.act = nn.ReLU(inplace=False)
        else:
            raise ValueError("nonlinearity must be 'tanh' or 'relu'")

    @property
    def requires(self): return ("embeddings")

    @property
    def provides(self): return ("hidden", "last_hidden")

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        x = inputs["embeddings"]

        B, T, _ = x.shape
        H = self.embed_size
        h_t = inputs.get("last_hidden", None)

        if h_t is None:
            h_t = torch.zeros(B, H, device=x.device, dtype=x.dtype)
        else:
            if h_t.dim() >= 2 and h_t.shape[0] != B:
                h_t = h_t[:B, ...].contiguous()

        hs = []
        for t in range(T):
            x_t = x[:, t, :]
            x_t = self.hx(x_t)
            hh_t = self.hh(h_t)
            h_t = x_t + hh_t
            h_t = self.act(h_t)
            hs.append(h_t.unsqueeze(1))

        h = torch.cat(hs, dim=1)

        return {"hidden": h, "last_hidden": h_t}


@register_layer("jordan")
class Jordan(Layer):
    def __init__(self, embed_size: int, output_size: int, input_size: Optional[int] = None,
                 nonlinearity: str = "tanh"):
        super().__init__()
        self.embed_size = embed_size
        self.output_size = output_size

        self.hx = nn.Linear(input_size, out_features=embed_size, bias=True)
        self.yh = nn.Linear(output_size, embed_size, bias=True)
        self.lm_head = nn.Linear(embed_size, output_size, bias=True)

        # Activation
        if nonlinearity.lower() == "tanh":
            self.act = nn.Tanh()
        elif nonlinearity.lower() == "relu":
            self.act = nn.ReLU(inplace=False)
        else:
            raise ValueError("nonlinearity must be 'tanh' or 'relu'")

    @property
    def requires(self): return ("embeddings", "last_logits")

    @property
    def provides(self): return ("hidden", "logits", "last_logits")

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        x = inputs["embeddings"]

        B, T, _ = x.shape

        H, O = self.embed_size, self.output_size
        h_t = torch.zeros(B, H, device=x.device, dtype=x.dtype)
        y_t = inputs.get("last_logits", None)
        if y_t is None:
            y_t = torch.zeros(B, O, device=x.device, dtype=x.dtype)

        ys = []
        for t in range(T):
            x_t = x[:, t, :]    
            x_t = self.hx(x_t)
            y_h  = self.yh(y_t)
            h_t = self.act(x_t + y_h)
            y_t = self.lm_head(h_t)

            ys.append(y_t.unsqueeze(1))

        y = torch.cat(ys, dim=1)
        return {"logits": y, "last_logits": y_t}


@register_layer("lm_head")
class LMHead(Layer):
    def __init__(self, 
                 embed_size: int, 
                 vocab_size: int, 
                 bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(embed_size, vocab_size, bias=bias)

    @property
    def requires(self): return ("hidden",)

    @property
    def provides(self): return ("logits",)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        h = inputs.get("hidden")
        if h.dim() != 3:
            raise ValueError("LMHeadLayer expects 'hidden' with shape (B,T,C)")
        return {"logits": self.proj(h)}
