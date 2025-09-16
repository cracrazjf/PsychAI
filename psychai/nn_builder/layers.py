from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Callable, Union, Optional
import torch
import torch.nn as nn

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
class LinearLayer(Layer):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    @property
    def requires(self): return ("hidden",)

    @property
    def provides(self): return ("hidden",)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        outputs = {}
        outputs["hidden"] = self.linear(inputs["hidden"])
        return outputs


@register_layer("relu")
class ReLULayer(Layer):
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)

    @property
    def requires(self): return ("hidden",)

    @property
    def provides(self): return ("hidden",)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        outputs = {}
        outputs["hidden"] = self.relu(inputs["hidden"])
        return outputs


@register_layer("gelu")
class GELULayer(Layer):
    def __init__(self):
        super().__init__()
        self.gelu = nn.GELU()

    @property
    def requires(self): return ("hidden",)

    @property
    def provides(self): return ("hidden",)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        outputs = {}
        outputs["hidden"] = self.gelu(inputs["hidden"])
        return outputs


@register_layer("tanh")
class TanhLayer(Layer):
    """
    Applies elementwise tanh activation.
    Expects and returns `hidden`.
    """
    def __init__(self):
        super().__init__()
        self.act = nn.Tanh()

    @property
    def requires(self): 
        return ("hidden",)

    @property
    def provides(self): 
        return ("hidden",)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        outputs = {}
        outputs["hidden"] = self.act(inputs["hidden"])
        return outputs


@register_layer("sigmoid")
class SigmoidLayer(Layer):
    """
    Applies elementwise sigmoid activation.
    Expects and returns `hidden`.
    """
    def __init__(self):
        super().__init__()
        self.act = nn.Sigmoid()

    @property
    def requires(self): 
        return ("hidden",)

    @property
    def provides(self): 
        return ("hidden",)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        outputs = {}
        outputs["hidden"] = self.act(inputs["hidden"])
        return outputs


@register_layer("dropout")
class DropoutLayer(Layer):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.drop = nn.Dropout(p)

    @property
    def requires(self): return ("hidden",)

    @property
    def provides(self): return ("hidden",)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        outputs = {}
        outputs["hidden"] = self.drop(inputs["hidden"])
        return outputs


@register_layer("layernorm")
class LayerNormLayer(Layer):
    def __init__(self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    @property
    def requires(self): return ("hidden",)

    @property
    def provides(self): return ("hidden",)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        outputs = {}
        if inputs.get("hidden", None) is not None:
            outputs["hidden"] = self.ln(inputs["hidden"])
        else:
            outputs["hidden"] = self.ln(inputs["embeddings"])
        if inputs.get("attention_mask", None) is not None:
            outputs["attention_mask"] = inputs["attention_mask"]
        return outputs


@register_layer("batchnorm1d")
class BatchNorm1dLayer(Layer):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)

    @property
    def requires(self): return ("hidden",)

    @property
    def provides(self): return ("hidden",)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        outputs = {}
        # BN expects (B, C) or (B, C, L) → for sequence, assume hidden = (B, C, L)
        h = inputs["hidden"]
        if h.dim() == 2:  # (B, C)
            outputs["hidden"] = self.bn(h)
        elif h.dim() == 3:  # (B, T, C) → move C to channel dim
            outputs["hidden"] = self.bn(h.transpose(1, 2)).transpose(1, 2)
        else:
            raise ValueError("Unsupported shape for BatchNorm1d")
        return outputs


@register_layer("embedding")
class EmbeddingLayer(Layer):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        kind: str = "learned",   # 'learned' or 'one_hot'
        dtype: torch.dtype = torch.float32,
        pass_attention_mask: bool = False,
    ):
        super().__init__()
        assert kind in ("learned", "one_hot"), "kind must be 'learned' or 'one_hot'"
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.kind = kind
        self.dtype = dtype
        self.pass_attention_mask = pass_attention_mask
        if self.kind == "learned":
            self.emb = nn.Embedding(self.num_embeddings, self.embedding_dim)
        else:
            embedding_weights = torch.eye(self.num_embeddings)
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
            if self.embedding_dim != self.num_embeddings:
                raise ValueError(
                    f"EmbeddingLayer(kind='one_hot') requires embedding_dim == num_embeddings "
                    f"(got embedding_dim={self.embedding_dim}, num_embeddings={self.num_embeddings}). "
                    f"If you want a projection, follow with a LinearLayer."
                )
            out = self.emb(ids)
            if inputs.get("attention_mask", None) is not None:
                mask = inputs["attention_mask"].to(dtype=out.dtype, device=out.device)
                mask = mask.unsqueeze(-1)
                out = out * mask
        outputs = {}
        outputs["embeddings"] = out
        if self.pass_attention_mask:
            outputs["attention_mask"] = inputs.get("attention_mask", None)
        return outputs


@register_layer("positional_embedding")
class PositionalEmbeddingLayer(Layer):
    def __init__(
        self,
        hidden_size: int,
        max_position_embeddings: int = 2048,
        kind: str = "learned",
        dropout: float = 0.0,
        batch_first: bool = True,
    ):
        super().__init__()
        assert kind in ("learned", "fixed"), "kind must be 'learned' or 'fixed'"
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.kind = kind
        self.batch_first = batch_first
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        if self.kind == "learned":
            self.pos_embed = nn.Embedding(max_position_embeddings, hidden_size)
        else:
            # Build sinusoidal table once and register as buffer: (max_pos, H)
            pe = torch.zeros(max_position_embeddings, hidden_size)
            position = torch.arange(0, max_position_embeddings, dtype=torch.float32).unsqueeze(1)  # (max_pos,1)
            div_term = torch.exp(
                torch.arange(0, hidden_size, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / hidden_size)
            )  # (H/2,)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pos_buffer", pe, persistent=False)  # not saved as parameter

    @property
    def requires(self):
        # 'position_ids' is optional
        return ("embeddings",)

    @property
    def provides(self):
        return ("embeddings",)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        h = inputs["embeddings"]  # (B,T,H) or (T,B,H)
        if not self.batch_first:
            h = h.transpose(0, 1)  # -> (B,T,H)

        B, T, H = h.shape
        if H != self.hidden_size:
            raise ValueError(f"PositionalEmbeddingLayer: hidden size mismatch (got {H}, expected {self.hidden_size})")
        if T > self.max_position_embeddings:
            raise ValueError(
                f"Sequence length {T} exceeds max_position_embeddings {self.max_position_embeddings}."
            )

        # Position ids: (B,T)
        if "position_ids" in inputs:
            pos_ids = inputs["position_ids"]
            if pos_ids.dim() == 1:
                pos_ids = pos_ids.unsqueeze(0).expand(B, T)  # (T,) -> (B,T)
        else:
            pos_ids = torch.arange(T, device=h.device).unsqueeze(0).expand(B, T)  # (B,T)

        # Get positional encodings: (B,T,H)
        if self.kind == "learned":
            pos = self.pos_embed(pos_ids)  # (B,T,H)
        else:
            pos = self.pos_buffer.index_select(0, pos_ids.reshape(-1)).view(B, T, H).to(h.dtype)  # (B,T,H)

        if not self.batch_first:
            h = h.transpose(0, 1)  # back to (T,B,H)

        outputs = {}
        outputs["embeddings"] = h
        if inputs.get("attention_mask", None) is not None:
            outputs["attention_mask"] = inputs["attention_mask"]
        return outputs


@register_layer("causal_mask")
class CausalMaskLayer(Layer):
    def __init__(self, max_position_embeddings: int = 4096, batch_first: bool = True):
        super().__init__()
        self.max_position_embeddings = max_position_embeddings
        self.batch_first = batch_first
        tril = torch.tril(torch.ones(max_position_embeddings, max_position_embeddings, dtype=torch.bool))
        self.register_buffer("tril_buffer", tril, persistent=False)

    @property
    def requires(self):
        return ("hidden", "q")

    @property
    def provides(self):
        return ("causal_mask",)

    def _infer_B_T(self, inputs: Dict[str, Any]) -> tuple[int, int, torch.device]:
        if "hidden" in inputs:
            h = inputs["hidden"]
            if not self.batch_first:
                h = h.transpose(0, 1)  # -> (B,T,*)
            B, T = h.shape[:2]
            return B, T, h.device
        if "q" in inputs:
            q = inputs["q"]  # (B, heads, T, Dh)
            B, _, T, _ = q.shape
            return B, T, q.device
        raise ValueError("CausalMaskLayer requires either 'hidden' or 'q' to infer batch and sequence length.")

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        B, T, device = self._infer_B_T(inputs)
        if T > self.max_position_embeddings:
            raise ValueError(
                f"CausalMaskLayer: sequence length {T} exceeds max_position_embeddings={self.max_position_embeddings}."
            )
        tril_T = self.tril_buffer[:T, :T].to(device=device)  # (T,T) bool keep-mask
        mask = tril_T.view(1, 1, T, T).expand(B, 1, T, T)    # (B,1,T,T) bool
        outputs = {**inputs}
        outputs["causal_mask"] = mask  # (B,1,T,T) bool
        return outputs


@register_layer("qkv_projection")
class QKVProjectionLayer(Layer):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        bias: bool = True,
        fused: bool = True,
        batch_first: bool = True,
    ):
        super().__init__()
        if head_dim is None:
            if hidden_size % num_heads != 0:
                raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads}) "
                                 "when head_dim is not provided.")
            head_dim = hidden_size // num_heads
        if hidden_size != num_heads * head_dim:
            raise ValueError(f"hidden_size ({hidden_size}) must equal num_heads * head_dim "
                             f"({num_heads} * {head_dim} = {num_heads * head_dim}).")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.batch_first = batch_first
        self.fused = fused
        self.sublayers = nn.ModuleDict()

        if fused:
            # single projection to 3H, then split
            self.sublayers["w_qkv"] = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        else:
            # separate projections
            self.sublayers["w_q"] = nn.Linear(hidden_size, hidden_size, bias=bias)
            self.sublayers["w_k"] = nn.Linear(hidden_size, hidden_size, bias=bias)
            self.sublayers["w_v"] = nn.Linear(hidden_size, hidden_size, bias=bias)

    @property
    def requires(self):
        return ("hidden",)

    @property
    def provides(self):
        return ("q", "k", "v")

    def _reshape_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, H) -> (B, num_heads, T, head_dim)
        """
        B, T, H = x.shape
        x = x.view(B, T, self.num_heads, self.head_dim)
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, heads, T, head_dim)
        return x

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        h = inputs.get("hidden", inputs.get("embeddings"))
        if not self.batch_first:
            h = h.transpose(0, 1)  # -> (B, T, H)

        B, T, H = h.shape
        if H != self.hidden_size:
            raise ValueError(f"QKVProjectionLayer: last dim of hidden ({H}) != hidden_size ({self.hidden_size}).")

        if self.fused:
            qkv = self.sublayers["w_qkv"](h)                      # (B, T, 3H)
            q, k, v = torch.chunk(qkv, 3, dim=-1)                 # each (B, T, H)
        else:
            q = self.sublayers["w_q"](h)                          # (B, T, H)
            k = self.sublayers["w_k"](h)                          # (B, T, H)
            v = self.sublayers["w_v"](h)                          # (B, T, H)

        q = self._reshape_to_heads(q)                # (B, heads, T, head_dim)
        k = self._reshape_to_heads(k)                # (B, heads, T, head_dim)
        v = self._reshape_to_heads(v)                # (B, heads, T, head_dim)

        outputs = {}
        outputs["q"] = q
        outputs["k"] = k
        outputs["v"] = v
        if inputs.get("attention_mask", None) is not None:
            outputs["attention_mask"] = inputs["attention_mask"]

        return outputs


@register_layer("scaled_dot_product_attention")
class ScaledDotProductAttentionLayer(Layer):

    def __init__(self, head_dim: int, dropout: float = 0.0, return_attn_weights: bool = False):
        super().__init__()
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.return_attn_weights = return_attn_weights

    @property
    def requires(self): return ("q", "k", "v",)

    @property
    def provides(self): return ("context",)

    def _expand_bt_to_b1tqt(self, mask: torch.Tensor, B: int, Tq: int, Tk: int) -> torch.Tensor:
        if mask.dim() == 2 and mask.shape == (B, Tk):
            return mask.view(B, 1, 1, Tk).expand(B, 1, Tq, Tk)
        return mask

    def _apply_mask(self, scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.dtype == torch.bool:
            return scores.masked_fill(~mask, float("-inf"))
        return scores + mask.to(dtype=scores.dtype, device=scores.device)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        q = inputs["q"]  # (B,H,Tq,Dh)
        k = inputs["k"]  # (B,H,Tk,Dh)
        v = inputs["v"]  # (B,H,Tk,Dh)
        if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
            raise ValueError("q, k, v must be 4D: (B, H, T, Dh)")
        B, H, Tq, Dh = q.shape
        Bk, Hk, Tk, Dhk = k.shape
        Bv, Hv, Tv, Dhv = v.shape
        if not (B == Bk == Bv and H == Hk == Hv and Dh == Dhk == Dhv and Tk == Tv):
            raise ValueError(f"Shape mismatch: q{tuple(q.shape)} k{tuple(k.shape)} v{tuple(v.shape)}")
        if Dh != self.head_dim:
            raise ValueError(f"head_dim={self.head_dim} but q.size(-1)={Dh}")

        # Scores: (B,H,Tq,Tk) in fp32 for stability
        scores = torch.matmul(q.float(), k.transpose(-1, -2).float()) * self.scale

        # Masks (make them broadcastable to (B,1,Tq,Tk))
        attn_mask = inputs.get("attention_mask", None)
        causal_mask = inputs.get("causal_mask", None)

        if attn_mask is not None:
            attn_mask = self._expand_bt_to_b1tqt(attn_mask, B, Tq, Tk)
            scores = self._apply_mask(scores, attn_mask)

        if causal_mask is not None:
            causal_mask = self._expand_bt_to_b1tqt(causal_mask, B, Tq, Tk)
            scores = self._apply_mask(scores, causal_mask)

        # Softmax over keys
        attn = torch.softmax(scores, dim=-1).to(dtype=q.dtype)   # (B,H,Tq,Tk)
        attn = self.drop(attn)

        # Context per head: (B,H,Tq,Dh)
        ctx_h = torch.matmul(attn, v)  # (B,H,Tq,Dh)

        # Merge heads → (B,Tq,H*Dh)
        context = ctx_h.permute(0, 2, 1, 3).contiguous().view(B, Tq, H * Dh)  # (B,Tq,H*Dh)

        outputs = {}
        outputs["context"] = context
        if self.return_attn_weights:
            outputs["attn_weights"] = attn
        return outputs


@register_layer("out_projection")
class OutProjectionLayer(Layer):
    def __init__(self, hidden_size: int, bias: bool = True, batch_first: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.proj = nn.Linear(hidden_size, hidden_size, bias=bias)

    @property
    def requires(self): 
        return ("context",)

    @property
    def provides(self): 
        return ("hidden",)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        x = inputs["context"]  # (B,T,H) or (T,B,H)
        if not self.batch_first:
            x = x.transpose(0, 1)  # -> (B,T,H)

        h = self.proj(x)  # (B,T,H)

        if not self.batch_first:
            h = h.transpose(0, 1)  # -> (T,B,H)

        outputs = {}
        outputs["hidden"] = h
        return outputs


@register_layer("save_residual")
class SaveResidualLayer(Layer):
    def __init__(self):
        super().__init__()

    @property
    def requires(self): 
        return ("hidden",)

    @property
    def provides(self): 
        return ("residual",)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        outputs = {**inputs}
        outputs["residual"] = inputs["hidden"]
        return outputs


@register_layer("residual_add")
class ResidualAddLayer(Layer):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = float(alpha)

    @property
    def requires(self): 
        return ("hidden", "residual")

    @property
    def provides(self): 
        return ("hidden",)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        h = inputs["hidden"]
        r = inputs["residual"]
        if h.shape != r.shape:
            raise ValueError(f"ResidualAddLayer: shape mismatch hidden{tuple(h.shape)} vs residual{tuple(r.shape)}")
        outputs = {}
        outputs["hidden"] = h + self.alpha * r
        return outputs


@register_layer("lstm_cell")
class LSTMCellLayer(Layer):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sublayers = nn.ModuleDict()
        self.sublayers["gates"] = LinearLayer(in_features=input_size+hidden_size, out_features=hidden_size*4, bias=True)
        with torch.no_grad():
            self.sublayers["gates"].linear.bias[self.hidden_size:2*self.hidden_size].fill_(1.0)

    @property
    def requires(self): return ("embeddings", "last_hidden")

    @property
    def provides(self): return ("hidden", "last_hidden")

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        (h_prev, c_prev) = inputs["last_hidden"]
        x = inputs["embeddings"]
        z = self.sublayers["gates"]({"hidden": torch.cat([x, h_prev], dim=-1)})["hidden"]
        i, f, g, o = z.chunk(4, dim=-1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)

        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)
        outputs = {}
        outputs["hidden"] = h_t
        outputs["last_hidden"] = (h_t, c_t)
        return outputs


@register_layer("custom_lstm")
class CustomLSTMLayer(Layer):
    def __init__(self, input_size: int, hidden_size: int, batch_first: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sublayers = nn.ModuleDict()
        self.sublayers["lstm_cell"] = LSTMCellLayer(input_size=input_size, hidden_size=hidden_size)
        self.batch_first = batch_first

    @property
    def requires(self): return ("embeddings")

    @property
    def provides(self): return ("hidden", "last_hidden")

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        x = inputs["embeddings"]
        if not self.batch_first:
            x = x.transpose(0, 1)

        B, T, _ = x.shape
        last_hc = inputs.get("last_hidden", None)
        if last_hc is None:
            last_h = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
            last_c = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            last_h, last_c = last_hc
            if last_h.dim() >= 2 and last_h.shape[0] != B:
                last_h = last_h[:B, ...].contiguous()
            if last_c.dim() >= 2 and last_c.shape[0] != B:
                last_c = last_c[:B, ...].contiguous()
        outs = []
        for t in range(T):
            x_t = x[:, t, :]
            out = self.sublayers["lstm_cell"]({"embeddings": x_t, "last_hidden": (last_h, last_c)})
            outs.append(out["hidden"].unsqueeze(1))
            last_h, last_c = out["last_hidden"]
        h = torch.cat(outs, dim=1)
        if not self.batch_first:
            h = h.transpose(0, 1)
        outputs = {}
        outputs["hidden"] = h
        outputs["last_hidden"] = (last_h, last_c)
        return outputs


@register_layer("lstm")
class LSTMLayer(Layer):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 dropout: float = 0.0, bidirectional: bool = False, batch_first: bool = True):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=dropout if num_layers > 1 else 0.0,
                            bidirectional=bidirectional, batch_first=batch_first)

    @property
    def requires(self): return ("embeddings", "last_hidden")

    @property
    def provides(self): return ("hidden", "last_hidden")

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        x = inputs["embeddings"]
        last_hidden = inputs.get("last_hidden", None)
        if last_hidden is not None:
            h, (h_n, c_n) = self.lstm(x, last_hidden)
        else:
            h, (h_n, c_n) = self.lstm(x)
        outputs = {}
        outputs["hidden"] = h
        outputs["last_hidden"] = (h_n, c_n)
        return outputs


@register_layer("elman")
class ElmanLayer(Layer):
    def __init__(self, hidden_size: int, input_size: int,
                 nonlinearity: str = "tanh", batch_first: bool = True):
        super().__init__()
        self.sublayers = nn.ModuleDict()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.sublayers["hx_linear"] = LinearLayer(in_features=input_size, out_features=hidden_size, bias=True)
        self.sublayers["hh_linear"] = LinearLayer(in_features=hidden_size, out_features=hidden_size, bias=True)

        if nonlinearity.lower() == "tanh":
            self.sublayers["act"] = TanhLayer()
        elif nonlinearity.lower() == "relu":
            self.sublayers["act"] = ReLULayer(inplace=False)
        else:
            raise ValueError("nonlinearity must be 'tanh' or 'relu'")

    @property
    def requires(self): return ("embeddings")

    @property
    def provides(self): return ("hidden", "last_hidden")

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        x = inputs["embeddings"]
        if not self.batch_first:
            x = x.transpose(0, 1)

        B, T, _ = x.shape
        H = self.hidden_size
        last_h = inputs.get("last_hidden", None)
        if last_h is None:
            last_h = torch.zeros(B, H, device=x.device, dtype=x.dtype)
        else:
            if last_h.dim() >= 2 and last_h.shape[0] != B:
                last_h = last_h[:B, ...].contiguous()
        outs = []
        for t in range(T):
            x_t = x[:, t, :]
            x_t = self.sublayers["hx_linear"]({"hidden": x_t})["hidden"]
            hh_t = self.sublayers["hh_linear"]({"hidden": last_h})["hidden"]
            h_t = x_t + hh_t
            h_t = self.sublayers["act"]({"hidden": h_t})["hidden"]
            outs.append(h_t.unsqueeze(1))              # (B,1,H)
            last_h = h_t

        h = torch.cat(outs, dim=1)                     # (B,T,H)
        if not self.batch_first:
            h = h.transpose(0, 1)                      # (T,B,H)

        return {"hidden": h, "last_hidden": last_h}


@register_layer("jordan")
class JordanLayer(Layer):
    def __init__(self, hidden_size: int, output_size: int, input_size: Optional[int] = None,
                 nonlinearity: str = "tanh", batch_first: bool = True):
        super().__init__()
        self.sublayers = nn.ModuleDict()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_first = batch_first

        # Optional input projection H_in -> H
        self.sublayers["in_linear"] = LinearLayer(in_features=input_size, out_features=hidden_size, bias=True)

        # Feedback transform from previous output y_{t-1} (O -> H)
        self.sublayers["y_linear"] = LinearLayer(in_features=output_size, out_features=hidden_size, bias=True)

        # Output projection from hidden to logits (H -> O) (uses your primitive style step-wise)
        self.sublayers["out_linear"] = LinearLayer(in_features=hidden_size, out_features=output_size, bias=True)

        # Activation
        if nonlinearity.lower() == "tanh":
            self.sublayers["act"] = TanhLayer()
        elif nonlinearity.lower() == "relu":
            self.sublayers["act"] = ReLULayer(inplace=False)
        else:
            raise ValueError("nonlinearity must be 'tanh' or 'relu'")

    @property
    def requires(self): return ("embeddings", "last_logits")

    @property
    def provides(self): return ("hidden", "logits", "last_logits")

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        x = inputs["embeddings"]  # (B,T,H_in) or (T,B,H_in)
        if not self.batch_first:
            x = x.transpose(0, 1)  # -> (B,T,H_in)

        B, T, _ = x.shape

        H, O = self.hidden_size, self.output_size
        h_t = torch.zeros(B, H, device=x.device, dtype=x.dtype)
        y_t = inputs.get("last_logits", None)
        if y_t is None:
            y_t = torch.zeros(B, O, device=x.device, dtype=x.dtype)

        h_seq, y_seq = [], []
        for t in range(T):
            x_t = x[:, t, :]    
            x_t = self.sublayers["in_linear"]({"hidden": x_t})["hidden"]
            yh  = self.sublayers["y_linear"]({"hidden": y_t})["hidden"]
            h_t = self.sublayers["act"]({"hidden": x_t + yh})["hidden"]
            y_t = self.sublayers["out_linear"]({"hidden": h_t})["hidden"]

            h_seq.append(h_t.unsqueeze(1))                # (B,1,H)
            y_seq.append(y_t.unsqueeze(1))                # (B,1,O)

        h = torch.cat(h_seq, dim=1)                       # (B,T,H)
        y = torch.cat(y_seq, dim=1)                       # (B,T,O)
        if not self.batch_first:
            h = h.transpose(0, 1)                         # (T,B,H)
            y = y.transpose(0, 1)                         # (T,B,O)

        outputs = {}
        outputs["hidden"] = h
        outputs["logits"] = y
        outputs["last_logits"] = y_t
        return outputs


@register_layer("mlp")
class MLPLayer(Layer):
    def __init__(
        self,
        hidden_size: int,
        expansion: int = 4,
        activation: str = "gelu",
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        inter = hidden_size * int(expansion)
        self.sublayers = nn.ModuleDict()
        # Linear layers (use your LinearLayer primitive)
        self.sublayers["fc1"] = LinearLayer(in_features=hidden_size, out_features=inter, bias=bias)
        self.sublayers["fc2"] = LinearLayer(in_features=inter, out_features=hidden_size, bias=bias)

        # Activation layer (use your registered activation layers)
        if activation == "relu":
            self.sublayers["act"] = ReLULayer()
        elif activation == "gelu":
            self.sublayers["act"] = GELULayer()
        elif activation == "tanh":
            self.sublayers["act"] = TanhLayer()
        elif activation == "sigmoid":
            self.sublayers["act"] = SigmoidLayer()

        # Dropouts (use your DropoutLayer primitive)
        self.sublayers["drop1"] = DropoutLayer(p=float(dropout)) if dropout and dropout > 0 else None
        self.sublayers["drop2"] = DropoutLayer(p=float(dropout)) if dropout and dropout > 0 else None

    @property
    def requires(self): 
        return ("hidden",)

    @property
    def provides(self): 
        return ("hidden",)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        x = dict(inputs)
        # Linear 1
        x = self.sublayers["fc1"](x)
        # Activation
        x = self.sublayers["act"](x)
        # Dropout 1
        if self.sublayers["drop1"] is not None:
            x = self.sublayers["drop1"](x)
        # Linear 2
        x = self.sublayers["fc2"](x)
        # Dropout 2
        if self.sublayers["drop2"] is not None:
            x = self.sublayers["drop2"](x)
        outputs = {}
        outputs["hidden"] = x
        return outputs


@register_layer("conv2d")
class Conv2dLayer(Layer):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int], str] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)

    @property
    def requires(self): 
        return ("pixel_values", "hidden")

    @property
    def provides(self): return ("hidden",)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        x = inputs.get("hidden", None)
        if x is None:
            x = inputs["pixel_values"]
        if x.dim() != 4:
            raise ValueError("Conv2dLayer expects a 4D tensor (B,C,H,W)")
        outputs = {}
        outputs["hidden"] = self.conv(x)
        return outputs
        

@register_layer("lm_head")
class LMHeadLayer(Layer):
    def __init__(self, hidden_size: int, vocab_size: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=bias)

    @property
    def requires(self): return ("hidden",)

    @property
    def provides(self): return ("logits",)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        h = inputs["hidden"]
        if h.dim() != 3:
            raise ValueError("LMHeadLayer expects 'hidden' with shape (B,T,C)")
        return {"logits": self.proj(h)}


@register_layer("classifier_head")
class ClassifierHeadLayer(Layer):
    def __init__(self, in_features: int, num_classes: int, bias: bool = True):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes, bias=bias)

    @property
    def requires(self): return ("hidden",)

    @property
    def provides(self): return ("logits",)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        h = inputs["hidden"]
        if h.dim() == 3:           # (B, T, C) -> mean over T
            feat = h.mean(dim=1)
        elif h.dim() == 4:         # (B, C, H, W) -> global avg pool
            feat = h.mean(dim=(2, 3))
        else:
            raise ValueError("ClassifierHeadLayer expects 'hidden' of shape (B,T,C) or (B,C,H,W)")
        return {"logits": self.fc(feat)}