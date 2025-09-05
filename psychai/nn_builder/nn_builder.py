from typing import Dict, Any, Tuple, Optional, Set
import torch
import torch.nn as nn
from .layers import build_layer
import os
import json
import hashlib
import sys
import platform
import datetime


class ModelSpec:
    def __init__(self, vocab_size: int = None, image_shape: Tuple[int, int, int] = None):
        self.vocab_size = vocab_size
        self.image_shape = image_shape
        self.layers: list[dict] = []
        self._names: set[str] = set()

    def add_layer(self, layer_spec: Dict[str, Any], name: str = None):
        if "type" not in layer_spec:
            raise ValueError("layer_spec must include a 'type' key.")
        if name is None:
            name = f"{layer_spec['type']}_{len(self.layers)}"
        if name in self._names:
            raise ValueError(f"Duplicate layer name: {name}")
        self._names.add(name)
        spec = dict(layer_spec)
        spec["_name"] = name
        self.layers.append(spec)
        return name
    
class Model(nn.Module):
    def __init__(self, specs: ModelSpec | list[dict]):
        super().__init__()
        self.spec = specs
        if isinstance(specs, ModelSpec):
            layer_specs = specs.layers
        else:
            layer_specs = specs

        self._order: list[str] = []
        self.layers_by_name = nn.ModuleDict()
        for i, spec in enumerate(layer_specs):
            name = spec.get("_name") or f"{spec['type']}_{i}"
            layer = build_layer({k: v for k, v in spec.items() if k != "_name"})
            self.layers_by_name[name] = layer
            self._order.append(name)

    def forward(self, inputs: dict, state: dict | None = None, detach_state: bool = False) -> dict:
        if state is None:
            for name in self._order:
                layer = self.layers_by_name[name]
                outputs = layer(inputs)
                inputs = outputs
            return inputs
        
        else:
            next_state = state
            for name in self._order:
                layer = self.layers_by_name[name]
                if name in next_state and "last_hidden" in next_state[name]:
                    inputs["last_hidden"] = next_state[name]["last_hidden"]
                if name in next_state and "last_logits" in next_state[name]:
                    inputs["last_logits"] = next_state[name]["last_logits"]
                outputs = layer(inputs)

                if "last_hidden" in outputs:
                    lh = outputs["last_hidden"]
                    if detach_state:
                        if isinstance(lh, tuple):  # e.g., LSTM (h_n, c_n)
                            lh = tuple(t.detach() if hasattr(t, "detach") else t for t in lh)
                        elif hasattr(lh, "detach"):
                            lh = lh.detach()
                    next_state[name] = {"last_hidden": lh}
                if "last_logits" in outputs:
                    ll = outputs["last_logits"]
                    if detach_state:
                        if hasattr(ll, "detach"):
                            ll = ll.detach()
                    next_state[name] = {"last_logits": ll}

                inputs = outputs
            return inputs, next_state

    def get_layer(self, name: str) -> nn.Module:
        return self.layers_by_name[name]

    def layer_names(self) -> list[str]:
        return list(self._order)

    def summary(self) -> str:
        """Run a dry forward pass and return an ASCII summary table."""
        from torch import nn

        def _num_params(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters())

        rows = []
        header = ["Layer", "Type", "Params"]
        for name in self.layer_names():
            layer = self.get_layer(name)
            pcount = _num_params(layer)
            rows.append([name, layer.__class__.__name__, f"{pcount:,}"])
            if hasattr(layer, "sublayers"):
                for cname, child in layer.sublayers.items():
                    qname = f"{name}.{cname}"
                    rows.append([qname, child.__class__.__name__, f"{_num_params(child):,}"])

        colw = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
        def _fmt_line(cols): return "  ".join(str(c).ljust(w) for c, w in zip(cols, colw))
        lines = [_fmt_line(header), _fmt_line(["-"*len(h) for h in header])]
        lines += [_fmt_line(r) for r in rows]
        return "\n".join(lines)

class CausalLMWrapper(nn.Module):
    def __init__(self, model: nn.Module, loss_fn: nn.Module=None, ignore_index: int=-100):
        super().__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index) if loss_fn is None else loss_fn

    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor]=None, 
                labels: Optional[torch.Tensor]=None, 
                state: dict | None = None, 
                detach_state: bool = False, 
                **kwargs):

        if state is None:
            out = self.model({"input_ids": input_ids, "attention_mask": attention_mask})
        else:
            out, next_state = self.model({"input_ids": input_ids, "attention_mask": attention_mask},
                             state=state,
                             detach_state=detach_state,
                             **kwargs)
        logits = out["logits"]  # (B, T, V)
        loss = None
        if labels is not None:
            if labels.dtype != torch.long:
                labels = labels.long()
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1))
        if state is None:
            return {"loss": loss, "logits": shift_logits, 'labels': shift_labels}
        else:
            return {"loss": loss, "logits": shift_logits, "labels": shift_labels, "state": next_state}