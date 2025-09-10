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
        self.names: set[str] = set()

    def add_layer(self, layer_spec: Dict[str, Any], name: str = None):
        if "type" not in layer_spec:
            raise ValueError("layer_spec must include a 'type' key.")
        spec = dict(layer_spec)

        # add name of the layer if not provided
        # the name is the type of the layer and the index of the layer
        if name is None:
            name = f"{layer_spec['type']}_{len(self.layers)}"
        # check if the name is already in the set
        if name in self.names:
            raise ValueError(f"Duplicate layer name: {name}")
        self.names.add(name)
        spec["name"] = name
        self.layers.append(spec)
        return name
    
class Model(nn.Module):
    def __init__(self, specs: ModelSpec):
        super().__init__()
        self.spec = specs
        layer_specs = specs.layers

        self.order: list[str] = []
        self.layers_by_name = nn.ModuleDict()
        for spec in layer_specs:
            name = spec.get("name")
            layer = build_layer({k: v for k, v in spec.items() if k != "name"})
            self.layers_by_name[name] = layer
            self.order.append(name)

    def forward(self, inputs: dict, state: dict | None = None, detach_state: bool = False) -> dict:
        if state is None:
            for name in self.order:
                layer = self.layers_by_name[name]
                outputs = layer(inputs)
                inputs = outputs
            return inputs
        
        else:
            # this is the training mode for stateful rnn models
            next_state = state
            for name in self.order:
                layer = self.layers_by_name[name]
                if name in next_state and "last_hidden" in next_state[name]:
                    inputs["last_hidden"] = next_state[name]["last_hidden"]
                if name in next_state and "last_logits" in next_state[name]:
                    inputs["last_logits"] = next_state[name]["last_logits"]
                outputs = layer(inputs)

                if "last_hidden" in outputs:
                    last_hidden = outputs["last_hidden"]
                    if detach_state:
                        # this is for LSTM layer
                        if isinstance(last_hidden, tuple):
                            last_hidden = tuple(t.detach() if hasattr(t, "detach") else t for t in last_hidden)
                        elif hasattr(last_hidden, "detach"):
                            last_hidden = last_hidden.detach()
                    next_state[name] = {"last_hidden": last_hidden}
                if "last_logits" in outputs:
                    last_logits = outputs["last_logits"]
                    if detach_state:
                        if hasattr(last_logits, "detach"):
                            last_logits = last_logits.detach()
                    next_state[name] = {"last_logits": last_logits}

                inputs = outputs
            return inputs, next_state

    def get_layer(self, name: str) -> nn.Module:
        return self.layers_by_name[name]

    def layer_names(self) -> list[str]:
        return list(self.order)

    def summary(self) -> str:
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
        logits = out["logits"]
        loss = None
        if labels is not None:
            if labels.dtype != torch.long:
                labels = labels.long()
            # shift the logits and labels by one
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1))
        if state is None:
            return {"loss": loss, "logits": shift_logits, "labels": shift_labels}
        else:
            return {"loss": loss, "logits": shift_logits, "labels": shift_labels, "state": next_state}