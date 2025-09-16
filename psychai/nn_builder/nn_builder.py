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

    def forward(self, inputs: dict, state: dict | None = None, detach_state: bool = False, return_repr: bool = False) -> dict:
        representations = {}
        if state is None:
            for name in self.order:
                layer = self.layers_by_name[name]
                outputs = layer(inputs)
                inputs = outputs
                representations[name] = outputs
            return inputs, representations
        
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
                if return_repr:
                    representations[name] = outputs
            return inputs, next_state, representations

    def get_layer(self, name: str) -> nn.Module:
        return self.layers_by_name[name]

    def layer_names(self) -> list[str]:
        return list(self.order)

    def get_weights(self) -> dict:
        weights = {}
        for name in self.layer_names():
            layer = self.get_layer(name)
            weights[name] = layer.state_dict()
        
        def _get_shape_dtype(x: torch.Tensor):
            shape = tuple(getattr(x, "shape", ()))
            if hasattr(x, "numel"):
                try:
                    numel = int(x.numel())
                except TypeError:
                    numel = int(x.numel)
            elif hasattr(x, "size"):
                numel = int(x.size if isinstance(x.size, (int,)) else x.size)  # numpy
            else:
                numel = 0
            dtype = getattr(x, "dtype", "unknown")
            dtype = str(dtype)
            return shape, numel, dtype
        
        def _commas(n: int) -> str:
            return f"{n:,}"
        
        rows = []
        for name, weight in weights.items():
            for pname, tensor in weight.items():
                full_name = f"{name}.{pname}"
                shape, numel, dtype = _get_shape_dtype(tensor)
                rows.append((full_name, shape, numel, dtype))
        headers = ("Layer.Param", "Shape", "#Params", "Dtype")

        name_w = max(len(headers[0]), *(len(r[0]) for r in rows)) if rows else len(headers[0])
        shape_w = max(len(headers[1]), *(len(str(r[1])) for r in rows)) if rows else len(headers[1])
        num_w   = max(len(headers[2]), *(len(_commas(r[2])) for r in rows)) if rows else len(headers[2])
        dtype_w = max(len(headers[3]), *(len(str(r[3])) for r in rows)) if rows else len(headers[3])

        line = f"{headers[0]:<{name_w}}  {headers[1]:<{shape_w}}  {headers[2]:>{num_w}}  {headers[3]:<{dtype_w}}"
        sep  = "-" * len(line)
        out_lines = [line, sep]

        for name, shape, numel, dtype in rows:
            out_lines.append(
                f"{name:<{name_w}}  {str(shape):<{shape_w}}  {_commas(numel):>{num_w}}  {str(dtype):<{dtype_w}}"
            )
        print("\n".join(out_lines))
        return weights

    def summary(self) -> str:
        from torch import nn

        def _num_params(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters())
        
        def _collect_sublayers(layer: nn.Module, name: str) -> list[nn.Module]:
            rows = []
            rows.append([name, layer.__class__.__name__, f"{_num_params(layer):,}"])
            # if it has children, recurse
            if hasattr(layer, "sublayers"):
                for cname, child in layer.sublayers.items():
                    qname = f"{name}.{cname}" if name else cname
                    rows.extend(_collect_sublayers(child, qname))
            return rows

        rows = []
        header = ["Layer", "Type", "Params"]
        for name in self.layer_names():
            layer = self.get_layer(name)
            pcount = _num_params(layer)
            if hasattr(layer, "sublayers"):
                rows.extend(_collect_sublayers(layer, name))
            else:
                rows.append([name, layer.__class__.__name__, f"{pcount:,}"])

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
                return_repr: bool = False,
                **kwargs):

        if state is None:
            out, representations = self.model({"input_ids": input_ids, "attention_mask": attention_mask},
                                             return_repr=return_repr)
        else:
            out, next_state, representations = self.model({"input_ids": input_ids, "attention_mask": attention_mask},
                             state=state,
                             detach_state=detach_state,
                             return_repr=return_repr,
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
            return {"loss": loss, "logits": shift_logits, "labels": shift_labels, "representations": representations}
        else:
            return {"loss": loss, "logits": shift_logits, "labels": shift_labels, "state": next_state, "representations": representations}