from typing import Dict, Any, Tuple, Set
import torch
import torch.nn as nn
from .layer import build_layer


class ModelSpec:
    def __init__(self, 
                 vocab_size: int = None, 
                 image_shape: Tuple[int, int, int] = None):
        self.vocab_size = vocab_size
        self.image_shape = image_shape
        self.layers_spec: list[dict] = []
        self.names: set[str] = set()

    def add_layer(self, layer_spec: Dict[str, Any], name: str = None):
        if "type" not in layer_spec:
            raise ValueError("layer_spec must include a 'type' key.")
        
        spec = dict(layer_spec)
        if name is None:
            name = f"{layer_spec['type']}_{len(self.layers_spec)}"
        if name in self.names:
            raise ValueError(f"Duplicate layer name: {name}")
            
        self.names.add(name)
        spec["name"] = name
        self.layers_spec.append(spec)
        return name
   
def _detach(x):
    if isinstance(x, torch.Tensor):
        return x.detach()
    if isinstance(x, (list, tuple)):
        t = type(x)
        return t(_detach(v) for v in x)
    if isinstance(x, dict):
        return {k: _detach(v) for k, v in x.items()}
    return x   

class Model(nn.Module):
    def __init__(self, specs: ModelSpec):
        super().__init__()
        self.spec = specs
        layers_spec = specs.layers_spec

        self.order: list[str] = []
        self.layers = nn.ModuleDict()
        for layer_spec in layers_spec:
            name = layer_spec.get("name")
            layer = build_layer({k: v for k, v in layer_spec.items() if k != "name"})
            self.layers[name] = layer
            self.order.append(name)

    def forward(self, 
                inputs: dict, 
                *,
                recurrent_state: dict | None = None, 
                detach_state: bool = True, 
                return_embeds: bool = False):

        embeds = {} if return_embeds else None
        next_state = {} if recurrent_state is not None else None
        prev_state = recurrent_state or {}
        
        for name in self.order:
            layer = self.layers[name]
            layer_state = prev_state.get(name, {})
            merged_inputs = inputs if not layer_state else {**inputs, **layer_state}
            outputs = layer(merged_inputs)
            
            if recurrent_state is not None:
                state_out = {}
                if "last_hidden" in outputs:
                    state_out["last_hidden"] = outputs["last_hidden"]
                if "last_logits" in outputs:
                    state_out["last_logits"] = outputs["last_logits"]

                if state_out:
                    if detach_state:
                        state_out = _detach(state_out)
                    next_state[name] = state_out

            inputs = outputs

            if return_embeds:
                cpu_embeds = {}
                for key, value in outputs.items():  
                    cpu_embeds[key] = value
                embeds[name] = cpu_embeds

        return inputs, next_state, embeds

    def get_layer(self, name: str) -> nn.Module:
        return self.layers[name]

    def layer_names(self) -> list[str]:
        return list(self.order)

    def get_weights(self, verbose=False) -> dict:
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
        full_name_weights = {}
        for name, weight in weights.items():
            for pname, tensor in weight.items():
                full_name = f"{name}.{pname}"
                full_name_weights[full_name] = tensor
                shape, numel, dtype = _get_shape_dtype(tensor)
                rows.append((full_name, shape, numel, dtype))

        if verbose:
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
        return full_name_weights

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
        self.base_model = model
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index) if loss_fn is None else loss_fn

    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: torch.Tensor | None = None, 
                labels: torch.Tensor | None = None, 
                recurrent_state: dict | None = None, 
                detach_state: bool = True, 
                return_embeds: bool = False):
        
        if labels is not None:
            model_input_ids = input_ids[:, :-1]
            if attention_mask is not None:
                model_attention_mask = attention_mask[:, :-1]
            else:
                model_attention_mask = None
        else:
            model_input_ids = input_ids
            model_attention_mask = attention_mask

        out, next_state, embeds = self.base_model({"input_ids": model_input_ids, 
                                                   "attention_mask": model_attention_mask},
                                                    recurrent_state=recurrent_state,
                                                    detach_state=detach_state,
                                                    return_embeds=return_embeds)
        
        logits = out["logits"]
        loss = None
        if labels is not None:
            labels = labels[:, 1:].contiguous()
            if labels.dtype != torch.long:
                labels = labels.long()

            loss = self.loss_fn(logits.view(-1, logits.size(-1)),
                                labels.view(-1))
        else:
            labels = torch.tensor([], device=input_ids.device)
            loss = torch.tensor(0.0, device=input_ids.device)


        return {"input_ids": model_input_ids,
                "attention_mask": model_attention_mask,
                "loss": loss, 
                "logits": logits,
                "labels": labels, 
                "recurrent_state": next_state, 
                "embeds": embeds}
        
