from psychai.model_manager.language import LM_ModelManager
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from itertools import chain
from datasets import Dataset
from torch.utils.data import DataLoader
import torch
from pathlib import Path

class Custom_Evaluator:
    def __init__(self, config):
        self.config = config
        self.model_manager = LM_ModelManager()

    def load_model_and_tokenizer(self, model_path):
        self.model_manager.load_model(self.config.MODEL_NAME,
                                      model_path, 
                                      self.config.TASK, 
                                      custom=self.config.CUSTOMIZED_MODEL,
                                      tokenizer_path=model_path,
                                      trust_remote_code=self.config.TRUST_REMOTE_CODE)
    
    def tokenize_language_data(self, dataset_path):
        if self.model_manager.tokenizer.pad_token is None: self.model_manager.tokenizer.pad_token = self.model_manager.tokenizer.eos_token
        dataset = load_dataset("json", data_files=dataset_path, split="train")
        def tokenize_function(batch):
            return self.model_manager.tokenizer(batch["text"], add_special_tokens=False, truncation=False)

        task = self.config.TASK
        if task == "causal_lm":
            collator = DataCollatorForLanguageModeling(tokenizer=self.model_manager.tokenizer, mlm=False)
        elif task == "masked_lm":
            collator = DataCollatorForLanguageModeling(tokenizer=self.model_manager.tokenizer, mlm=True)
        else:
            raise ValueError(f"Unsupported task type: {task}")

        tokenized_dataset = dataset.map(tokenize_function, batched=True, batch_size=1, num_proc=1)
        return tokenized_dataset, collator

    def create_dataloader(self, dataset, collator):
        def _concatenate_input_ids_and_attention_mask(dataset):
            input_ids_lists = dataset["input_ids"]
            attn_mask_lists = dataset["attention_mask"]
            input_ids = list(chain.from_iterable(input_ids_lists))
            attention = list(chain.from_iterable(attn_mask_lists))
            return {"input_ids": input_ids, "attention_mask": attention}

        def _create_sliding_windows(input_ids, attention_mask, sequence_length, pad_token_id):
            assert sequence_length > 1, "sequence_length must be >= 2 for causal LM"
            n = len(input_ids)
            windows = {"input_ids": [], "attention_mask": []}
            for start in range(0, n, 1):
                end = start + sequence_length
                ids = input_ids[start:end]
                msk = attention_mask[start:end]

                # pad if last window is too short
                if len(ids) < sequence_length:
                    pad_len = sequence_length - len(ids)
                    ids = ids + [pad_token_id] * pad_len
                    msk = msk + [0] * pad_len

                windows["input_ids"].append(ids)
                windows["attention_mask"].append(msk)

                if end >= n:
                    break
            return windows
        
        def _create_nonoverlapping_windows(input_ids, attention_mask, sequence_length, pad_token_id):
            assert sequence_length > 1, "sequence_length must be >= 2 for causal LM"
            windows = {"input_ids": [], "attention_mask": []}

            for i in range(0, len(input_ids), sequence_length):
                ids = input_ids[i : i + sequence_length]
                mask = attention_mask[i : i + sequence_length]

                if len(ids) < sequence_length:
                    pad_len = sequence_length - len(ids)
                    ids = ids + [pad_token_id] * pad_len
                    mask = mask + [0] * pad_len

                windows["input_ids"].append(ids)
                windows["attention_mask"].append(mask)
            return windows
        
        dataset = _concatenate_input_ids_and_attention_mask(dataset)
        if self.config.OVERLAPPING_SEQUENCES:
            windows = _create_sliding_windows(dataset["input_ids"], dataset["attention_mask"], self.config.SEQUENCE_LENGTH, self.model_manager.tokenizer.pad_token_id)
        else:
            windows = _create_nonoverlapping_windows(dataset["input_ids"], dataset["attention_mask"], self.config.SEQUENCE_LENGTH, self.model_manager.tokenizer.pad_token_id)
        dataset = Dataset.from_dict(windows)
        dataloader = DataLoader(dataset, batch_size=self.config.BATCH_SIZE, 
                        shuffle=False, collate_fn=collator,
                        pin_memory=self.config.PIN_MEMORY, drop_last=self.config.DROP_LAST)
        return dataloader
    
    def build_batch_representations_map(self,input_ids, attention_mask, representations, sample_id):
        mask = attention_mask.bool()
        ids_cpu = input_ids.detach().cpu()

        batch_representations_map = {}
        b_idx, t_idx = mask.nonzero(as_tuple=True)
        for bi, ti in zip(b_idx.tolist(), t_idx.tolist()):
            sid = sample_id+bi
            tok_id = int(ids_cpu[bi, ti])
            map = {"token_id": tok_id, "layers": {}}
            map["token_string"] = self.model_manager.tokenizer.decode([tok_id], skip_special_tokens=True)
            for layer_name, layer_repr in representations.items():
                layer_bucket = {}
                for key, tensor in layer_repr.items():
                    if isinstance(tensor, tuple):
                        vec = tuple(t[bi, ti].detach().cpu().numpy() for t in tensor)
                    elif hasattr(tensor, "detach"):
                        vec = tensor[bi, ti].detach().cpu().numpy()
                    layer_bucket[key] = vec
                map["layers"][layer_name] = layer_bucket
            batch_representations_map[(sid, ti)] = map
        return batch_representations_map
    
    def print_token_map_shapes(self, token_map, max_tokens=5):
        for i, ((sid, pos), rec) in enumerate(token_map.items()):
            if i >= max_tokens:
                break
            tok_id = rec.get("token_id")
            tok_str = rec.get("token_string", None)
            print(f"(sample={sid}, pos={pos}) token_id={tok_id} token_str={tok_str!r}")
            for layer, sub in rec["layers"].items():
                print(f"  {layer}:")
                for key, val in sub.items():
                    if hasattr(val, "shape"):  # tensor
                        print(f"    {key:<12} {tuple(val.shape)}")
                    elif isinstance(val, dict) and "shape" in val:
                        print(f"    {key:<12} {val['shape']}")
                    else:
                        print(f"    {key:<12} {type(val)}")

    def evaluate_language_model(self):
        eval_dataset, collator = self.tokenize_language_data(self.config.EVAL_DATA_PATH)
        eval_loader = self.create_dataloader(eval_dataset, collator)
        legacy_eval = self.config.SEQUENCE_LENGTH == 2
        model_root = Path(self.config.MODEL_ROOT)
        for subdir in model_root.rglob("export"):
            self.load_model_and_tokenizer(subdir)
            model = self.model_manager.model
            weights = model.model.get_weights()
        
            if legacy_eval:
                state = {}
            model.eval()
            token_representations_map = {}
            global_sample_id = 0
            with torch.no_grad():
                for batch in eval_loader:
                    if legacy_eval:
                        outputs = model.forward(input_ids=batch["input_ids"], 
                                                attention_mask=batch["attention_mask"], 
                                                labels=batch["labels"],
                                                return_repr=True, 
                                                state=state)
                        state = outputs["state"]
                    else:
                        outputs = model.forward(input_ids=batch["input_ids"], 
                                                attention_mask=batch["attention_mask"], 
                                                labels=batch["labels"], 
                                                return_repr=True)
                    batch_representations_map = self.build_batch_representations_map(batch["input_ids"], 
                                                                                     batch["attention_mask"], 
                                                                                     outputs["representations"],
                                                                                     global_sample_id)
                    token_representations_map.update(batch_representations_map)
                    global_sample_id += len(batch["input_ids"])
            self.print_token_map_shapes(token_representations_map)
            return token_representations_map, weights