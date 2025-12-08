import torch
import gc
from typing import Tuple, Optional, Any, List
try:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
from torch.utils.data import DataLoader
from tqdm import tqdm
from pprint import pformat
from psychai.utils.serialization import to_serializable
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

class ModelManager:
    def __init__(self):
        self.model = None
        self.model_name = None
        self.model_type = None
        self.model_path = None
        self.reasoning = None

        self.tokenizer = None
        
    def load_model(self, 
                   model_name: str, 
                   model_path: str, 
                   model_type: str,
                   *,
                   max_seq_length: int, 
                   load_in_4bit: bool,
                   dtype: str):
        self.free_memory()
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type

        if UNSLOTH_AVAILABLE:
            self.model, self.tokenizer = self.load_model_unsloth(
                max_seq_length=max_seq_length,
                load_in_4bit=load_in_4bit, 
                dtype=dtype)
        else:
            self.model, self.tokenizer = self.load_hf_model(
                load_in_4bit=load_in_4bit,
                dtype=dtype)

    def apply_lora(self, 
                   *,
                   rank: int, 
                   alpha: int, 
                   dropout: float, 
                   target_modules: List[str]):

        if self.model is None:
            raise ValueError("Model not loaded. Please load the model first.")
        
        if UNSLOTH_AVAILABLE:
            self.model = self.apply_lora_unsloth(
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                target_modules=target_modules,
            )
        else:
            self.model = self.apply_lora(
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                target_modules=target_modules
            )

    def free_memory(self) -> None:
        if hasattr(self, 'model') and self.model is not None:
            try:
                del self.model
                print("Current model deleted")
            except Exception:
                pass
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            try:
                del self.tokenizer
                print("Current tokenizer deleted")
            except Exception:
                pass
        gc.collect()
        torch.cuda.empty_cache()
        self.model = None
        self.model_name = None
        self.model_type = None
        self.model_path = None
        self.tokenizer = None
        print("Cache cleared")

    def load_hf_model(self,
                      load_in_4bit: bool = True,
                      dtype: str = None,
    ) -> Tuple[Any, Any]:
        
        print(f"Loading model: {self.model_name} from {self.model_path}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_path if self.model_path else self.model_name)
            
        if torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path if self.model_path else self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=dtype
            )
            model = prepare_model_for_kbit_training(model)
        else:
            # Use CPU fallback
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path if self.model_path else self.model_name,
                torch_dtype=torch.float32,
                device_map="cpu"
            )
        return model, tokenizer

    def load_model_unsloth(self,
                           dtype: str = "bfloat16",
                           max_seq_length: int = 512, 
                           load_in_4bit: bool = True,
                           full_finetuning: bool = False,
    ) -> Tuple[Any, Any]:
        
        model_path = self.model_path if self.model_path else self.model_name

        print(f"Loading {self.model_name} with Unsloth from: {model_path}")
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            full_finetuning=full_finetuning,
        )
        return model, tokenizer

    def apply_lora(self,
                   rank: int = 16,
                   alpha: int = 32,
                   dropout: float = 0.1,
                   target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj"],
                   bias: str = "none",
                   task_type: str = "CAUSAL_LM") -> Any:
        
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias=bias,
            task_type=task_type
        )
        
        return get_peft_model(self.model, lora_config)

    def apply_lora_unsloth(self,
                           rank: int = 16,
                           alpha: int = 32,
                           dropout: float = 0.1,
                           target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                           bias: str = "none",
                           use_gradient_checkpointing: str = "unsloth",
                           random_state: int = 42,
                           use_rslora: bool = False,
                           loftq_config: Optional[Any] = None) -> Any:
    
        return FastLanguageModel.get_peft_model(
            self.model,
            r=rank,
            target_modules=target_modules,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias=bias,
            use_gradient_checkpointing=use_gradient_checkpointing,
            random_state=random_state,
            use_rslora=use_rslora,
            loftq_config=loftq_config,
        )
    
class TrainingManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.mm = ModelManager()

    def evaluate(self, 
                 dataloader: DataLoader, 
                 eval_fn: Optional[Any], 
                 epoch: Optional[int], 
                 step: Optional[int] = 0, 
                 eval_path: Optional[str] = None):
        
        device = next(self.mm.model.parameters()).device
        if UNSLOTH_AVAILABLE:
            FastLanguageModel.for_inference(self.mm.model)
        else:
            self.mm.model.eval()

        preds_per_batch = []
        labels_per_batch = []
        inputs_per_batch = []
        logits_per_batch = []
        weights = {}
        embedding_maps = []

        def _collect_embeddings(input_ids, attention_mask, embeddings):
            batch_size = input_ids.shape[0]
            embeddings_list = [[] for _ in range(batch_size)]

            b_idx, t_idx = attention_mask.nonzero(as_tuple=True)

            for bi, ti in zip(b_idx.tolist(), t_idx.tolist()):
                token_id = int(input_ids[bi, ti])

                token_entry = {
                    "token_id": token_id,
                    "embedding": embeddings[bi, ti, :].detach(),
                }
                embeddings_list[bi].append(token_entry)
            return embeddings_list

        eval_loss = 0
        with tqdm(total=len(dataloader),
                  desc=f"Evaluating",
                  position=1,
                  leave=True,
                  ncols=100,
                ) as eval_bar:
            for i, batch in enumerate(dataloader):
                with torch.inference_mode():
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch.get("labels", None)
                    if labels is not None:
                        labels = labels.to(device)
                    
                    outputs = self.mm.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels, 
                        return_dict=True,
                        output_logits=True,
                        output_hidden_states=self.cfg.logging.return_embeddings)
                    
                    logits = outputs["logits"]
                    preds = torch.argmax(logits, dim=-1)
                    inputs_per_batch.append(batch["input_ids"].cpu().numpy())
                    if labels is not None:
                        labels_per_batch.append(outputs["labels"].cpu().numpy())
                    logits_per_batch.append(logits.cpu().numpy())
                    assert logits.shape[1] == input_ids.shape[1], "Logits and input_ids sequence length mismatch"
                    preds_per_batch.append(preds.cpu().numpy())
                    if self.cfg.logging.return_embeddings:
                        batch_embeddings = _collect_embeddings(
                            input_ids,
                            attention_mask,
                            outputs["hidden_states"][self.cfg.layer_of_interest]
                        )
                        embedding_maps.append(batch_embeddings)

                    eval_bar.update(1)
                    eval_bar.set_postfix({"loss": f"{eval_loss / (i + 1):.4f}"})
            
            eval_info = {"epoch": epoch + 1, "step": step, "eval_loss": eval_loss / len(dataloader)}

            if eval_fn is not None:
                eval_bar.clear()
                eval_result = eval_fn(self.mm, 
                                      self.cfg, 
                                      inputs_per_batch, 
                                      labels_per_batch, 
                                      logits_per_batch, 
                                      preds_per_batch, 
                                      embeddings_per_batch, 
                                      weights)
                eval_bar.refresh()

                assert isinstance(eval_result, dict), "eval_fn must return a dict"

                tqdm.write("accuracy:\n" + pformat(eval_result["accuracy"]))
                                                
                for key, value in eval_result.items():
                    if isinstance(value, dict):
                        for k, v in value.items():
                            value[k] = to_serializable(v)
                    eval_info[key] = to_serializable(value)

                if eval_path is not None:
                    with open(eval_path, "a") as f:
                        f.write(json.dumps(eval_info) + "\n")

