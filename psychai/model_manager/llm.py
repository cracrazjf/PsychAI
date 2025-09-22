import torch
import gc
import logging
from typing import Tuple, Optional, Any, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("transformers").setLevel(logging.INFO)

# Check for optional dependencies
try:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

class LLM_ModelManager:
    def __init__(self):
        self.model = None
        self.model_name = None
        self.tokenizer = None
        self.reasoning = None
    def load_model(self, 
                   model_name: str, 
                   model_path: str, 
                   *,
                   reasoning: bool,
                   use_unsloth: bool, 
                   for_training: bool,
                   max_seq_length: int, 
                   load_in_4bit: bool,
                   full_finetuning: bool, 
                   dtype: str):
        self.free_memory()
        
        self.model_name = model_name
        self.model_company = self.infer_model_company(model_name)
        self.reasoning = reasoning

        if use_unsloth and UNSLOTH_AVAILABLE:
            self.model, self.tokenizer = load_model_unsloth(
                model_name=model_name,
                model_path=model_path,
                max_seq_length=max_seq_length,
                load_in_4bit=load_in_4bit,
                full_finetuning=full_finetuning,
                dtype=dtype,
                for_training=for_training
            )
        else:
            self.model, self.tokenizer = load_model(
                model_name=model_name,
                model_path=model_path,
                load_in_4bit=load_in_4bit,
                dtype=dtype,
                for_training=for_training
            )
        print(f"Loaded {model_name} from {self.model_company}")

    def infer_model_company(self, model_name: str):
        for company in ['llama', 'gpt', 'deepseek', 'qwen', 'mistral', 'gemma'] :
            if company in model_name.lower():
                return company
        return None

    def apply_chat_template(self, chat_template: str):
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template,
        )

    def apply_lora(self, 
                   *,
                   use_unsloth: bool, 
                   rank: int, 
                   alpha: int, 
                   dropout: float, 
                   target_modules: List[str], 
                   bias: str, 
                   use_gradient_checkpointing: str,
                   random_state: int, 
                   use_rslora: bool,
                   loftq_config: Optional[Any]):

        if self.model is None:
            raise ValueError("Model not loaded. Please load the model first.")
        
        if use_unsloth and UNSLOTH_AVAILABLE:
            self.model = apply_lora_unsloth(
                self.model,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                target_modules=target_modules,
                bias=bias,
                use_gradient_checkpointing=use_gradient_checkpointing,
                random_state=random_state,
                use_rslora=use_rslora,
                loftq_config=loftq_config
            )
        else:
            self.model = apply_lora(    
                self.model,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                target_modules=target_modules,
                bias=bias
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
        self.reasoning = None
        self.tokenizer = None
        self.model_name = None
        print("Cache cleared")

def load_model(
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", 
    model_path: str = None,
    load_in_4bit: bool = True,
    dtype: str = None,
    for_training: bool = False
) -> Tuple[Any, Any]:
    
    print(f"Loading model: {model_name} from {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path if model_path else model_name)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if for_training:
        print("Loading model for training...")
        
        if torch.cuda.is_available():
            # Use CUDA with 4-bit quantization for training
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path if model_path else model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=dtype
            )
            model = prepare_model_for_kbit_training(model)
        else:
            # Use CPU fallback
            model = AutoModelForCausalLM.from_pretrained(
                model_path if model_path else model_name,
                torch_dtype=torch.float32,
                device_map="cpu"
            )
    else:
        print("Loading model for inference...")
        
        if torch.cuda.is_available():
            print("Using CUDA for inference.")
            model = AutoModelForCausalLM.from_pretrained(
                model_path if model_path else model_name,
                torch_dtype=torch.float16,
                device_map="cuda"
            )
        else:
            print("Using CPU for inference.")
            model = AutoModelForCausalLM.from_pretrained(
                model_path if model_path else model_name,
                torch_dtype=torch.float32,
                device_map="cpu"
            )

    return model, tokenizer

def load_model_unsloth(
    model_name: str, 
    model_path: str = None,
    dtype: str = None,
    max_seq_length: int = 512, 
    load_in_4bit: bool = True, 
    full_finetuning: bool = False,
    for_training: bool = True
) -> Tuple[Any, Any]:
    
    if not UNSLOTH_AVAILABLE:
        raise ImportError("Unsloth is not installed. Please install it first.")

    # Use local path if provided, otherwise use model_name
    model_path = model_path if model_path else model_name
    
    quantization_str = "4-bit" if load_in_4bit else "16-bit"
    print(f"Loading {model_name} with Unsloth from: {model_path} in {quantization_str}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        full_finetuning=full_finetuning,
    )
    
    if not for_training:
        # FastLanguageModel.for_inference(model)
        model.eval()
    
    return model, tokenizer

def apply_lora(
    model: Any,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
    bias: str = "none",
    task_type: str = "CAUSAL_LM"
) -> Any:
    if target_modules is None:
        # Default target modules for most transformer models
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias=bias,
        task_type=task_type
    )
    
    return get_peft_model(model, lora_config)

def apply_lora_unsloth(
    model: Any,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
    bias: str = "none",
    use_gradient_checkpointing: str = "unsloth",
    random_state: int = 42,
    use_rslora: bool = False,
    loftq_config: Optional[Any] = None
) -> Any:
    
    if not UNSLOTH_AVAILABLE:
        raise ImportError("Unsloth is not installed. Please install it first.")
    
    if target_modules is None:
        # Default target modules for Unsloth (typically more comprehensive)
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    
    print("🔧 Applying LoRA with Unsloth...")
    
    model = FastLanguageModel.get_peft_model(
        model,
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
    
    return model