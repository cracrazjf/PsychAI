"""
Generic training pipeline for language models

This module provides a flexible trainer class that can work with different
models, datasets, and training configurations.
"""

# removed unused import from re
import torch
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Union
from datasets import Dataset

# Optional dependencies
try:
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

from transformers import TrainingArguments, Trainer as HFTrainer
from sklearn.metrics import accuracy_score

from ..models import ModelLoader, load_model_unsloth
from ..models.lora import apply_lora, apply_lora_unsloth
from ..config import TextTrainingConfig

from ..utils import print_memory_usage
from ..data import train_test_split


class Trainer:
    """
    Generic trainer class for language model fine-tuning
    
    Supports various training strategies including SFT, LoRA, and Unsloth optimization.
    Can work with any dataset format as long as it's converted to the expected format.
    """
    
    def __init__(self, config: TextTrainingConfig = None):
        """
        Initialize trainer with configuration
        
        Args:
            config: Training configuration instance
        """
        self.config = config or TextTrainingConfig()
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def load_model_and_tokenizer(
        self, 
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        use_unsloth: bool = True,
        apply_lora: bool = True,
        for_training: bool = True
    ) -> Tuple[Any, Any]:
        """
        Load model and tokenizer
        
        Args:
            model_name: Model name override (uses config if None)
            model_path: Model path override (uses config if None)
            use_unsloth: Whether to use Unsloth optimization
            for_training: Whether model is for training
            
        Returns:
            Tuple of (model, tokenizer)
        """
        model_name = model_name or self.config.MODEL_NAME
        model_path = model_path or self.config.MODEL_PATH
        
        print(f"🚀 Loading model: {model_name} from {model_path}")
        
        if use_unsloth and UNSLOTH_AVAILABLE:
            model, tokenizer = load_model_unsloth(
                model_name=model_name,
                model_path=model_path,
                max_seq_length=self.config.MAX_LENGTH,
                load_in_4bit=True,
                for_training=for_training
            )
            
            # Apply LoRA if training
            if apply_lora:
                model = apply_lora_unsloth(
                    model,
                    rank=self.config.LORA_RANK,
                    alpha=self.config.LORA_ALPHA,
                    dropout=self.config.LORA_DROPOUT,
                    target_modules=self.config.LORA_TARGET_MODULES,
                    random_state=self.config.RANDOM_STATE
                )
        else:
            # Use standard model loading
            loader = ModelLoader()
            model, tokenizer = loader.load_model(
                model_name=model_name,
                for_training=for_training,
                use_unsloth=False
            )

            if apply_lora:
                model = apply_lora(
                    model,
                    rank=self.config.LORA_RANK,
                    alpha=self.config.LORA_ALPHA,
                    dropout=self.config.LORA_DROPOUT,
                    target_modules=self.config.LORA_TARGET_MODULES,
                )

        self.model = model
        self.tokenizer = tokenizer
        
        return model, tokenizer
    
    def prepare_datasets(
        self, 
        train_data: List[Any], 
        eval_data: Optional[List[Any]] = None,
        validation_split: float = 0.1
    ) -> Tuple[Dataset, Dataset]:
        """
        Prepare datasets for training
        
        Args:
            train_data: Training data in chat format
            eval_data: Evaluation data (optional)
            validation_split: Fraction of training data to use for validation
            
        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        if eval_data is None:
            # Split training data
            train_data, eval_data = train_test_split(
                train_data, 
                test_size=validation_split, 
                random_state=self.config.RANDOM_STATE
            )
        
        # Convert to Unsloth format
        train_texts = []
        for i, sample in enumerate(train_data):
            try:
                formatted = self._convert_to_unsloth_format(sample)
                train_texts.append(formatted)
                
                if i < 3:  # Show first few samples
                    print(f"📝 Sample {i}:")
                    print(f"  Text length: {len(formatted['text'])}")
                    
            except Exception as e:
                print(f"⚠️ Skipping training sample {i}: {e}")
                continue
        
        eval_texts = []
        for i, sample in enumerate(eval_data):
            try:
                formatted = self._convert_to_unsloth_format(sample)
                eval_texts.append(formatted)
            except Exception as e:
                print(f"⚠️ Skipping eval sample {i}: {e}")
                continue
        
        train_dataset = Dataset.from_list(train_texts)
        eval_dataset = Dataset.from_list(eval_texts)
        
        print(f"📊 Prepared datasets: {len(train_dataset)} train, {len(eval_dataset)} eval")
        
        return train_dataset, eval_dataset
    
    def _convert_to_unsloth_format(self, data_sample: Any) -> Dict[str, str]:
        """
        Convert data sample to Unsloth format
        
        Args:
            data_sample: Input data sample (chat format, dict, or string)
            
        Returns:
            Dictionary with 'text' key containing formatted text
        """
        if isinstance(data_sample, list):
            # Chat format - apply chat template
            chat_data = data_sample
            
            # Smart truncation for long inputs
            for message in chat_data:
                if self.config.MODEL_TYPE == "llama":
                    try:
                        if message["role"] == "user":
                            user_content = message["content"]
                            
                            # Estimate token usage
                            system_msgs = [m for m in chat_data if m["role"] == "system"]
                            assistant_msgs = [m for m in chat_data if m["role"] == "assistant"]
                    except:
                        raise ValueError(f"Failed to truncate user message: {e}")
                    
                    system_tokens = sum(len(self.tokenizer.encode(m["content"], add_special_tokens=False)) for m in system_msgs)
                    assistant_tokens = sum(len(self.tokenizer.encode(m["content"], add_special_tokens=False)) for m in assistant_msgs)
                    special_tokens_estimate = 50
                    
                    reserved_tokens = system_tokens + assistant_tokens + special_tokens_estimate
                    available_for_user = self.config.MAX_LENGTH - reserved_tokens
                    
                    # Truncate if necessary
                    user_tokens = self.tokenizer.encode(user_content, add_special_tokens=False)
                    if len(user_tokens) > available_for_user:
                        truncated_user_tokens = user_tokens[:available_for_user]
                        message["content"] = self.tokenizer.decode(truncated_user_tokens, skip_special_tokens=True)
            
            # Apply chat template
            try:
                if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
                    formatted_text = self.tokenizer.apply_chat_template(
                        chat_data, 
                        tokenize=False,
                        add_generation_prompt=False
                    )
                else:
                    if self.config.MODEL_TYPE == "llama":
                        formatted_text = self._format_llama_chat_manual(chat_data)
                    else:
                        raise ValueError(f"No chat template found for {self.config.MODEL_TYPE}")
            except Exception as e:
                raise ValueError(f"Failed to format chat data: {e}")
                
        elif isinstance(data_sample, dict):
            if 'text' in data_sample:
                formatted_text = data_sample['text']
            elif 'input' in data_sample and 'output' in data_sample:
                # Convert to chat format
                chat_data = [
                    {"role": "user", "content": data_sample['input']},
                    {"role": "assistant", "content": data_sample['output']}
                ]
                formatted_text = self.tokenizer.apply_chat_template(
                    chat_data, tokenize=False, add_generation_prompt=False
                )
            else:
                formatted_text = str(data_sample)
                
        elif isinstance(data_sample, str):
            formatted_text = data_sample
        else:
            formatted_text = str(data_sample)
        
        # Final length check
        tokens = self.tokenizer.encode(formatted_text, add_special_tokens=False)
        if len(tokens) > self.config.MAX_LENGTH - 5:
            truncated_tokens = tokens[:self.config.MAX_LENGTH - 5]
            formatted_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        
        return {"text": formatted_text}
    
    def _format_llama_chat_manual(self, chat_data: List[Dict]) -> str:
        """Manual Llama chat formatting as fallback"""
        formatted_parts = ["<|begin_of_text|>"]
        
        for message in chat_data:
            role = message["role"]
            content = message["content"]
            formatted_parts.extend([
                f"<|start_header_id|>{role}<|end_header_id|>",
                f"\n\n{content}<|eot_id|>"
            ])
        
        return "".join(formatted_parts)
    
    def create_training_arguments(self, output_dir: Optional[str] = None) -> TrainingArguments:
        """
        Create training arguments from configuration
        
        Args:
            output_dir: Override output directory
            
        Returns:
            TrainingArguments instance
        """
        output_dir = output_dir or self.config.OUTPUT_DIR
        
        return TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.config.BATCH_SIZE,
            gradient_accumulation_steps=self.config.GRAD_ACCUM_STEPS,
            warmup_steps=self.config.WARMUP_STEPS,
            max_steps=self.config.NUM_STEPS*100,
            # num_train_epochs=self.config.NUM_EPOCHS,
            learning_rate=self.config.LEARNING_RATE,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=self.config.LOGGING_STEPS,
            optim=self.config.OPTIMIZER,
            weight_decay=self.config.WEIGHT_DECAY,
            lr_scheduler_type=self.config.LR_SCHEDULER,
            seed=self.config.RANDOM_STATE,
            save_steps=self.config.SAVE_STEPS,
            save_total_limit=self.config.SAVE_TOTAL_LIMIT,
            eval_steps=self.config.EVAL_STEPS,
            eval_strategy=self.config.EVAL_STRATEGY,
            logging_dir=self.config.LOGS_DIR,
            report_to="tensorboard",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
    
    def train(
        self, 
        train_data: List[Any], 
        eval_data: Optional[List[Any]] = None,
        output_dir: Optional[str] = None
    ) -> Any:
        """
        Train the model
        
        Args:
            train_data: Training data
            eval_data: Evaluation data (optional)
            output_dir: Output directory override
            
        Returns:
            Trained model
        """
        print("🏋️ Starting training...")
        
        # Load model if not already loaded
        if self.model is None or self.tokenizer is None:
            self.load_model_and_tokenizer()
        
        # Prepare datasets
        train_dataset, eval_dataset = self.prepare_datasets(train_data, eval_data)
        
        # Create training arguments
        training_args = self.create_training_arguments(output_dir)
        
        # Create trainer
        if UNSLOTH_AVAILABLE:
            trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                dataset_text_field="text",
                max_seq_length=self.config.MAX_LENGTH,
                args=training_args,
            )
        else:
            # Use standard HuggingFace trainer
            trainer = HFTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                args=training_args,
            )
        
        self.trainer = trainer
        
        # Train
        trainer.train()
        
        # Save model
        print("💾 Saving model...")
        trainer.save_model()
        finetuned_model_name = f"{self.config.MODEL_NAME}_{self.config.DATA_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.tokenizer.save_pretrained(os.path.join(self.config.OUTPUT_DIR, finetuned_model_name))
        
        if hasattr(self.model, 'save_pretrained_merged') and self.config.SAVE_MODEL:
            # Save merged model for Unsloth
            merged_dir = os.path.join(self.config.MODELS_PATH, f"merged_{finetuned_model_name}")
            self.model.save_pretrained_merged(merged_dir, self.tokenizer, save_method="merged_16bit")
            print(f"✅ Merged model saved to: {merged_dir}")
        
        print("✅ Training completed!")
        
        return self.model
    
    def save_model(self, output_dir: str):
        """
        Save the trained model
        
        Args:
            output_dir: Directory to save the model
        """
        if self.trainer is None:
            raise ValueError("No trainer available. Train the model first.")
        
        self.trainer.save_model(output_dir)
        print(f"💾 Model saved to: {output_dir}")


def create_compute_metrics_function(tokenizer):
    """
    Create a simple compute_metrics function for training
    
    Args:
        tokenizer: Model tokenizer
        
    Returns:
        Compute metrics function
    """
    def compute_metrics(eval_preds):
        """Simple metrics computation for language model training"""
        predictions, labels = eval_preds
        
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        try:
            # Token-level accuracy
            predictions = predictions.reshape(-1, predictions.shape[-1])
            labels = labels.reshape(-1)
            
            predicted_tokens = predictions.argmax(axis=-1)
            valid_mask = labels != -100
            
            if valid_mask.sum() > 0:
                token_accuracy = (predicted_tokens[valid_mask] == labels[valid_mask]).mean()
                
                return {
                    "eval_token_accuracy": float(token_accuracy),
                    "eval_valid_tokens": int(valid_mask.sum())
                }
            else:
                return {
                    "eval_token_accuracy": 0.0,
                    "eval_valid_tokens": 0
                }
                
        except Exception as e:
            print(f"Error in compute_metrics: {e}")
            return {"eval_token_accuracy": 0.0, "eval_error": 1.0}
    
    return compute_metrics