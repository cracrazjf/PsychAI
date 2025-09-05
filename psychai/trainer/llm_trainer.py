from typing import Any, Tuple, Optional
from datasets import Dataset, load_dataset
from ..model_manager import LLM_ModelManager
import os
from trl import SFTTrainer, SFTConfig

class LLM_Trainer:
    
    def __init__(self, config):
        self.config = config
        self.model_manager = LLM_ModelManager()
        self.trainer = None
        
    def load_model_and_tokenizer(
        self
    ) -> Tuple[Any, Any]:

        model_name = self.config.MODEL_NAME
        model_path = self.config.MODEL_PATH
        use_unsloth = self.config.USE_UNSLOTH
        apply_lora = self.config.APPLY_LORA
        reasoning = self.config.REASONING

        chat_template = self.config.CHAT_TEMPLATE

        print(f"🚀 Loading model: {model_name} from {model_path}")
        
        self.model_manager = LLM_ModelManager()
        self.model_manager.load_model(
            model_name=model_name,
            model_path=model_path,
            reasoning=reasoning,
            use_unsloth=use_unsloth,
            for_training=True,
            max_seq_length=self.config.MAX_SEQ_LENGTH,
            load_in_4bit=self.config.LOAD_IN_4BIT,
            full_finetuning=self.config.FULL_FINETUNING,
            dtype=self.config.DTYPE,
        )

        if chat_template is not None:
            self.model_manager.apply_chat_template(chat_template)

        if apply_lora:
            self.model_manager.apply_lora(
                use_unsloth=use_unsloth,
                rank=self.config.LORA_RANK,
                alpha=self.config.LORA_ALPHA,
                dropout=self.config.LORA_DROPOUT,
                target_modules=self.config.LORA_TARGET_MODULES,
                bias=self.config.BIAS,
                use_gradient_checkpointing=self.config.USE_GRADIENT_CHECKPOINTING,
                random_state=self.config.RANDOM_STATE,
                use_rslora=self.config.USE_RSLORA,
                loftq_config=self.config.LOFTQ_CONFIG
            )

    def prepare_datasets(
        self,
        prompt_template: Optional[str] = None,
    ) -> Tuple[Dataset, Dataset]:   

        train_dataset = load_dataset("json", data_files=self.config.TRAIN_DATA_PATH, split="train")
        if self.config.EVAL_DATA_PATH is not None:
            eval_dataset = load_dataset("json", data_files=self.config.EVAL_DATA_PATH, split="train")
        else:
            eval_dataset = None

        data_type = self.config.DATA_TYPE
        ESO_TOKEN = self.model_manager.tokenizer.eos_token

        if not isinstance(train_dataset[0], dict):
            raise ValueError("train_data must be a list of dictionaries")

        if data_type == "chat":
            train_dataset = train_dataset.map(self._format_chat_prompt, batched=True)
        elif data_type == "instruction":
            train_dataset = train_dataset.map(self._format_instruction_prompt, 
                                              fn_kwargs={"prompt_template": prompt_template, 
                                              "ESO_TOKEN": ESO_TOKEN}, batched=True)
        else:
            raise ValueError("Invalid data type")
        
        print("printing examples of training data:")
        for i in range (3):
            print(f"Example {i}: {train_dataset[i]['text']}")
            print("--------------------------------")

        if eval_dataset is not None:
            if data_type == "chat":
                eval_dataset = eval_dataset.map(self._format_chat_prompt, batched=True)
            elif data_type == "instruction":
                eval_dataset = eval_dataset.map(self._format_instruction_prompt, 
                                                fn_kwargs={"prompt_template": prompt_template, 
                                                "ESO_TOKEN": ESO_TOKEN}, batched=True)
            else:
                raise ValueError("Invalid data type")

        return train_dataset, eval_dataset

    def _format_chat_prompt(self, examples):
        reasoning_effort = self.config.REASONING_EFFORT
        if reasoning_effort is None:
            reasoning_effort = "low"
        convos = examples['messages']
        texts = [self.model_manager.tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False, reasoning_effort = reasoning_effort) for convo in convos]
        
        return { "text" : texts}

    def _format_instruction_prompt(self, examples, prompt_template, ESO_TOKEN):
        if prompt_template is None:
            prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
            
            ### Instruction:
            {}
            
            ### Input:
            {}
            
            ### Response:
            {}"""
            
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = prompt_template.format(instruction, input, output) + ESO_TOKEN
            texts.append(text)
        return { "text" : texts}

    def create_training_arguments(self):
        print(self.config.USE_UNSLOTH)
        if self.config.USE_UNSLOTH:
            return SFTConfig(
                dataset_text_field = "text",
                max_length=self.config.MAX_SEQ_LENGTH,
                packing=False,
                per_device_train_batch_size=self.config.PER_DEVICE_TRAIN_BATCH_SIZE,
                gradient_accumulation_steps=self.config.GRADIENT_ACCUMULATION_STEPS,
                warmup_steps=self.config.WARMUP_STEPS,
                #num_train_epochs=self.config.NUM_EPOCHS,
                max_steps=self.config.MAX_STEPS,
                learning_rate=self.config.LEARNING_RATE,
                logging_steps=self.config.LOGGING_STEPS,
                optim=self.config.OPTIMIZER,
                weight_decay=self.config.WEIGHT_DECAY,
                lr_scheduler_type=self.config.LR_SCHEDULER,
                seed=self.config.RANDOM_STATE,
                output_dir=self.config.OUTPUT_DIR,
                report_to=self.config.REPORT_TO,
                save_strategy=self.config.SAVE_STRATEGY,
                eval_strategy=self.config.EVAL_STRATEGY,
                per_device_eval_batch_size=self.config.PER_DEVICE_EVAL_BATCH_SIZE,
                eval_accumulation_steps=self.config.EVAL_ACCUMULATION_STEPS,
                eval_steps=self.config.EVAL_STEPS,
                logging_dir=self.config.LOGGING_DIR,
                load_best_model_at_end=self.config.LOAD_BEST_MODEL_AT_END,
                metric_for_best_model=self.config.METRIC_FOR_BEST_MODEL,  
                save_steps=self.config.SAVE_STEPS,
                save_total_limit=self.config.SAVE_TOTAL_LIMIT,
                greater_is_better=self.config.GREATER_IS_BETTER,
            )
        # else:
        #     return TrainingArguments(output_dir=self.config.OUTPUT_DIR,
        #                         per_device_train_batch_size=self.config.PER_DEVICE_TRAIN_BATCH_SIZE,
        #                         gradient_accumulation_steps=self.config.GRADIENT_ACCUMULATION_STEPS,
        #                         warmup_steps=self.config.WARMUP_STEPS,
        #                         max_steps=self.config.MAX_STEPS,
        #                         learning_rate=self.config.LEARNING_RATE,
        #                         logging_steps=self.config.LOGGING_STEPS,
        #                         optim=self.config.OPTIMIZER,
        #                         weight_decay=self.config.WEIGHT_DECAY,
        #                         lr_scheduler_type=self.config.LR_SCHEDULER,
        #                         seed=self.config.RANDOM_STATE,
        #                         save_steps=self.config.SAVE_STEPS,
        #                         save_total_limit=self.config.SAVE_TOTAL_LIMIT,
        #                         eval_steps=self.config.EVAL_STEPS,
        #                         eval_strategy=self.config.EVAL_STRATEGY,
        #                         logging_dir=self.config.LOGGING_DIR,
        #                         report_to=self.config.REPORT_TO,
        #                         load_best_model_at_end=self.config.LOAD_BEST_MODEL_AT_END,
        #                         metric_for_best_model=self.config.METRIC_FOR_BEST_MODEL,
        #                         greater_is_better=self.config.GREATER_IS_BETTER,
        #                         )
    
    def train(
        self, 
        prompt_template: Optional[str] = None,
    ) -> Any:
    
        print("🏋️ Starting training...")
        
        if self.model_manager.model is None or self.model_manager.tokenizer is None:
            self.load_model_and_tokenizer()
        
        train_dataset, eval_dataset = self.prepare_datasets(prompt_template)
        
        # Create training arguments
        self.training_args = self.create_training_arguments()
        
        # Create trainer
        if self.config.USE_UNSLOTH:
            trainer = SFTTrainer(
                model=self.model_manager.model,
                tokenizer=self.model_manager.tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                args=self.training_args,
            )
        else:
            raise ValueError("Unsloth is not available")
            # # Use standard HuggingFace trainer
            # trainer = HFTrainer(
            #     model=self.model_manager.model,
            #     tokenizer=self.model_manager.tokenizer,
            #     train_dataset=train_dataset,
            #     eval_dataset=eval_dataset,
            #     args=self.training_args,
            # )
        
        self.trainer = trainer
        
        # Train
        trainer.train()
        
        # Save model
        print("💾 Saving model...")
        self.save_model()
        print("✅ Training completed!")
        
        return self.model_manager.model, self.model_manager.tokenizer
    
    def save_model(self):
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H")
        save_path = self.config.MODELS_PATH + "/" + self.model_manager.model_name + "_" +self.config.DATA_NAME + "_" +timestamp
        os.makedirs(save_path, exist_ok=True)
        if self.trainer is None:
            raise ValueError("No trainer available. Train the model first.")
        self.model_manager.tokenizer.save_pretrained(save_path)
        self.model_manager.model.save_pretrained(save_path)
        print(f"💾 Lora model saved to: {save_path}")