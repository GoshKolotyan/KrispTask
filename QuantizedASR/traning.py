import torch
import logging
from typing import Tuple, Any, Dict
from transformers import TrainingArguments, Trainer

from configs import QuantizeConfigs
from dataloading import ArmenianDataLoader
from model import QuantizedArmenianModelLoader
from helpers import prepare_dataset, DataCollatorCTCWithPadding
from metrics import Metrics 

logger = logging.getLogger(__name__)


class ArmenianASRTrainer:
    
    def __init__(self, config_path: str):
        self.config = QuantizeConfigs(config_path)
        self.repo_name = getattr(self.config, 'repo_name', './wav2vec2-armenian-frozen')
        
        self.model = None
        self.processor = None
        self.trainer = None
        self.metrics = None
        
    def load_datasets(self, verbose: bool = True) -> Tuple[Any, Any, Dict]:
        logger.info("Loading datasets...")
        dataloader = ArmenianDataLoader(self.config)
        train_dataset, test_dataset, vocab_dict = dataloader(verbose=verbose)
        
        logger.info(f"Training examples: {len(train_dataset):,}")
        logger.info(f"Test examples: {len(test_dataset):,}") 
        logger.info(f"Vocabulary size: {len(vocab_dict)}")
        
        return train_dataset, test_dataset, vocab_dict
    
    def load_model(self) -> Tuple[Any, Any]:
        logger.info("Loading model...")
        model_loader = QuantizedArmenianModelLoader(self.config)
        
        self.model, self.processor = model_loader(
            # repo_name=self.repo_name if hasattr(self.config, 'push_to_hub') and self.config.push_to_hub else None,
            add_adapters=True
        )
        
        return self.model, self.processor
    
    def prepare_datasets(self, train_dataset: Any, test_dataset: Any) -> Tuple[Any, Any]:
        logger.info("Preparing datasets...")
        
        def prepare_batch(batch):
            return prepare_dataset(batch, self.processor)
        
        logger.info("Processing training data...")
        common_voice_train = train_dataset.map(
            prepare_batch, 
            remove_columns=train_dataset.column_names,
            desc="Preparing train dataset"
        )
        
        logger.info("Processing test data...")
        common_voice_test = test_dataset.map(
            prepare_batch, 
            remove_columns=test_dataset.column_names,
            desc="Preparing test dataset"
        )
        
        sample = common_voice_train[0]
        logger.info(f"Preprocessed sample keys: {list(sample.keys())}")
        if 'input_values' in sample:
            logger.info(f"Input shape: {sample['input_values'].shape if hasattr(sample['input_values'], 'shape') else 'N/A'}")
        
        return common_voice_train, common_voice_test
    
    def setup_training_components(self, vocab_dict: Dict) -> Tuple[Any, Any, TrainingArguments]:
        logger.info("Setting up training components...")
        
        data_collator = DataCollatorCTCWithPadding(processor=self.processor)
        self.metrics = Metrics(processor=self.processor)
        
        training_args = TrainingArguments(
            output_dir=self.repo_name,
            group_by_length=self.config.training.group_by_length,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            eval_strategy=self.config.training.eval_strategy,
            num_train_epochs=self.config.training.num_train_epochs,
            gradient_checkpointing=self.config.training.gradient_checkpointing,
            fp16=self.config.training.fp16,
            save_strategy=self.config.training.save_strategy,     
            save_total_limit=self.config.training.save_total_limit,       
            eval_steps=self.config.training.eval_steps,
            logging_steps=self.config.training.logging_steps,
            warmup_steps=self.config.training.warmup_steps,
            learning_rate=self.config.training.learning_rate,
            push_to_hub=self.config.training.push_to_hub,
            hub_model_id=self.config.training.hub_model_id
        )
        
        return data_collator, self.metrics, training_args
    
    def create_trainer(self, data_collator: Any, training_args: TrainingArguments, train_dataset: Any, test_dataset: Any) -> Trainer:
        logger.info("Creating trainer...")
        
        self.trainer = Trainer(
            model=self.model,
            data_collator=data_collator,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=self.processor,
            compute_metrics=self.metrics.compute_metrics,
        )
        
        return self.trainer
    
    def log_training_summary(self, vocab_dict: Dict, train_dataset: Any, 
                           test_dataset: Any, training_args: TrainingArguments) -> None:
        """Log comprehensive training summary."""
        logger.info("TRAINING SUMMARY")
        logger.info("-" * 40)
        logger.info(f"Model: {type(self.model).__name__}")
        logger.info(f"Vocabulary: {len(vocab_dict)} characters")
        logger.info(f"Train samples: {len(train_dataset):,}")
        logger.info(f"Test samples: {len(test_dataset):,}")
        
        batch_size_total = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        logger.info(f"Effective batch size: {batch_size_total}")
        logger.info(f"Learning rate: {training_args.learning_rate}")
        logger.info(f"Epochs: {training_args.num_train_epochs}")
        logger.info(f"Output directory: {training_args.output_dir}")
        
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name()
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            allocated_memory = torch.cuda.memory_allocated(0) / 1e9
            logger.info(f"Using GPU: {device_name}")
            logger.info(f"GPU Memory: {total_memory:.1f}GB total, {allocated_memory:.1f}GB allocated")
        else:
            logger.info("Using CPU (training will be slower)")
    
    def train(self) -> None:
        logger.info("Starting training...")
        logger.info("=" * 60)
        
        try:
            self.trainer.train()
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            raise
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def save_model_and_metrics(self, training_args: TrainingArguments) -> None:
        try:
            logger.info("Saving final model...")
            self.trainer.save_model()
            self.processor.save_pretrained(training_args.output_dir)
            
            self.metrics.save_metrics_history(f"{training_args.output_dir}/metrics_history.json")
            best_metrics = self.metrics.get_best_metrics()
            logger.info(f"Best WER: {best_metrics.get('best_wer', 'N/A'):.4f}")
            logger.info(f"Best CER: {best_metrics.get('best_cer', 'N/A'):.4f}")
            
            # if training_args.push_to_hub:
            #     logger.info(f"Pushing to Hub: {training_args.hub_model_id}")
            #     self.trainer.push_to_hub()
            
            logger.info(f"Training completed successfully!")
            logger.info(f"Model saved to: {training_args.output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def cleanup(self) -> None:
        """Clean up GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def run_full_training_pipeline(self) -> Tuple[Trainer, Any, Any]:
        """Run the complete training pipeline."""
        try:
            train_dataset, test_dataset, vocab_dict = self.load_datasets()
            self.load_model()
            
            train_dataset_prepared, test_dataset_prepared = self.prepare_datasets(train_dataset, test_dataset)
            
            data_collator, metrics, training_args = self.setup_training_components(vocab_dict)
            self.create_trainer(data_collator, training_args, train_dataset_prepared, test_dataset_prepared)
            
            self.log_training_summary(vocab_dict, train_dataset_prepared, test_dataset_prepared, training_args)
            self.train()
            
            # self.save_model_and_metrics(training_args) not need now 
            
            return self.trainer, self.model, self.processor
            
        finally:
            self.cleanup()