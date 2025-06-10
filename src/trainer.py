import argparse

from pathlib import Path
from huggingface_hub import notebook_login
from helper import HelperFunctions as Helper

from _logging import log
from metrics import Metrics
from loader import ArmenianAudioDataLoading
from configs import ArmenianAudioDatasetConfig
from model import ModelBuilding, DataCollatorCTCWithPadding
from transformers import Wav2Vec2BertForCTC, Trainer, TrainingArguments

class ArmenianASRTainer:
    def __init__(self, args, configs:ArmenianAudioDatasetConfig):
        self.configs = configs
        self.args = args

        self.processor = None
        self.model = None
        self.tokenizer = None
        self.data_loader = None
        self.metrics_calculator = None 
        
        self.val_dataset = None
        self.train_dataset = None
        self.test_dataset = None 

        self.data_collator = None
        self.trainer = None 

        self.traning_results = {}
        self.evaluation_results = {}

        log.info("ASR is Initlized")
        log.log_config(self.configs, "Model Configs")
        log.log_config(self.args, "Model Arguments")

    @log.log_phase("Evironment setup")
    def setup_env(self):
        log.info("Setting up train env")

        if self.args.push2hub:
            log.progress("Authentication with Hugging Face Hub")
            try:
                notebook_login()
            except Exception as e:
                log.error(f"Hub error failed: {e}")
                if self.args.push2hub:
                    raise
        
        Helper.set_random_seed(self.args.seed)
        self.configs.output_dir = Path(self.args.output_dir)

        log.success("Enironment setup completed")
    
    @log.log_phase("Loading Data")
    def load_and_prepare_data(self):

        self.data_loader = ArmenianAudioDataLoading(self.configs)

        self.train_dataset, self.val_dataset, self.test_dataset, self.vocab_dict = self.data_loader(use_separate_validation=True)
        dataset = {
            "train": self.train_dataset,
            'test': self.test_dataset
        }    

        if self.val_dataset:
            dataset['validation'] = self.val_dataset
        
        log.success("Data loading and preporcessing complateed")

    @log.log_phase("Loading Processor")
    def load_processor(self):
        self.build_model = ModelBuilding(configs=configs)

        self.processor = self.build_model(vocab_dict=self.vocab_dict,
                                          output_path=self.args.output_dir,
                                          model_name=self.configs.model_name,
                                          save_pretrained=True,
                                          push2hub=self.args.push2hub,
                                          repo_name=self.args.repo_name)
        
        self.metrics_calculator = Metrics(self.processor)

        self.data_collator = DataCollatorCTCWithPadding(
            processor=self.processor,
            padding=True,
        )

        log.success("Loaded processor successfully")

    @log.log_phase("Model Setup")
    def setup_model(self):
        model_configs = self.configs.get_model_args_dict

        self.model = Wav2Vec2BertForCTC.from_pretrained(
            self.configs.model_name,
            pad_token_id = self.processor.tokenizer.pad_token_id,
            vocab_size = len(self.processor.tokenizer),
            **model_configs
        )

        log.success("model loaded successfully")
    
    @log.log_phase("Feature Processing")
    def process_features(self):
        
        def prepare_dataset(batch):
            """Process batch for training"""
            audio = batch["audio"]
            
            batch["input_features"] = self.processor(
                audio["array"], 
                sampling_rate=audio["sampling_rate"]
            ).input_features[0]
            
            batch["input_length"] = len(batch["input_features"])
            
            batch["labels"] = self.processor(text=batch["sentence"]).input_ids
            
            return batch
                

        
        # Process all datasets
        log.progress("Processing training dataset...")
        self.train_dataset = self.train_dataset.map(
            prepare_dataset,
            remove_columns=self.train_dataset.column_names,
            desc="Processing train features"
        )
        
        if self.val_dataset:
            log.progress("Processing validation dataset...")
            self.val_dataset = self.val_dataset.map(
                prepare_dataset,
                remove_columns=self.val_dataset.column_names,
                desc="Processing validation features"
            )
        
        log.progress("Processing test dataset...")
        self.test_dataset = self.test_dataset.map(
            prepare_dataset,
            remove_columns=self.test_dataset.column_names,
            desc="Processing test features"
        )
        
        log.success("Feature processing completed")

    @log.log_phase("Training Setup")
    def setup_training(self):
        
        training_config = self.configs.get_training_args_dict
        
        arg_overrides = {
            "output_dir": self.args.output_dir,
            "num_train_epochs": self.args.num_train_epochs,
            "per_device_train_batch_size": self.args.per_device_train_batch_size,
            "per_device_eval_batch_size": self.args.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
            "learning_rate": self.args.learning_rate,
            "warmup_steps": self.args.warmup_steps,
            "save_steps": self.args.save_steps,
            "eval_steps": self.args.eval_steps,
            "logging_steps": self.args.logging_steps,
            "push_to_hub": self.args.push2hub,
        }
        
        for key, value in arg_overrides.items():
            if value is not None:
                training_config[key] = value
        
        # Create training arguments
        training_args = TrainingArguments(**training_config)
        
        log.info("📋 Training Configuration:")
        for key, value in training_config.items():
            log.info(f"   {key}: {value}")
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset if self.val_dataset else self.test_dataset,
            data_collator=self.data_collator,
            compute_metrics=self.metrics_calculator.compute_metrics,
            tokenizer=self.processor.feature_extractor,  # For saving
        )
        
        log.success("Training setup completed")

    @log.log_phase("Model Training")
    def train(self):
        log.progress("Beginning training process...")
        self.training_results = self.trainer.train()
        
        # Log training completion
        log.success("Training completed successfully!")
        
        # Save final model
        if self.args.push_to_hub:
            log.progress("Pushing model to Hub...")
            self.trainer.push_to_hub()
            log.success(f"Model pushed to {self.args.repo_name}")
        else:
            log.progress("Saving model locally...")
            self.trainer.save_model()
            log.success(f"Model saved to {self.args.output_dir}")
        
        # Save training metrics
        metrics_path = Path(self.args.output_dir) / "training_metrics.json"
        self.metrics_calculator.save_metrics_history(str(metrics_path))
        
        return self.training_results

    @log.log_phase("Building pipline")
    def build_pipline(self):
        self.setup_env()

        self.load_and_prepare_data()

        self.load_processor()

        self.setup_model()

        self.process_features()

        self.setup_training()

        # self.train()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Armenian ASR model with Wav2Vec2-BERT")
    
    # Model and data arguments
    parser.add_argument("--output_dir", type=str, default="./armenian-w2v2-bert",
                       help="Output directory for model and logs")
    parser.add_argument("--repo_name", type=str, default="armenian-w2v2-bert-cv16",
                       help="Repository name for HuggingFace Hub")
    parser.add_argument("--use_separate_validation", action="store_true", default=True,
                       help="Use separate validation set instead of merging with train")
    parser.add_argument("--apply_quality_filters", action="store_true", default=True,
                       help="Apply data quality filtering")
    
    # Training hyperparameters
    parser.add_argument("--num_train_epochs", type=int, default=5,
                       help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8,
                       help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                       help="Evaluation batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                       help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                       help="Warmup steps")
    parser.add_argument("--save_steps", type=int, default=1000,
                       help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=500,
                       help="Evaluate every N steps")
    parser.add_argument("--logging_steps", type=int, default=100,
                       help="Log every N steps")
    
    # System arguments
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                       help="Number of dataloader workers")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu)")
    
    # Misc arguments
    parser.add_argument("--push2hub", action="store_true",
                       help="Push model to HuggingFace Hub")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    parser.add_argument("--run_evaluation_only", action="store_true",
                       help="Only run evaluation on existing model")
    parser.add_argument("--max_audio_length", type=int, default=None,
                       help="Maximum audio length for padding")
    
    return parser.parse_args()
if __name__ == "__main__":

    args = parse_arguments()

    configs = ArmenianAudioDatasetConfig()

    ASR = ArmenianASRTainer(configs=configs, args=args)

    ASR.build_pipline()