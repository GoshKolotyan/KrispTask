from pathlib import Path
from huggingface_hub import notebook_login

from _logging import log
from metrics import Metrics
from helper import HelperFunctions as Helper
from loader import ArmenianAudioDataLoading
from configs import ArmenianAudioDatasetConfig
from model import ModelBuilding, DataCollatorCTCWithPadding
from transformers import Wav2Vec2BertForCTC, Trainer, TrainingArguments

class ArmenianASRTainer:
    def __init__(self, configs:ArmenianAudioDatasetConfig):
        self.configs = configs
        self.model_args = self.configs.get_model_args_dict  # Model config
        self.args = self.configs.script_args  # Script arguments

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

    @log.log_phase("Evironment setup")
    def setup_env(self):
        log.info("Setting up train env")

        if self.args['push2hub']:
            log.progress("Authentication with Hugging Face Hub")
            try:
                notebook_login()
            except Exception as e:
                log.error(f"Hub error failed: {e}")
                if self.args["push2hub"]:
                    raise
        
        Helper.set_random_seed(self.args["seed"])
        self.configs.output_dir = Path(self.args["output_dir"])

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
        self.build_model = ModelBuilding(configs=self.configs)

        self.processor = self.build_model(vocab_dict=self.vocab_dict,
                                          output_path=self.args["output_dir"],
                                          model_name=self.configs.model_name,
                                          save_pretrained=True,
                                          push2hub=self.args["push2hub"],
                                          repo_name=self.args["repo_name"])
        
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
        if self.args["push2hub"]:
            log.progress("Pushing model to Hub...")
            self.trainer.push_to_hub()
            log.success(f"Model pushed to {self.args['repo_name']}")
        else:
            log.progress("Saving model locally...")
            self.trainer.save_model()
            log.success(f"Model saved to {self.args['output_dir']}")
        
        # Save training metrics
        metrics_path = Path(self.args["output_dir"]) / "training_metrics.json"
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

        self.train()


# if __name__ == "__main__":

#     configs = ArmenianAudioDatasetConfig()

#     ASR = ArmenianASRTainer(configs=configs)

#     ASR.build_pipline()