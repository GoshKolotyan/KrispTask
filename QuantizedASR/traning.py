import torch
from transformers import TrainingArguments, Trainer

# Load required modules
from configs import QuantizeConfigs
from dataloading import ArmenianDataLoader
from model import QuantizedArmenianModelLoader
from helpers import prepare_dataset, DataCollatorCTCWithPadding
from metrics import Metrics 
from warnings import filterwarnings
filterwarnings("ignore")

def main():
    
    print("Starting Armenian ASR Training Pipeline")
    print("=" * 60)
    
    print("Loading configuration...")
    config = QuantizeConfigs("configs/quantize_configs.yml")
    
    repo_name = getattr(config, 'repo_name', './wav2vec2-armenian-frozen')
    print(f"Output directory: {repo_name}")
    
    print("\nLoading datasets...")
    dataloader = ArmenianDataLoader(config)
    train_dataset, test_dataset, vocab_dict = dataloader(verbose=True)
    
    print(f"Training examples: {len(train_dataset):,}")
    print(f"Test examples: {len(test_dataset):,}") 
    print(f"Vocabulary size: {len(vocab_dict)}")
    
    sample = train_dataset[0]
    print(f"Sample text: '{sample['sentence']}...'")
    
    print("\nLoading model...")
    model_loader = QuantizedArmenianModelLoader(config)
    
    model, processor = model_loader(
        # repo_name=repo_name if hasattr(config, 'push_to_hub') else None
        add_adapters=True
    )
    
    
    print("\nPreparing datasets...")
    
    def prepare_batch(batch):
        return prepare_dataset(batch, processor)
    
    print("Processing training data...")
    common_voice_train = train_dataset.map(
        prepare_batch, 
        remove_columns=train_dataset.column_names,
        desc="Preparing train dataset"
    )
    
    print("Processing test data...")
    common_voice_test = test_dataset.map(
        prepare_batch, 
        remove_columns=test_dataset.column_names,
        desc="Preparing test dataset"
    )

    sample = common_voice_train[0]
    print(f"Sample text after preprocessing: '{sample.keys()}...'")
    
    
    print("Dataset preparation completed successfully")
    
    print("\nSetting up training components...")
    
    data_collator = DataCollatorCTCWithPadding(processor=processor)
    
    metrics = Metrics(processor=processor)
    

    training_args = TrainingArguments(
        output_dir=repo_name,
        group_by_length=True,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        eval_strategy="steps",
        num_train_epochs=2,
        gradient_checkpointing=True,
        fp16=False,
        
        save_strategy="no",     
        save_total_limit=2,       
        
        eval_steps=10,
        logging_steps=10,
        warmup_steps=500,
        learning_rate=1e-4,
        push_to_hub=getattr(config, 'push_to_hub', True),
        hub_model_id=repo_name if getattr(config, 'push_to_hub', False) else None,
    )
    print("Training arguments configured successfully")
    
    print("\nCreating trainer...")
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=common_voice_train,
        eval_dataset=common_voice_test,
        tokenizer=processor,
        compute_metrics=metrics.compute_metrics,
    )
    
    print("Trainer created successfully")
    print("\nTRAINING SUMMARY")
    print("-" * 40)
    print(f"Model: {type(model).__name__}")
    print(f"Vocabulary: {len(vocab_dict)} characters")
    print(f"Train samples: {len(common_voice_train):,}")
    print(f"Test samples: {len(common_voice_test):,}")
    batch_size_total = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    print(f"Effective batch size: {batch_size_total}")
    print(f"Learning rate: {training_args.learning_rate}")
    print(f"Epochs: {training_args.num_train_epochs}")
    print(f"Output directory: {training_args.output_dir}")
    
    print(f"\nStarting training...")
    print("=" * 60)
    
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Using GPU: {device_name}")
        print(f"GPU Memory: {total_memory:.1f}GB")
    else:
        print("Using CPU (training will be slower)")
    
    trainer.train()
    
    print(f"\nSaving final model...")
    trainer.save_model()
    processor.save_pretrained(training_args.output_dir)
    
    if training_args.push_to_hub:
        print(f"Pushing to Hub: {training_args.hub_model_id}")
        trainer.push_to_hub()
    
    print(f"\nTraining completed successfully!")
    print(f"Model saved to: {training_args.output_dir}")
    metrics.save_metrics_history(f"{training_args.output_dir}/metrics_history.json")
    best_metrics = metrics.get_best_metrics()
    print(f"Best WER: {best_metrics.get('best_wer', 'N/A'):.4f}")
    print(f"Best CER: {best_metrics.get('best_cer', 'N/A'):.4f}")
    
    return trainer, model, processor



if __name__ == "__main__":
    trainer, model, processor = main()

    trainer.push_to_hub()
    
    if trainer is not None:
        print(f"\nTraining completed! Model ready for inference.")
    else:
        print(f"\nTraining failed. Check the error logs above.")