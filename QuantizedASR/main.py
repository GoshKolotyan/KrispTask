import argparse
import logging
import sys
from pathlib import Path
from warnings import filterwarnings

from traning import ArmenianASRTrainer

filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train Armenian ASR model')
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/quantize_configs.yml',
        help='Path to configuration file (default: configs/quantize_configs.yml)'
    )
    parser.add_argument(
        '--resume', 
        type=str, 
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Run setup without actual training'
    )
    return parser.parse_args()


def validate_config_file(config_path: str) -> bool:
    """Validate that config file exists."""
    if not Path(config_path).exists():
        logger.error(f"Configuration file not found: {config_path}")
        return False
    return True


def main():
    """Main training orchestration."""
    logger.info("Starting Armenian ASR Training Pipeline")
    logger.info("=" * 60)
    args = parse_arguments()
    
    if not validate_config_file(args.config):
        sys.exit(1)
    
    logger.info(f"Using configuration: {args.config}")
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
    
    trainer_instance = ArmenianASRTrainer(args.config)
    
    if args.dry_run:
        logger.info("Dry run mode - setting up pipeline without training...")
        train_dataset, test_dataset, vocab_dict = trainer_instance.load_datasets()
        trainer_instance.load_model()
        train_prep, test_prep = trainer_instance.prepare_datasets(train_dataset, test_dataset)
        data_collator, metrics, training_args = trainer_instance.setup_training_components(vocab_dict)
        trainer_instance.create_trainer(data_collator, training_args, train_prep, test_prep)
        trainer_instance.log_training_summary(vocab_dict, train_prep, test_prep, training_args)
        logger.info("Dry run completed successfully!")
        return
    
    trainer, model, processor = trainer_instance.run_full_training_pipeline()
    
    logger.info("Pipeline completed successfully!")
    return trainer, model, processor


if __name__ == "__main__":
    main()