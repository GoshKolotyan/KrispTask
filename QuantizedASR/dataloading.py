import os
import re
from typing import Dict, Tuple, Optional
from datasets import load_dataset, Audio, Dataset
from pathlib import Path

from helpers import (
    show_random_elements, remove_special_characters, clean_armenian_text,
    extract_all_chars, create_vocabulary_dict,
    save_vocabulary, load_vocabulary, remove_unwanted_columns
)
from configs import QuantizeConfigs


class ArmenianDataLoader:

    def __init__(self, config: QuantizeConfigs, cache_dir: str = "/tmp/hf_datasets_cache"):
        self.config = config
        self.cache_dir = cache_dir
        self.train_dataset = None
        self.test_dataset = None
        self.vocab_dict = None
        
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    def load_datasets(self) -> Tuple[Dataset, Dataset]:
        try:
            self.train_dataset = load_dataset(
                self.config.dataset.path,
                self.config.dataset.language_code,
                split="train+validation",
                cache_dir=self.cache_dir,
                trust_remote_code=True,
                verification_mode="no_checks"
            )
            
            self.test_dataset = load_dataset(
                self.config.dataset.path,
                self.config.dataset.language_code,
                split="test",
                cache_dir=self.cache_dir,
                trust_remote_code=True,
                verification_mode="no_checks"
            )
            
            return self.train_dataset, self.test_dataset
            
        except Exception as e:
            raise RuntimeError(f"Failed to load datasets: {e}")
    
    def preprocess_datasets(self) -> Tuple[Dataset, Dataset]:
        if self.train_dataset is None or self.test_dataset is None:
            raise ValueError("Datasets not loaded. Call load_datasets() first.")
        
        #remove  columns
        self.train_dataset = remove_unwanted_columns(self.train_dataset)
        self.test_dataset = remove_unwanted_columns(self.test_dataset)
        
        #remove special characters
        chars_to_remove_regex = self.config.chars_to_remove_regex
        self.train_dataset = self.train_dataset.map(
            lambda batch: remove_special_characters(batch, chars_to_remove_regex)
        )
        self.test_dataset = self.test_dataset.map(
            lambda batch: remove_special_characters(batch, chars_to_remove_regex)
        )
        
        #normalize Armenian text
        self.train_dataset = self.train_dataset.map(clean_armenian_text)
        self.test_dataset = self.test_dataset.map(clean_armenian_text)
        
        #set audio sampling rate
        self.train_dataset = self.train_dataset.cast_column(
            "audio", Audio(sampling_rate=self.config.dataset.sampling_rate)
        )
        self.test_dataset = self.test_dataset.cast_column(
            "audio", Audio(sampling_rate=self.config.dataset.sampling_rate)
        )
        
        return self.train_dataset, self.test_dataset
    
    def create_vocabulary(self, save_path: str = "vocab.json") -> Dict[str, int]:
        if self.train_dataset is None or self.test_dataset is None:
            raise ValueError("Datasets not preprocessed. Call preprocess_datasets() first.")
        
        vocab_train = self.train_dataset.map(
            extract_all_chars,
            batched=True,
            batch_size=-1,
            keep_in_memory=True,
            remove_columns=self.train_dataset.column_names
        )
        
        vocab_test = self.test_dataset.map(
            extract_all_chars,
            batched=True,
            batch_size=-1,
            keep_in_memory=True,
            remove_columns=self.test_dataset.column_names
        )
        
        self.vocab_dict = create_vocabulary_dict(vocab_train, vocab_test)
        
        if save_path:
            save_vocabulary(self.vocab_dict, save_path)
        
        return self.vocab_dict
    
    def load_existing_vocabulary(self, vocab_path: str) -> Dict[str, int]:
        self.vocab_dict = load_vocabulary(vocab_path)
        return self.vocab_dict
    
    def __repr__(self) -> str:
        status = "loaded" if self.train_dataset is not None else "not loaded"
        vocab_status = f", vocab_size={len(self.vocab_dict)}" if self.vocab_dict else ""
        
        return (f"ArmenianDataLoader(dataset={self.config.dataset.path}, "
                f"language={self.config.dataset.language_code}, "
                f"status={status}{vocab_status})")

    def __call__(
        self, 
        vocab_path: Optional[str] = None,
        save_vocab: bool = True,
        verbose: bool = False
    ) -> Tuple[Dataset, Dataset, Dict[str, int]]:
        
        if verbose:
            print("Loading datasets...")
        self.load_datasets()
        
        if verbose:
            print("Preprocessing datasets...")
        self.preprocess_datasets()
        
        if vocab_path and os.path.exists(vocab_path):
            if verbose:
                print(f"Loading existing vocabulary from {vocab_path}")
            self.load_existing_vocabulary(vocab_path)
        else:
            if verbose:
                print("Creating new vocabulary...")
            vocab_save_path = "vocab.json" if save_vocab else None
            self.create_vocabulary(vocab_save_path)

        
        return self.train_dataset, self.test_dataset, self.vocab_dict


if __name__ == "__main__":
    config = QuantizeConfigs("configs/quantize_configs.yml")
    dataloader = ArmenianDataLoader(config)
    
    train_dataset, test_dataset, vocab_dict = dataloader(verbose=True)
    
    print(f"Training dataset: {len(train_dataset)} examples")
    print(f"Test dataset: {len(test_dataset)} examples") 
    print(f"Vocabulary: {len(vocab_dict)} characters")
    
    sample = train_dataset[0]
    print(f"Sample: '{sample['sentence'][:50]}...'")