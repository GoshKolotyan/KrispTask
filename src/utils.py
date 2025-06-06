import os 
import re
import json
import random
from typing import Dict, Optional, List, Tuple, Any
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset, Dataset as HFDataset, Audio, load_from_disk

from configs import ArmenianAudioDatasetConfig


class ArmenianAudio(Dataset):
    def __init__(self, config: ArmenianAudioDatasetConfig):
        self.config: ArmenianAudioDatasetConfig = config
        self.train_dataset: Optional[HFDataset] = None
        self.test_dataset: Optional[HFDataset] = None
        self.vocab_dict: Optional[Dict[str, int]] = None
        self._load_dataset()

    def _load_dataset(self) -> None:
        print("Loading datasets...")
        
        local_train_path = "./datasets/train_dataset"
        local_test_path = "./datasets/test_dataset"
        
        if os.path.exists(local_train_path) and os.path.exists(local_test_path):
            try:
                print("Loading from local cache...")
                self.train_dataset = load_from_disk(local_train_path)
                self.test_dataset = load_from_disk(local_test_path)
                print("Successfully loaded from local cache")
            except Exception as e:
                print(f"Failed to load from cache: {e}")
                print("Falling back to remote download...")
                self._download_and_save_datasets()
        else:
            print("Local cache not found. Downloading from remote...")
            self._download_and_save_datasets()
        
        # Remove unnecessary columns
        self.train_dataset = self._remove_columns(self.train_dataset)
        self.test_dataset = self._remove_columns(self.test_dataset)
        
        print(f"Loaded {len(self.train_dataset)} training samples")
        print(f"Loaded {len(self.test_dataset)} test samples")
        
        # Check original sampling rate
        if len(self.train_dataset) > 0:
            sample_audio = self.train_dataset[0]['audio']
            print(f"Original audio sampling rate: {sample_audio['sampling_rate']}Hz")
            print(f"Original audio length: {len(sample_audio['array'])} samples")
        

    def _download_and_save_datasets(self) -> None:
        try:
            self.train_dataset = load_dataset(
                self.config.dataset_path, 
                self.config.language_code, 
                split="train+validation",
            )
            self.test_dataset = load_dataset(
                self.config.dataset_path, 
                self.config.language_code, 
                split="test"
            )
            
            os.makedirs("./datasets", exist_ok=True)
            self.train_dataset.save_to_disk("./datasets/train_dataset")
            self.test_dataset.save_to_disk("./datasets/test_dataset")
            print("Datasets saved locally for future use")
            
        except Exception as e:
            print(f"Error downloading datasets: {e}")
            raise

    def _remove_columns(self, dataset: HFDataset) -> HFDataset:
        """Remove unnecessary columns from dataset."""
        if self.config.remove_columns:
            existing_columns = set(dataset.column_names)
            columns_to_remove = [
                col for col in self.config.remove_columns 
                if col in existing_columns
            ]
            if columns_to_remove:
                print(f"Removing columns: {columns_to_remove}")
                return dataset.remove_columns(columns_to_remove)
        return dataset

    def resample_audio(self, target_sampling_rate: int = 16000) -> None:
        print(f"Resampling audio to {target_sampling_rate}Hz...")
        
        try:
            self.train_dataset = self.train_dataset.cast_column(
                "audio", Audio(sampling_rate=target_sampling_rate)
            )
            self.test_dataset = self.test_dataset.cast_column(
                "audio", Audio(sampling_rate=target_sampling_rate)
            )
            
            sample_audio = self.train_dataset[0]['audio']
                
            print(f"Resampled audio sampling rate: {sample_audio['sampling_rate']}Hz")
            print(f"Resampled audio length: {len(sample_audio['array'])} samples")
            
            print("Audio resampling completed")
            
        except Exception as e:
            print(f"Error during audio resampling: {e}")
            raise

    def clean_text(self, batch: dict) -> dict:
        clean_sentence = re.sub(
            self.config.unusable_chars_pattern, 
            '', 
            batch['sentence']
        ).lower().strip()
        
        clean_sentence = clean_sentence.replace(' ', '|')
        
        batch['sentence'] = clean_sentence
        return batch

    def _build_vocabulary(self) -> Dict[str, int]:
        print("Building vocabulary...")
        
        train_sentences = [item['sentence'] for item in self.train_dataset]
        test_sentences = [item['sentence'] for item in self.test_dataset]
        
        all_text = ' '.join(train_sentences + test_sentences)
        vocab_set = set(all_text)
        vocab_list = sorted(list(vocab_set))
        
        vocab_dict = {char: idx for idx, char in enumerate(vocab_list)}
        
        if " " in vocab_dict:
            space_idx = vocab_dict[" "]
            del vocab_dict[" "]  
            vocab_dict["|"] = space_idx  
        
        current_idx = len(vocab_dict)
        
        word_separator = self.config.special_tokens.get("word_separator", "|")
        if word_separator not in vocab_dict:
            vocab_dict[word_separator] = current_idx
            current_idx += 1
            
        unknown_token = self.config.special_tokens.get("unknown", "[UNK]")
        if unknown_token not in vocab_dict:
            vocab_dict[unknown_token] = current_idx
            current_idx += 1
            
        padding_token = self.config.special_tokens.get("padding", "[PAD]")
        if padding_token not in vocab_dict:
            vocab_dict[padding_token] = current_idx
        
        print(f"Vocabulary built with {len(vocab_dict)} unique characters")
        print(f"Word delimiter '|' has index: {vocab_dict.get('|', 'NOT FOUND')}")
        print(f"Unknown token '[UNK]' has index: {vocab_dict.get('[UNK]', 'NOT FOUND')}")
        return vocab_dict
    
    def save_vocabulary(self) -> None:
        """Save vocabulary to JSON file."""
        if not self.vocab_dict:
            print("No vocabulary to save")
            return
            
        try:
            vocab_dir = os.path.dirname(self.config.vocab_file_path)
            if vocab_dir:
                os.makedirs(vocab_dir, exist_ok=True)
                
            with open(self.config.vocab_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.vocab_dict, f, ensure_ascii=False, indent=2)
            print(f"Vocabulary saved to {self.config.vocab_file_path}")
        except Exception as e:
            print(f"Error saving vocabulary: {e}")
    
    def load_vocabulary(self) -> Optional[Dict[str, int]]:
        if os.path.exists(self.config.vocab_file_path):
            try:
                with open(self.config.vocab_file_path, 'r', encoding='utf-8') as f:
                    vocab_dict = json.load(f)
                print(f"Vocabulary loaded from {self.config.vocab_file_path}")
                return vocab_dict
            except Exception as e:
                print(f"Error loading vocabulary: {e}")
                return None
        return None
    
    def process_datasets(self) -> Dict[str, int]:
        print("Starting dataset processing...")
        
        existing_vocab = self.load_vocabulary()
        if existing_vocab:
            self.vocab_dict = existing_vocab
            print(f"Using existing vocabulary with {len(self.vocab_dict)} characters")
        
        print("Cleaning text...")
        self.train_dataset = self.train_dataset.map(self.clean_text)
        self.test_dataset = self.test_dataset.map(self.clean_text)
        print("Text cleaning completed!")
        
        self.resample_audio(target_sampling_rate=16000)
        
        if not self.vocab_dict:
            self.vocab_dict = self._build_vocabulary()
            self.save_vocabulary()
        
        print(f"Processing complete! Vocabulary size: {len(self.vocab_dict)}")
        return self.vocab_dict

    def __len__(self) -> int:
        return len(self.train_dataset) if self.train_dataset else 0
    
    def __getitem__(self, index: int) -> Dict[str, any]:
        if not self.train_dataset:
            raise ValueError("Dataset not loaded.")
        if index >= len(self.train_dataset):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.train_dataset)}")
        return self.train_dataset[index]


def test_preprocessing():
    """Test the complete preprocessing pipeline."""
    print("=== Testing Audio Preprocessing Pipeline ===")
    
    try:
        config = ArmenianAudioDatasetConfig()
        processor = ArmenianAudio(config)
        
        processor.process_datasets()
        
        
        if len(processor.train_dataset) > 0:
            print("\n=== Testing Single Sample Preprocessing ===")
            sample = processor.train_dataset[0]
            print(f"Original sentence: '{sample['sentence']}'")
            print(f"Audio shape: {sample['audio']['array'].shape}")
            print(f"Audio sampling rate: {sample['audio']['sampling_rate']}Hz")
            
            # Test tokenization if tokenizer is available
            try:
                from transformers import Wav2Vec2CTCTokenizer
                
                if os.path.exists("./vocab.json"):
                    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
                        "./", 
                        unk_token="[UNK]", 
                        pad_token="[PAD]", 
                        word_delimiter_token="|"
                    )
                    
                    tokenized_result = tokenizer.encode(sample['sentence'])
                    print(f"Token IDs: {tokenized_result}")
                    
                    decoded_text = tokenizer.decode(tokenized_result)
                    print(f"Decoded text: '{decoded_text}'")
                    
                    chars_in_sentence = set(sample['sentence'])
                    print(f"Characters in sentence: {sorted(chars_in_sentence)}")
                else:
                    print("Tokenizer vocab.json not found - skipping tokenization test")
                    
            except ImportError:
                print("Transformers library not available - skipping tokenization test")
            except Exception as e:
                print(f"Tokenization test failed: {e}")
        
        print("\nPreprocessing test completed successfully!")
        
    except Exception as e:
        print(f"Preprocessing test failed: {e}")
        raise


if __name__ == "__main__":
    test_preprocessing()