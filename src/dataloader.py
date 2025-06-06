"""
Letters (77): ԱԲԳԴԵԶԷԸԹԺԻԼԽԾԿՀՁՂՃՄՅՆՇՈՉՊՋՌՍՎՏՐՑՒՓՔՕՖաբգդեզէըթժիլխծկհձղճմյնշոչպջռսվտրցւփքօֆև
Digits (0): 
Punctuation (6): (),-.:
Whitespace (1): [' ']
Special characters (13): `«´»՚՛՜՝՞։֊’…
"""

import re
import json
import random
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset

from configs import ArmenianAudioDatasetConfig

class ArmenianAudio(Dataset):
    def __init__(self,config: ArmenianAudioDatasetConfig):

        self.config:ArmenianAudioDatasetConfig = config
        self.train_dataset: Optional[HFDataset] = None
        self.test_dataset: Optional[HFDataset] = None
        self.vocab_dict: Optional[Dict[str, int]] = None
        self._load_dataset()

    def _load_dataset(self)->None:

        self.train_dataset = load_dataset(self.config.dataset_path, self.config.language_code, split="train+validation")
        self.test_dataset  = load_dataset(self.config.dataset_path, self.config.language_code, split="test")

        self.train_dataset = self._remove_columns(self.train_dataset)
        self.test_dataset = self._remove_columns(self.test_dataset)

    def _remove_columns(self, dataset: HFDataset) -> HFDataset:
        if self.config.remove_columns:
            return dataset.remove_columns(self.config.remove_columns)
        return dataset

    def clean_text(self, batch: dict)-> dict:
        
        #clean unusable characters
        clean_sentence = re.sub(self.config.unusable_chars_pattern, '', batch['sentence'].lower())
        batch['sentence'] = clean_sentence
        return batch
    
    def extract_vocabulary(self, batch: dict)->dict:
        all_text = ''.join(batch['sentence'])
        vocab = list(set(all_text))
        return [{"vocab": vocab, "all_text": all_text}]

    def _build_vocabulary(self)->dict[str, int]:
        train_sentences = [item['sentence'] for item in self.train_dataset]
        test_sentences = [item['sentence'] for item in self.test_dataset]
        
        all_text = ' '.join(train_sentences + test_sentences)
        vocab_set = set(all_text)
        vocab_list = sorted(list(vocab_set))
        
        vocab_dict = {char: idx for idx, char in enumerate(vocab_list)}
        
        #for special tokens
        if " " in vocab_dict:
            vocab_dict[self.config.special_tokens.get("word_separator", "|")] = vocab_dict[" "]
            del vocab_dict[" "]
        
        #special tokens
        vocab_dict[self.config.special_tokens.get("unknown", "[UNK]")] = len(vocab_dict)
        vocab_dict[self.config.special_tokens.get("padding", "[PAD]")] = len(vocab_dict)
        
        print(f"Vocabulary built with {len(vocab_dict)} unique characters")
        return vocab_dict
    
    def save_vocabulary(self)-> None:
        if not self.vocab_dict:
            return
        with open(self.config.vocab_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.vocab_dict, f, ensure_ascii=False, indent=2)
    
    def process_datasets(self) -> Dict[str, int]:
        print("Starting dataset processing...")
        
        print("Cleaning text...")
        self.train_dataset = self.train_dataset.map(self.clean_text)
        self.test_dataset  = self.test_dataset.map(self.clean_text)
        
        print("Filtering empty sentences...")
        initial_train_size = len(self.train_dataset)
        initial_test_size = len(self.test_dataset)
        
        
        print(f"Filtered train: {initial_train_size} -> {len(self.train_dataset)}")
        print(f"Filtered test: {initial_test_size} -> {len(self.test_dataset)}")
        
        self.vocab_dict = self._build_vocabulary()
        
        self.save_vocabulary()
        
        print(f"Processing complete! Vocabulary size: {len(self.vocab_dict)}")
        return self.vocab_dict
    @staticmethod
    def show_random_elements(dataset:pd.DataFrame, num_example:int):
        assert num_example <= len(dataset), "Number of examples requested exceeds dataset size."
        picks = []
        for _ in range(num_example):
            pick = random.randint(0, len(dataset) - 1)
            while pick in picks:
                pick = random.randint(0, len(dataset) - 1)
            picks.append(pick)
        df = pd.DataFrame(dataset[picks])
        return df
    
    def __len__(self) -> int:
        return len(self.train_dataset) if self.train_dataset else 0
    
    def __getitem__(self, index: int) -> Dict[str, any]:
        if not self.train_dataset:
            raise ValueError("Dataset not loaded. Please call _load_dataset() first.")        
        return self.train_dataset[index]


if __name__ == "__main__":
    # You'll need to create your config object
    config = ArmenianAudioDatasetConfig()
    
    processor = ArmenianAudio(config)
    vocab_dict = processor.process_datasets()
    
    
    # Show random samples
    samples = ArmenianAudio.show_random_elements(processor.train_dataset, 3)
    print("Random samples:", samples)