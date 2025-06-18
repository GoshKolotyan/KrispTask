import torch
import re
import json
import random
from typing import Dict, List, Any
from datasets import Dataset
from transformers import Wav2Vec2BertProcessor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from datasets import ClassLabel, Dataset
import random
import pandas as pd
from IPython.display import display, HTML


def prepare_dataset(batch, processor):
    audio = batch["audio"]
    batch["input_features"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["input_length"] = len(batch["input_features"])

    batch["labels"] = processor(text=batch["sentence"]).input_ids
    return batch



def show_random_elements(dataset: Dataset, num_examples: int = 10) -> None:
    print(f"\n📋 Showing {num_examples} random examples from dataset:")
    print("=" * 80)
    
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)
    
    for pick in picks:
        example = dataset[pick]
        print(f"Example {pick}:")
        for key, value in example.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"  {key}: {value[:100]}...")
            else:
                print(f"  {key}: {value}")
        print("-" * 40)
    print("=" * 80)


def remove_special_characters(batch: Dict[str, Any], chars_to_remove_regex: str) -> Dict[str, Any]:
    batch["sentence"] = re.sub(chars_to_remove_regex, '', batch["sentence"]).lower()
    return batch


def normalize_armenian_text(text: str) -> str:
    # Replace 'և' with 'եվ'
    text = text.replace('և', 'եվ')
    
    # Replace 'ու' with 'ւ'  
    text = text.replace('ու', 'ւ')

    return text


def clean_armenian_text(example: Dict[str, Any]) -> Dict[str, Any]:

    example["sentence"] = normalize_armenian_text(example["sentence"])
    return example


def extract_all_chars(batch: Dict[str, List[str]]) -> Dict[str, List[List[str]]]:

    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


def create_vocabulary_dict(vocab_train: Dataset, vocab_test: Dataset) -> Dict[str, int]:

    vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
    
    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
    
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    
    return vocab_dict


def save_vocabulary(vocab_dict: Dict[str, int], filepath: str = 'vocab.json') -> None:

    try:
        with open(filepath, 'w', encoding='utf-8') as vocab_file:
            json.dump(vocab_dict, vocab_file, ensure_ascii=False, indent=2)
        print(f"Vocabulary saved to {filepath}")
    except Exception as e:
        print(f"Error saving vocabulary: {e}")


def load_vocabulary(filepath: str) -> Dict[str, int]:

    try:
        with open(filepath, 'r', encoding='utf-8') as vocab_file:
            vocab_dict = json.load(vocab_file)
        print(f"Vocabulary loaded from {filepath}")
        return vocab_dict
    except Exception as e:
        print(f"Error loading vocabulary: {e}")
        return {}


def remove_unwanted_columns(dataset: Dataset, columns_to_remove: List[str] = None) -> Dataset:

    if columns_to_remove is None:
        columns_to_remove = [
            "accent", "age", "client_id", "down_votes", 
            "gender", "locale", "segment", "up_votes"
        ]
    
    existing_columns = [col for col in columns_to_remove if col in dataset.column_names]
    
    dataset = dataset.remove_columns(existing_columns)
    print(f"Removed columns: {existing_columns}")
    
    return dataset


#Copied from tutorial
@dataclass
class DataCollatorCTCWithPadding:

    processor: Wav2Vec2BertProcessor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

# data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)