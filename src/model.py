import json
import torch

from dataclasses import dataclass
from typing import Dict, List, Union
from pathlib import Path
from transformers import Wav2Vec2CTCTokenizer, SeamlessM4TFeatureExtractor, Wav2Vec2BertProcessor

from _logging import log
from configs import ArmenianAudioDatasetConfig

class ModelBuilding:
    def __init__(self,configs: ArmenianAudioDatasetConfig):
        self.configs = configs
        self.tokenizer = None
        self.feature_extractor = None
        self.processor = None

    @log.log_phase("Creation Tokenizer")
    def create_tokenizer(self, vocab_dict:dict[str, int], output_path:str)->Wav2Vec2CTCTokenizer:
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        vocab_file = output_dir / "vocab.json"
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
        
        tokenizer_configs = {
            "unk_token":self.configs.special_tokens['unknown'],
            "pad_token":self.configs.special_tokens['padding'],
            "word_delimiter_token":self.configs.special_tokens['word_separator'],
            'do_lower_case':True,
        }
        self.tokenizer = Wav2Vec2CTCTokenizer(str(vocab_file), **tokenizer_configs)
        
        log.success("Tokenizer created successfully")
        return self.tokenizer

    @log.log_phase("Creation Feature Extractor")
    def create_feature_extractor(self, model_name:str=None)->SeamlessM4TFeatureExtractor:
        if model_name is None:
            model_name = getattr(self.configs, 'model_name', 'facebook/seamless-m4t-medium')
        
        self.feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(model_name)

        log.success("Feature Extractor created successfully")
        return self.feature_extractor

    @log.log_phase("Creation Processor")
    def create_processor(self)->Wav2Vec2BertProcessor:

        self.processor = Wav2Vec2BertProcessor(feature_extractor=self.feature_extractor,
                                               tokenizer=self.tokenizer)
        
        log.success("Processor created successfully")

        return self.processor
    
    def save_processor(self, output_path:str, push2hub:bool, repo_name:str=None):
        if push2hub and repo_name:
            self.processor.push_to_hub(repo_name)
            log.success(f"Processor pushed to {repo_name}")
        else:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            self.processor.save_pretrained(str(output_path))
            log.success(f"Processor saved to {output_path}")
        

    def __call__(self, vocab_dict:dict[str, int], output_path, model_name:str, 
                 push2hub:bool, repo_name:str=None, save_pretrained:bool=True)->Wav2Vec2BertProcessor:
        "Building all"

        self.create_tokenizer(vocab_dict, output_path)    

        self.create_feature_extractor(model_name)

        processor = self.create_processor()

        if push2hub and repo_name:
            self.save_processor(output_path, push2hub=True, repo_name=repo_name)
        
        elif save_pretrained:
            self.save_processor(output_path, push2hub=False)
        
        log.success("Model Pipeline Successful")

        return processor

#Just copy paseted from notebook of tutorial
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

if __name__ == "__main__":

    from loader import ArmenianAudioDataLoading

    configs = ArmenianAudioDatasetConfig()

    dataloader = ArmenianAudioDataLoading(configs=configs)

    model_builder = ModelBuilding(configs=configs)
    train_data, val_data, test_data, vocab_dict = dataloader(use_separate_validation=True)

    # HelperFunctions.show_random_samples(dataset=train_data, num_examples=1)

    processor = model_builder(vocab_dict=vocab_dict, 
                              model_name="facebook/w2v-bert-2.0", 
                              output_path="./model_outputs", 
                              push2hub=False,
                              save_pretrained=True)
    
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
