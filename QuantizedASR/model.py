from typing import Dict, Optional, Tuple
from transformers import (
    Wav2Vec2CTCTokenizer, 
    SeamlessM4TFeatureExtractor, 
    Wav2Vec2BertProcessor,
    Wav2Vec2BertForCTC
)
from configs import QuantizeConfigs


class ArmenianModelLoader:
    def __init__(self, configs: QuantizeConfigs):
        self.configs = configs
        self.tokenizer: Wav2Vec2CTCTokenizer = None
        self.vocab: Dict = None
        self.feature_extractor: SeamlessM4TFeatureExtractor = None 
        self.processor: Wav2Vec2BertProcessor = None
        self.model: Wav2Vec2BertForCTC = None

    def load_feature_extractor(self):
        self.feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
            self.configs.model.name
        )
        return self.feature_extractor

    def create_tokenizer(self, vocab_dict: Dict[str, int]):
        """Create tokenizer from vocabulary dictionary."""
        self.vocab = vocab_dict
        self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
            self.configs.model.tokenizer_path,
            unk_token=self.configs.model.unk_token,
            pad_token=self.configs.model.pad_token,
            word_delimiter_token=self.configs.model.word_delimiter_token
        )
        return self.tokenizer

    def build_processor(self):
        """Build the processor combining feature extractor and tokenizer."""
        if self.feature_extractor is None or self.tokenizer is None:
            raise ValueError("Feature extractor and tokenizer must be loaded first")
        
        self.processor = Wav2Vec2BertProcessor(
            feature_extractor=self.feature_extractor, 
            tokenizer=self.tokenizer
        )
        return self.processor

    def build_model(self):
        """Build the Wav2Vec2-BERT model for CTC."""
        if self.processor is None:
            raise ValueError("Processor must be built first")
        
        self.model = Wav2Vec2BertForCTC.from_pretrained(
            self.configs.model.name,
            attention_dropout=self.configs.model.attention_dropout,
            hidden_dropout=self.configs.model.hidden_dropout,
            feat_proj_dropout=self.configs.model.feat_proj_dropout,
            mask_time_prob=self.configs.model.mask_time_prob,
            layerdrop=self.configs.model.layerdrop,
            ctc_loss_reduction=self.configs.model.ctc_loss_reduction,
            add_adapter=self.configs.model.add_adapter,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            vocab_size=len(self.processor.tokenizer),
        )
        return self.model

    def freeze_backbone(self):
        """Freezing all layers except the classification head."""
        if self.model is None:
            raise ValueError("Model must be built first")
        
        #freeze wav2vec2_bert backbone
        for param in self.model.wav2vec2_bert.parameters():
            param.requires_grad = False
        
        #lm_head trainable
        for param in self.model.lm_head.parameters():
            param.requires_grad = True

    def push_to_hub(self, repo_name: str):
        """Push tokenizer and processor to Hugging Face Hub."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be created first")
        if self.processor is None:
            raise ValueError("Processor must be built first")
        
        self.tokenizer.push_to_hub(repo_name)
        self.processor.push_to_hub(repo_name)



    def __call__(
        self, 
        vocab_dict: Dict[str, int], 
        repo_name: Optional[str] = None,
        freeze_layers: bool = True
    ) -> Tuple[Wav2Vec2BertForCTC, Wav2Vec2BertProcessor]:
        
        #feature extractor
        self.load_feature_extractor()
        
        #tokenizer
        self.create_tokenizer(vocab_dict)
        
        #processor
        self.build_processor()
        
        #model
        self.build_model()
        
        #freeze
        if freeze_layers:
            self.freeze_backbone()
        
        #push2hub
        if repo_name:
            self.push_to_hub(repo_name)
        
        return self.model, self.processor


# Example usage
# if __name__ == "__main__":
#     from configs import QuantizeConfigs
    
#     configs = QuantizeConfigs("configs/quantize_configs.yml")
#     print(configs.model)
#     loader = ArmenianModelLoader(configs)
    
#     vocab_dict = {
#                 "ա": 1,
#                 "բ": 2,
#                 "գ": 3,
#                 "դ": 4,
#                 "ե": 5,
#                 "զ": 6,
#                 "է": 7,
#                 "ը": 8,
#                 "թ": 9,
#                 "ժ": 10,
#                 "ի": 11,
#                 "լ": 12,
#                 "խ": 13,
#                 "ծ": 14,
#                 "կ": 15,
#                 "հ": 16,
#                 "ձ": 17,
#                 "ղ": 18,
#                 "ճ": 19,
#                 "մ": 20,
#                 "յ": 21,
#                 "ն": 22,
#                 "շ": 23,
#                 "ո": 24,
#                 "չ": 25,
#                 "պ": 26,
#                 "ջ": 27,
#                 "ռ": 28,
#                 "ս": 29,
#                 "վ": 30,
#                 "տ": 31,
#                 "ր": 32,
#                 "ց": 33,
#                 "ւ": 34,
#                 "փ": 35,
#                 "ք": 36,
#                 "օ": 37,
#                 "ֆ": 38,
#                 "|": 0,
#                 "[UNK]": 39,
#                 "[PAD]": 40
#                 }
    
#     model, processor = loader(
#         vocab_dict=vocab_dict,
#         repo_name=configs.repo_name,
#         freeze_layers=True
#     )
