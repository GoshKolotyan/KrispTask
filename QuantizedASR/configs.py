import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    output_dir: str
    group_by_length: bool
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    eval_strategy: str
    num_train_epochs: int
    gradient_checkpointing: bool
    fp16: bool
    save_steps: int
    eval_steps: int
    logging_steps: int
    learning_rate: float
    warmup_steps: int
    save_total_limit: int
    push_to_hub: bool

@dataclass
class ModelConfig:
    name: str
    attention_dropout: float
    hidden_dropout: float
    feat_proj_dropout: float
    mask_time_prob: float
    layerdrop: float
    ctc_loss_reduction: str
    unk_token: str
    pad_token: str
    word_delimiter_token: str
    tokenizer_path: str
    add_adapter: bool
    freeze_feature_layers: bool

@dataclass
class DatasetConfig:
    path: str
    language_code: str
    sampling_rate: int

@dataclass
class LoRAConfig:
    rank: int
    alpha: int
    target_modules: list
    dropout: float

@dataclass
class QuantizationConfig:  # Fixed typo: QunatizeConfig -> QuantizationConfig
    enabled: bool           # Fixed: enable -> enabled
    bits: int              # Fixed: bites -> bits  
    quant_type: str
    double_quant: bool     # Fixed: download -> double_quant
    compute_dtype: str
    int8_threshold: float

class QuantizeConfigs:
    """Loads and manages YAML configuration files"""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self._config_data = None
        self.load_config()

    def load_config(self) -> None:
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(
                    f"Configuration file not found: {self.config_path}"
                )

            with open(self.config_path, "r", encoding="utf-8") as file:
                self._config_data = yaml.safe_load(file)

            print(f"Configuration loaded successfully from {self.config_path}")

        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading configuration: {e}")

    @property
    def training(self) -> TrainingConfig:
        training_data = self._config_data.get("training")
        return TrainingConfig(**training_data)

    @property
    def model(self) -> ModelConfig:
        model_data = self._config_data.get("model")
        return ModelConfig(**model_data)

    @property
    def dataset(self) -> DatasetConfig:
        dataset_data = self._config_data.get("dataset")
        return DatasetConfig(**dataset_data)

    @property
    def repo_name(self) -> str:
        return self._config_data.get("repository").get("name", "")

    @property
    def chars_to_remove_regex(self) -> str:
        """Get text processing regex pattern"""
        return self._config_data.get("text_processing").get(
            "chars_to_remove_regex", "")

    @property
    def hf_token(self) -> Optional[str]:
        token = os.getenv("HF_TOKEN")
        if token:
            return token
        return self._config_data.get("authentication").get("hf_token")
    
    @property
    def lora(self) -> Optional[LoRAConfig]:
        lora_data = self._config_data.get("lora")
        if not lora_data:
            return None
        return LoRAConfig(**lora_data)
    
    @property
    def quantization(self) -> Optional[QuantizationConfig]:  
        quantization_data = self._config_data.get("quantization") 
        if not quantization_data:
            return None
        return QuantizationConfig(**quantization_data)

    def get_raw_config(self) -> Dict[str, Any]:
        return self._config_data.copy()

    def get_section(self, section_name: str) -> Dict[str, Any]:
        return self._config_data.get(section_name)

    def __repr__(self) -> str:
        return f"QuantizeConfigs(config_path='{self.config_path}', sections={list(self._config_data.keys())})"