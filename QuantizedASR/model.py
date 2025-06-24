# main_loader.py
import torch
import logging
from torch import nn
from typing import Optional, Tuple
from transformers import (
    Wav2Vec2CTCTokenizer,
    SeamlessM4TFeatureExtractor,
    Wav2Vec2FeatureExtractor, # Added for fallback
    Wav2Vec2BertProcessor, 
    Wav2Vec2BertForCTC,
    BitsAndBytesConfig
)
# Assuming you have a configs.py file
from configs import QuantizeConfigs
# Ensure you have the latest versions:
# pip install -U torch bitsandbytes transformers peft accelerate
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuantizedArmenianModelLoader:
    """
    A professional model loader for quantized Armenian ASR models.
    Supports 4-bit and 8-bit quantization using BitsAndBytes and PEFT for training.
    """
    
    def __init__(self, configs: QuantizeConfigs):
        """
        Initialize the model loader with configuration.
        
        Args:
            configs: A QuantizeConfigs object containing model, quantization, and LoRA settings.
        """
        self.configs = configs
        self.tokenizer: Optional[Wav2Vec2CTCTokenizer] = None
        self.feature_extractor: Optional[Wav2Vec2FeatureExtractor] = None 
        self.processor: Optional[Wav2Vec2BertProcessor] = None
        self.model: Optional[Wav2Vec2BertForCTC] = None
        self.quantization_config = self._create_quantization_config()

    def _create_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """
        Create BitsAndBytes quantization configuration after verifying system support.
        
        Returns:
            BitsAndBytesConfig object if quantization is enabled and supported, None otherwise.
        """
        quant_config = self.configs.quantization
        
        if not quant_config or not quant_config.enabled:
            logger.info("Quantization is disabled in the configuration.")
            return None
            
        try:
            import bitsandbytes as bnb
            logger.info(f"Successfully imported bitsandbytes version: {bnb.__version__}")
        except ImportError:
            logger.error("bitsandbytes library not found. Please install it via 'pip install bitsandbytes'.")
            return None
        
        if not torch.cuda.is_available():
            logger.error("CUDA is not available. Quantization requires a compatible NVIDIA GPU.")
            return None
        
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Detected GPU: {torch.cuda.get_device_name(0)}")

        if quant_config.bits == 4:
            compute_dtype = getattr(torch, quant_config.compute_dtype, torch.float32)
            logger.info(f"Creating 4-bit quantization config (compute dtype: {quant_config.compute_dtype}).")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quant_config.quant_type,
                bnb_4bit_use_double_quant=quant_config.double_quant,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        elif quant_config.bits == 8:
            logger.info("Creating 8-bit quantization config.")
            return BitsAndBytesConfig(load_in_8bit=True)
        else:
            logger.error(f"Unsupported quantization bits: {quant_config.bits}. Only 4 and 8 are supported.")
            return None

    def load_feature_extractor(self) -> Wav2Vec2FeatureExtractor:
        """Load and return the feature extractor, with a fallback mechanism."""
        try:
            self.feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(self.configs.model.name)
            logger.info("SeamlessM4TFeatureExtractor loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load SeamlessM4TFeatureExtractor ({e}). Falling back to Wav2Vec2FeatureExtractor.")
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.configs.model.name, trust_remote_code=True)
            logger.info("Wav2Vec2FeatureExtractor loaded as a fallback.")
        return self.feature_extractor

    def create_tokenizer(self, vocab_path: str) -> Wav2Vec2CTCTokenizer:
        """Create tokenizer from a specified vocabulary file."""
        model_configs = self.configs.model
        self.tokenizer = Wav2Vec2CTCTokenizer(
            vocab_file=vocab_path,
            unk_token=model_configs.unk_token,
            pad_token=model_configs.pad_token,
            word_delimiter_token=model_configs.word_delimiter_token,
        )
        logger.info(f"Tokenizer created with {self.tokenizer.vocab_size} tokens from '{vocab_path}'.")
        return self.tokenizer

    def build_processor(self) -> Wav2Vec2BertProcessor:
        """Build processor by combining the feature extractor and tokenizer."""
        if not self.feature_extractor or not self.tokenizer:
            raise RuntimeError("Feature extractor and tokenizer must be loaded before building the processor.")
        
        self.processor = Wav2Vec2BertProcessor(
            feature_extractor=self.feature_extractor, 
            tokenizer=self.tokenizer
        )
        logger.info("Processor built successfully.")
        return self.processor

    def build_model(self) -> Wav2Vec2BertForCTC:
        """Load the Wav2Vec2-BERT model, applying quantization if configured."""
        if not self.processor:
            raise RuntimeError("Processor must be built before loading the model.")
        
        model_kwargs = {
            "attention_dropout": self.configs.model.attention_dropout,
            "hidden_dropout": self.configs.model.hidden_dropout,
            "feat_proj_dropout": self.configs.model.feat_proj_dropout,
            "mask_time_prob": self.configs.model.mask_time_prob,
            "layerdrop": self.configs.model.layerdrop,
            "ctc_loss_reduction": self.configs.model.ctc_loss_reduction,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
            "vocab_size": self.processor.tokenizer.vocab_size,
            "trust_remote_code": True,
            "torch_dtype": torch.float32
        }
        
        if self.quantization_config:
            logger.info("Attempting to load model with quantization...")
            model_kwargs["quantization_config"] = self.quantization_config
            model_kwargs["device_map"] = "auto"
        else:
            logger.info("Loading model without quantization...")
        
        try:
            self.model = Wav2Vec2BertForCTC.from_pretrained(
                self.configs.model.name,
                **model_kwargs
            )
            if self.quantization_config:
                logger.info("Model loaded successfully with quantization.")
            else:
                logger.info("Model loaded successfully without quantization.")
        except Exception as e:
            logger.error(f"Model loading failed: {e}", exc_info=True)
            if self.quantization_config:
                logger.warning("Falling back to standard (un-quantized) model loading.")
                self.quantization_config = None
                model_kwargs.pop("quantization_config", None)
                model_kwargs.pop("device_map", None)
                self.model = Wav2Vec2BertForCTC.from_pretrained(
                    self.configs.model.name, **model_kwargs
                )
        
        if not self.quantization_config and torch.cuda.is_available():
            self.model = self.model.to("cuda")
            logger.info("Model moved to GPU.")
        
        return self.model

    def add_peft_adapters(self) -> Wav2Vec2BertForCTC:
        """Prepare model for training and add PEFT LoRA adapters."""
        # FIX: Manually enable gradient checkpointing instead of using prepare_model_for_kbit_training,
        # which is incompatible with Wav2Vec2-BERT models as it expects input_embeddings.
        if self.quantization_config:
            logger.info("Enabling gradient checkpointing for quantized model...")
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
            else:
                logger.warning("Model does not have 'gradient_checkpointing_enable' method. Skipping.")
        
        try:
            lora_configs = self.configs.lora
            lora_config = LoraConfig(
                r=lora_configs.rank,
                lora_alpha=lora_configs.alpha,
                target_modules=lora_configs.target_modules,
                lora_dropout=lora_configs.dropout,
                bias="none",
                modules_to_save=["lm_head"],
            )
            
            logger.info("Adding PEFT LoRA adapters...")
            self.model = get_peft_model(self.model, lora_config)
            
            logger.info("Successfully added PEFT adapters.")
            self.model.print_trainable_parameters()
        
        except Exception as e:
            logger.error("Failed to add PEFT adapters.", exc_info=True)
            raise RuntimeError(
                f"Could not apply PEFT adapters to the model. The trainer will fail. "
                f"Please check your PEFT/LoRA configuration. Original error: {e}"
            ) from e

        return self.model

    def load_pipeline(
        self, 
        vocab_path: str,
        add_adapters: bool = True
    ) -> Tuple[Wav2Vec2BertForCTC, Wav2Vec2BertProcessor]:
        """
        Execute the complete model loading pipeline.
        
        Args:
            vocab_path: Path to the vocabulary JSON file.
            add_adapters: Whether to add PEFT adapters for training.
            
        Returns:
            A tuple containing the configured model and processor.
        """
        logger.info("--- Starting Armenian ASR Model Loading Pipeline ---")
        self.load_feature_extractor()
        self.create_tokenizer(vocab_path=vocab_path)
        self.build_processor()
        self.build_model()
        
        if add_adapters:
            self.add_peft_adapters()
        
        self.get_memory_usage()
        logger.info("--- Model Loading Pipeline Completed Successfully ---")
        return self.model, self.processor

    def get_memory_usage(self) -> None:
        """Log the current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU Memory Footprint - Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

    def __call__(
        self, 
        vocab_path: str='vocab.json',
        add_adapters: bool = True
    ) -> Tuple[Wav2Vec2BertForCTC, Wav2Vec2BertProcessor]:
        return self.load_pipeline(vocab_path=vocab_path, add_adapters=add_adapters)
