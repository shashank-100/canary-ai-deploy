"""
Model Manager for AI Model Serving Platform
Handles model loading, caching, and inference
"""
import asyncio
import time
import hashlib
import pickle
from typing import Dict, Any, List, Optional, Union
from abc import ABC, abstractmethod
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
from torchvision import models, transforms
from PIL import Image
import io
import base64

from ..common.config import config
from ..common.logging_config import get_model_logger


class BaseModel(ABC):
    """Abstract base class for all models"""
    
    def __init__(self, model_name: str, model_version: str):
        self.model_name = model_name
        self.model_version = model_version
        self.logger = get_model_logger(model_name, model_version)
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_loaded = False
    
    @abstractmethod
    async def load_model(self) -> None:
        """Load the model"""
        pass
    
    @abstractmethod
    async def predict(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Make prediction on single input"""
        pass
    
    @abstractmethod
    async def batch_predict(self, batch_data: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Make predictions on batch input"""
        pass
    
    async def unload_model(self) -> None:
        """Unload the model from memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.is_loaded = False
        self.logger.info("Model unloaded from memory")


class BertNerModel(BaseModel):
    """BERT Named Entity Recognition Model"""
    
    def __init__(self, model_name: str = "bert-ner", model_version: str = "v1.0.0"):
        super().__init__(model_name, model_version)
        self.model_path = "dbmdz/bert-large-cased-finetuned-conll03-english"
        self.max_length = 512
    
    async def load_model(self) -> None:
        """Load BERT NER model"""
        try:
            self.logger.info(f"Loading BERT NER model from {self.model_path}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            self.is_loaded = True
            self.logger.info(f"BERT NER model loaded successfully on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load BERT NER model: {e}", exc_info=True)
            raise
    
    async def predict(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Make NER prediction on single text input"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        text = data.get("text", "")
        if not text:
            raise ValueError("Text input is required")
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_token_class_ids = predictions.argmax(dim=-1)
            
            # Convert predictions to entities
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            entities = []
            
            for i, (token, class_id) in enumerate(zip(tokens, predicted_token_class_ids[0])):
                if token in ["[CLS]", "[SEP]", "[PAD]"]:
                    continue
                
                label = self.model.config.id2label[class_id.item()]
                confidence = predictions[0][i][class_id].item()
                
                if label != "O":  # Not "Outside" label
                    entities.append({
                        "token": token,
                        "label": label,
                        "confidence": confidence,
                        "position": i
                    })
            
            return entities
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}", exc_info=True)
            raise
    
    async def batch_predict(self, batch_data: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Make NER predictions on batch text inputs"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        texts = [item.get("text", "") for item in batch_data]
        if not all(texts):
            raise ValueError("All items must have text input")
        
        try:
            # Tokenize batch
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Make batch prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_token_class_ids = predictions.argmax(dim=-1)
            
            # Convert predictions to entities for each text
            batch_entities = []
            for batch_idx in range(len(texts)):
                tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][batch_idx])
                entities = []
                
                for i, (token, class_id) in enumerate(zip(tokens, predicted_token_class_ids[batch_idx])):
                    if token in ["[CLS]", "[SEP]", "[PAD]"]:
                        continue
                    
                    label = self.model.config.id2label[class_id.item()]
                    confidence = predictions[batch_idx][i][class_id].item()
                    
                    if label != "O":  # Not "Outside" label
                        entities.append({
                            "token": token,
                            "label": label,
                            "confidence": confidence,
                            "position": i
                        })
                
                batch_entities.append(entities)
            
            return batch_entities
            
        except Exception as e:
            self.logger.error(f"Batch prediction failed: {e}", exc_info=True)
            raise


class ResNetClassifierModel(BaseModel):
    """ResNet Image Classifier Model"""
    
    def __init__(self, model_name: str = "resnet-classifier", model_version: str = "v1.0.0"):
        super().__init__(model_name, model_version)
        self.num_classes = 1000  # ImageNet classes
        self.image_size = 224
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load ImageNet class labels
        self.class_labels = self._load_imagenet_labels()
    
    def _load_imagenet_labels(self) -> List[str]:
        """Load ImageNet class labels"""
        # This is a simplified version - in production, load from a file
        return [f"class_{i}" for i in range(self.num_classes)]
    
    async def load_model(self) -> None:
        """Load ResNet classifier model"""
        try:
            self.logger.info("Loading ResNet classifier model")
            
            # Load pre-trained ResNet model
            self.model = models.resnet50(pretrained=True)
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            self.is_loaded = True
            self.logger.info(f"ResNet classifier model loaded successfully on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load ResNet classifier model: {e}", exc_info=True)
            raise
    
    def _decode_image(self, image_data: str) -> Image.Image:
        """Decode base64 image data"""
        try:
            # Remove data URL prefix if present
            if image_data.startswith("data:image"):
                image_data = image_data.split(",")[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            return image
            
        except Exception as e:
            raise ValueError(f"Failed to decode image: {e}")
    
    async def predict(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Make classification prediction on single image input"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        image_data = data.get("image", "")
        if not image_data:
            raise ValueError("Image input is required")
        
        try:
            # Decode and preprocess image
            image = self._decode_image(image_data)
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                
                # Get top 5 predictions
                top5_prob, top5_indices = torch.topk(probabilities, 5)
            
            # Format predictions
            predictions = []
            for i in range(5):
                predictions.append({
                    "class": self.class_labels[top5_indices[i].item()],
                    "confidence": top5_prob[i].item(),
                    "class_id": top5_indices[i].item()
                })
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}", exc_info=True)
            raise
    
    async def batch_predict(self, batch_data: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Make classification predictions on batch image inputs"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        images_data = [item.get("image", "") for item in batch_data]
        if not all(images_data):
            raise ValueError("All items must have image input")
        
        try:
            # Decode and preprocess images
            batch_tensors = []
            for image_data in images_data:
                image = self._decode_image(image_data)
                input_tensor = self.transform(image)
                batch_tensors.append(input_tensor)
            
            # Stack tensors and move to device
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Make batch prediction
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # Get top 5 predictions for each image
                top5_prob, top5_indices = torch.topk(probabilities, 5, dim=1)
            
            # Format predictions for each image
            batch_predictions = []
            for batch_idx in range(len(images_data)):
                predictions = []
                for i in range(5):
                    predictions.append({
                        "class": self.class_labels[top5_indices[batch_idx][i].item()],
                        "confidence": top5_prob[batch_idx][i].item(),
                        "class_id": top5_indices[batch_idx][i].item()
                    })
                batch_predictions.append(predictions)
            
            return batch_predictions
            
        except Exception as e:
            self.logger.error(f"Batch prediction failed: {e}", exc_info=True)
            raise


class ModelCache:
    """Simple in-memory cache for model predictions"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = {}
        self.access_times = {}
    
    def _generate_key(self, data: Dict[str, Any]) -> str:
        """Generate cache key from input data"""
        # Create a hash of the input data
        data_str = str(sorted(data.items()))
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def get(self, data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Get cached prediction"""
        key = self._generate_key(data)
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def set(self, data: Dict[str, Any], prediction: List[Dict[str, Any]]) -> None:
        """Cache prediction"""
        key = self._generate_key(data)
        
        # Remove oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = prediction
        self.access_times[key] = time.time()


class ModelManager:
    """Manages model loading, caching, and inference"""
    
    def __init__(self, model_name: str, model_version: str):
        self.model_name = model_name
        self.model_version = model_version
        self.logger = get_model_logger(model_name, model_version)
        
        # Initialize model based on name
        if model_name == "bert-ner":
            self.model = BertNerModel(model_name, model_version)
        elif model_name == "resnet-classifier":
            self.model = ResNetClassifierModel(model_name, model_version)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Initialize cache
        self.cache = ModelCache(config.model.cache_size)
        self.cache_enabled = config.model.cache_size > 0
        
        self._ready = False
    
    async def initialize(self) -> None:
        """Initialize the model manager"""
        try:
            self.logger.info("Initializing model manager")
            await self.model.load_model()
            self._ready = True
            self.logger.info("Model manager initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize model manager: {e}", exc_info=True)
            raise
    
    def is_ready(self) -> bool:
        """Check if model manager is ready"""
        return self._ready and self.model.is_loaded
    
    async def predict(self, data: Dict[str, Any], model_version: Optional[str] = None) -> Dict[str, Any]:
        """Make single prediction"""
        if not self.is_ready():
            raise RuntimeError("Model manager not ready")
        
        # Check cache first
        if self.cache_enabled:
            cached_result = self.cache.get(data)
            if cached_result is not None:
                self.logger.debug("Cache hit for prediction")
                return {
                    "predictions": cached_result,
                    "model_version": self.model_version,
                    "cached": True
                }
        
        # Make prediction
        predictions = await self.model.predict(data)
        
        # Cache result
        if self.cache_enabled:
            self.cache.set(data, predictions)
            self.logger.debug("Cached prediction result")
        
        return {
            "predictions": predictions,
            "model_version": self.model_version,
            "cached": False
        }
    
    async def batch_predict(self, batch_data: List[Dict[str, Any]], 
                          batch_size: Optional[int] = None,
                          model_version: Optional[str] = None) -> Dict[str, Any]:
        """Make batch predictions"""
        if not self.is_ready():
            raise RuntimeError("Model manager not ready")
        
        batch_size = batch_size or config.model.batch_size
        
        # Process in chunks if batch is too large
        if len(batch_data) > batch_size:
            all_predictions = []
            for i in range(0, len(batch_data), batch_size):
                chunk = batch_data[i:i + batch_size]
                chunk_predictions = await self.model.batch_predict(chunk)
                all_predictions.extend(chunk_predictions)
            predictions = all_predictions
        else:
            predictions = await self.model.batch_predict(batch_data)
        
        return {
            "predictions": predictions,
            "model_version": self.model_version,
            "cached": False
        }
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "device": str(self.model.device),
            "is_loaded": self.model.is_loaded,
            "is_ready": self.is_ready(),
            "cache_enabled": self.cache_enabled,
            "cache_size": len(self.cache.cache) if self.cache_enabled else 0
        }
    
    async def cleanup(self) -> None:
        """Cleanup model manager"""
        try:
            self.logger.info("Cleaning up model manager")
            await self.model.unload_model()
            self._ready = False
            self.logger.info("Model manager cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}", exc_info=True)

