"""
FastAPI Server for Uni-Sign Inference
Thread-safe implementation for concurrent users
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import tempfile
import shutil
from contextlib import asynccontextmanager
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import asyncio
from threading import Lock

from models import Uni_Sign
import utils as utils
from datasets import S2T_Dataset_online
from config import *
from rtmlib import Wholebody
from torch.nn.utils.rnn import pad_sequence

# ✅ THAY ĐỔI 1: Thread-safe model wrapper
class ModelManager:
    """Thread-safe model manager"""
    def __init__(self):
        self.model = None
        self.wholebody = None
        self.args = None
        self.lock = Lock()  # Mutex for thread safety
        self.device = None
        
    def load(self):
        """Load model once at startup"""
        print("=" * 60)
        print(" Initializing Uni-Sign Inference Server...")
        print("=" * 60)
        
        # Set up args
        import argparse
        parser = argparse.ArgumentParser('Uni-Sign API Server', parents=[utils.get_args_parser()])
        self.args = parser.parse_args([])
        self.args.rgb_support = True  # Always use RGB support
        
        # Override checkpoint path
        checkpoint_path = os.getenv("UNISIGN_CHECKPOINT")
        
        # Make path absolute if relative
        if checkpoint_path and not os.path.isabs(checkpoint_path):
            # Get project root (parent of demo folder)
            project_root = Path(__file__).parent.parent
            checkpoint_path = str(project_root / checkpoint_path)
        
        if checkpoint_path:
            self.args.finetune = checkpoint_path
        else:
            # Fallback to default path
            self.args.finetune = str(Path(__file__).parent.parent / "pretrained_weight" / "best_checkpoint.pth")
        
        if not os.path.exists(self.args.finetune):
            raise FileNotFoundError(f"Checkpoint not found: {self.args.finetune}")
        
        print(f"📋 Config: dataset={self.args.dataset}, task={self.args.task}, rgb_support={self.args.rgb_support}")
        
        # Set seed
        utils.set_seed(self.args.seed)
        
        # Initialize pose detector
        print("\n📊 Initializing pose detector...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.device == "cuda":
            try:
                import onnxruntime as ort
                if 'CUDAExecutionProvider' not in ort.get_available_providers():
                    print("⚠️ CUDA not available for ONNX Runtime, using CPU")
                    self.device = "cpu"
                else:
                    print(" Using CUDA for pose extraction")
            except ImportError:
                print(" onnxruntime not found, using CPU")
                self.device = "cpu"
        
        self.wholebody = Wholebody(
            to_openpose=False,
            mode="lightweight",
            backend="onnxruntime",
            device=self.device
        )
        
        # Create model
        print("\nCreating model...")
        self.model = Uni_Sign(args=self.args)
        
        if torch.cuda.is_available():
            self.model.cuda()
            print("Model loaded on CUDA")
        else:
            print("Model loaded on CPU")
        
        self.model.eval()
        
        # Load checkpoint
        print(f"\nLoading checkpoint: {self.args.finetune}")
        checkpoint = torch.load(self.args.finetune, map_location='cpu', weights_only=True)
        
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        ret = self.model.load_state_dict(state_dict, strict=True)
        
        if not ret.missing_keys and not ret.unexpected_keys:
            print("Checkpoint loaded perfectly!")
        
        # Set dtype
        if torch.cuda.is_available():
            self.model.to(torch.bfloat16)
        else:
            self.model.to(torch.float32)
        
        print("\n" + "=" * 60)
        print("Server ready for inference!")
        print("=" * 60 + "\n")
    
    def process_frame(self, frame):
        """Thread-safe frame processing"""
        frame = np.uint8(frame)
        keypoints, scores = self.wholebody(frame)
        H, W, C = frame.shape
        return keypoints, scores, [W, H]
    
    def extract_pose(self, video_path: str, max_workers: int = 8):
        """Extract pose from video"""
        data = {"keypoints": [], "scores": []}
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        vid_data = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            vid_data.append(frame)
        cap.release()
        
        if len(vid_data) == 0:
            raise ValueError("No frames extracted from video")
        
        # Process frames
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_frame, frame) for frame in vid_data]
            for f in tqdm(futures, desc="Extracting pose", total=len(vid_data)):
                results.append(f.result())
        
        for keypoints, scores, w_h in results:
            data['keypoints'].append(keypoints / np.array(w_h)[None, None])
            data['scores'].append(scores)
        
        return data
    
    def inference(self, video_path: str):
        """
        ✅ THAY ĐỔI 3: Thread-safe inference with lock
        Only one inference at a time to avoid GPU memory conflicts
        """
        with self.lock:  # ← Mutex: Chỉ 1 request được chạy tại 1 thời điểm
            print(f"\n{'='*60}")
            print(f" Processing video: {Path(video_path).name}")
            print(f"{'='*60}")
            
            # Extract pose
            pose_data = self.extract_pose(video_path)
            
            # Create dataset (local instance, không dùng global)
            print("Creating dataset...")
            online_data = S2T_Dataset_online(args=self.args)
            online_data.rgb_data = video_path
            online_data.pose_data = pose_data
            
            online_sampler = torch.utils.data.SequentialSampler(online_data)
            online_dataloader = torch.utils.data.DataLoader(
                online_data,
                batch_size=1,
                collate_fn=online_data.collate_fn,
                sampler=online_sampler,
            )
            
            # Run inference
            print("Running inference...")
            target_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            
            with torch.no_grad():
                tgt_pres = []
                
                for step, (src_input, tgt_input) in enumerate(online_dataloader):
                    if target_dtype is not None:
                        for key in src_input.keys():
                            if isinstance(src_input[key], torch.Tensor):
                                src_input[key] = src_input[key].to(target_dtype)
                                if torch.cuda.is_available():
                                    src_input[key] = src_input[key].cuda()
                    
                    stack_out = self.model(src_input, tgt_input)
                    
                    output = self.model.generate(
                        stack_out,
                        max_new_tokens=100,
                        num_beams=4,
                    )
                    
                    for i in range(len(output)):
                        tgt_pres.append(output[i])
            
            # Decode
            tokenizer = self.model.mt5_tokenizer
            padding_value = tokenizer.eos_token_id
            
            if torch.cuda.is_available():
                pad_tensor = torch.ones(150 - len(tgt_pres[0])).cuda() * padding_value
            else:
                pad_tensor = torch.ones(150 - len(tgt_pres[0])) * padding_value
            
            tgt_pres[0] = torch.cat((tgt_pres[0], pad_tensor.long()), dim=0)
            
            tgt_pres = pad_sequence(tgt_pres, batch_first=True, padding_value=padding_value)
            tgt_pres = tokenizer.batch_decode(tgt_pres, skip_special_tokens=True)
            
            prediction = tgt_pres[0]
            print(f"Prediction: {prediction}")
            print(f"{'='*60}\n")
            
            return prediction


# THAY ĐỔI 4: Singleton instance
model_manager = ModelManager()


class PredictionResponse(BaseModel):
    success: bool
    prediction: Optional[str] = None
    error: Optional[str] = None
    frames_processed: Optional[int] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan: load model on startup"""
    model_manager.load()
    yield
    print("Shutting down server...")


app = FastAPI(
    title="Uni-Sign Inference API",
    description="Sign Language Translation API (Multi-user support)",
    version="2.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    return {
        "status": "running",
        "model_loaded": model_manager.model is not None,
        "cuda_available": torch.cuda.is_available(),
        "message": "Uni-Sign Inference API (Thread-safe)"
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model_manager.model is not None,
        "pose_detector_loaded": model_manager.wholebody is not None,
        "cuda_available": torch.cuda.is_available(),
        "device": model_manager.device,
        "concurrent_support": True,  # ← Chỉ 1 request tại 1 thời điểm (thread-safe)
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Thread-safe prediction endpoint
    Handles concurrent requests with mutex lock
    """
    if model_manager.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    ext = Path(file.filename).suffix.lower()
    if ext not in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}"
        )
    
    temp_dir = tempfile.mkdtemp()
    temp_video_path = os.path.join(temp_dir, f"upload{ext}")
    
    try:
        # Save uploaded file
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Thread-safe inference
        prediction = model_manager.inference(temp_video_path)
        
        # Count frames
        cap = cv2.VideoCapture(temp_video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        return PredictionResponse(
            success=True,
            prediction=prediction,
            frames_processed=frame_count
        )
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return PredictionResponse(success=False, error=str(e))
    
    finally:
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Cleanup error: {e}")


@app.post("/predict_url")
async def predict_from_url(video_url: str):
    """Predict from URL (thread-safe)"""
    if model_manager.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    import requests
    
    temp_dir = tempfile.mkdtemp()
    temp_video_path = os.path.join(temp_dir, "download.mp4")
    
    try:
        print(f"Downloading: {video_url}")
        response = requests.get(video_url, stream=True)
        response.raise_for_status()
        
        with open(temp_video_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        prediction = model_manager.inference(temp_video_path)
        
        cap = cv2.VideoCapture(temp_video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        return PredictionResponse(
            success=True,
            prediction=prediction,
            frames_processed=frame_count
        )
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return PredictionResponse(success=False, error=str(e))
    
    finally:
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


if __name__ == "__main__":
    import uvicorn
    
    # Get server configuration from environment variables
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    log_level = os.getenv("LOG_LEVEL", "info")
    
    print(f"\nStarting Uni-Sign API Server...")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Log Level: {log_level}")
    print(f"   Checkpoint: {os.getenv('UNISIGN_CHECKPOINT', 'default')}\n")
    
    uvicorn.run(app, host=host, port=port, log_level=log_level)