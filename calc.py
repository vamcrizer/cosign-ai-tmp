"""
Profile VRAM usage during inference
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import gc
import argparse
from models import Uni_Sign
import utils


def get_gpu_memory():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        return allocated, reserved
    return 0, 0


def profile_memory(args):
    """Profile memory usage step by step"""
    
    print("=" * 80)
    print("💾 GPU MEMORY PROFILING")
    print("=" * 80)
    
    # Baseline
    torch.cuda.empty_cache()
    gc.collect()
    alloc, reserved = get_gpu_memory()
    print(f"\n📊 Baseline:")
    print(f"   Allocated: {alloc:.2f} MB")
    print(f"   Reserved:  {reserved:.2f} MB")
    
    # Create model
    print(f"\n🤖 Creating model...")
    model = Uni_Sign(args=args)
    alloc, reserved = get_gpu_memory()
    print(f"   Allocated: {alloc:.2f} MB (CPU only, no GPU yet)")
    print(f"   Reserved:  {reserved:.2f} MB")
    
    # Move to CUDA
    print(f"\n🚀 Moving to CUDA...")
    model.cuda()
    alloc, reserved = get_gpu_memory()
    print(f"   Allocated: {alloc:.2f} MB")
    print(f"   Reserved:  {reserved:.2f} MB")
    model_memory = alloc
    
    # Convert to BF16
    print(f"\n🔄 Converting to BF16...")
    model.to(torch.bfloat16)
    alloc, reserved = get_gpu_memory()
    print(f"   Allocated: {alloc:.2f} MB")
    print(f"   Reserved:  {reserved:.2f} MB")
    bf16_memory = alloc
    
    # Set eval mode
    model.eval()
    
    # ✅ KIỂM TRA: MT5 model có được load không?
    print(f"\n🔍 Checking MT5 model...")
    if hasattr(model, 'mt5_model') and model.mt5_model is not None:
        mt5_params = sum(p.numel() for p in model.mt5_model.parameters())
        mt5_memory_estimated = (mt5_params * 2) / 1024**2  # BF16 = 2 bytes
        print(f"   MT5 parameters: {mt5_params:,}")
        print(f"   MT5 estimated memory: {mt5_memory_estimated:.2f} MB")
    
    # ✅ KIỂM TRA: RGB backbone
    if hasattr(model, 'rgb_support_backbone') and model.rgb_support_backbone is not None:
        rgb_params = sum(p.numel() for p in model.rgb_support_backbone.parameters())
        rgb_memory_estimated = (rgb_params * 2) / 1024**2
        print(f"\n🖼️ RGB Support Backbone:")
        print(f"   Parameters: {rgb_params:,}")
        print(f"   Estimated memory: {rgb_memory_estimated:.2f} MB")
    
    # Simulate inference with dummy data
    print(f"\n🔮 Simulating inference with dummy data...")
    
    batch_size = 1
    seq_length = 200  # Typical video length
    
    # Create dummy pose data
    dummy_pose = torch.randn(batch_size, seq_length, 543, 3).to(torch.bfloat16).cuda()
    
    alloc_before = torch.cuda.memory_allocated() / 1024**2
    
    with torch.no_grad():
        # Dummy input (simplified)
        src_input = {
            'pose_data': dummy_pose,
        }
        
        if args.rgb_support:
            # RGB data: (batch, frames, 3, H, W)
            dummy_rgb = torch.randn(batch_size, seq_length, 3, 224, 224).to(torch.bfloat16).cuda()
            src_input['rgb_data'] = dummy_rgb
            
            alloc_rgb = torch.cuda.memory_allocated() / 1024**2
            print(f"   After loading RGB data: {alloc_rgb:.2f} MB (+{alloc_rgb - alloc_before:.2f} MB)")
    
    alloc, reserved = get_gpu_memory()
    print(f"   After loading data: {alloc:.2f} MB")
    print(f"   Reserved:           {reserved:.2f} MB")
    
    inference_overhead = alloc - bf16_memory
    print(f"   Data loading overhead: {inference_overhead:.2f} MB")
    
    # Summary
    print(f"\n" + "=" * 80)
    print(f"📊 MEMORY BREAKDOWN (Before actual inference):")
    print(f"=" * 80)
    print(f"   Model weights (BF16):     {bf16_memory:>10.2f} MB ({bf16_memory/1024:.2f} GB)")
    print(f"   Input data overhead:      {inference_overhead:>10.2f} MB ({inference_overhead/1024:.2f} GB)")
    print(f"   Total allocated:          {alloc:>10.2f} MB ({alloc/1024:.2f} GB)")
    print(f"   Total reserved (CUDA):    {reserved:>10.2f} MB ({reserved/1024:.2f} GB)")
    print(f"=" * 80)
    
    # Explain the overhead
    print(f"\n💡 WHY DOES INFERENCE USE ~11GB VRAM?")
    print(f"=" * 80)
    print(f"1. 🧠 MT5 Model (Largest component):")
    print(f"   - Model weights: ~{bf16_memory/1024:.2f} GB")
    print(f"   - MT5 has ~580M parameters")
    
    print(f"\n2. 🔑 KV Cache during generation (MAIN CULPRIT):")
    print(f"   - MT5 stores key-value cache for each attention layer")
    print(f"   - With num_beams=4, memory is multiplied by 4x")
    print(f"   - For max_new_tokens=100, KV cache can be 3-5 GB")
    print(f"   - Formula: layers × heads × seq_len × hidden_dim × beams × dtype")
    
    print(f"\n3. 📹 Video data in VRAM:")
    if args.rgb_support:
        rgb_size = (batch_size * seq_length * 3 * 224 * 224 * 2) / 1024**2
        print(f"   - RGB frames (200 frames): {rgb_size:.2f} MB ({rgb_size/1024:.2f} GB)")
    pose_size = (batch_size * seq_length * 543 * 3 * 2) / 1024**2
    print(f"   - Pose keypoints: {pose_size:.2f} MB ({pose_size/1024:.2f} GB)")
    
    print(f"\n4. 🔄 Intermediate activations:")
    print(f"   - STGCN forward pass")
    print(f"   - Deformable attention maps")
    print(f"   - MT5 encoder hidden states")
    print(f"   - MT5 decoder states during generation")
    print(f"   - Estimated: 1-2 GB")
    
    print(f"\n5. 🎯 CUDA memory allocator overhead:")
    print(f"   - PyTorch pre-allocates memory for efficiency")
    print(f"   - Reserved - Allocated = {(reserved - alloc)/1024:.2f} GB overhead")
    
    print(f"=" * 80)
    
    # Detailed breakdown
    print(f"\n📊 ESTIMATED BREAKDOWN DURING INFERENCE:")
    print(f"=" * 80)
    model_gb = bf16_memory / 1024
    kv_cache_gb = 4.0  # Estimated with beam=4
    data_gb = inference_overhead / 1024
    activation_gb = 1.5  # Estimated
    overhead_gb = 1.0  # CUDA allocator
    
    total_estimated = model_gb + kv_cache_gb + data_gb + activation_gb + overhead_gb
    
    print(f"   Model weights:            {model_gb:>6.2f} GB ({model_gb/total_estimated*100:>5.1f}%)")
    print(f"   KV Cache (beam=4):        {kv_cache_gb:>6.2f} GB ({kv_cache_gb/total_estimated*100:>5.1f}%)")
    print(f"   Input data:               {data_gb:>6.2f} GB ({data_gb/total_estimated*100:>5.1f}%)")
    print(f"   Activations:              {activation_gb:>6.2f} GB ({activation_gb/total_estimated*100:>5.1f}%)")
    print(f"   CUDA overhead:            {overhead_gb:>6.2f} GB ({overhead_gb/total_estimated*100:>5.1f}%)")
    print(f"   " + "-" * 50)
    print(f"   TOTAL ESTIMATED:          {total_estimated:>6.2f} GB")
    print(f"=" * 80)
    
    # Recommendations
    print(f"\n💡 HOW TO REDUCE VRAM USAGE:")
    print(f"=" * 80)
    print(f"1. ✂️ Reduce beam search:")
    print(f"   model.generate(..., num_beams=1)  # Greedy decoding")
    print(f"   Expected savings: ~3-4 GB (from ~11GB → ~7GB)")
    
    print(f"\n2. 📏 Reduce max sequence length:")
    print(f"   model.generate(..., max_new_tokens=50)  # Instead of 100")
    print(f"   Expected savings: ~0.5-1 GB")
    
    print(f"\n3. 🎬 Process shorter videos:")
    print(f"   Trim videos to <100 frames if possible")
    print(f"   Expected savings: ~0.2-0.5 GB")
    
    print(f"\n4. 🔧 Use gradient checkpointing (if training):")
    print(f"   model.mt5_model.gradient_checkpointing_enable()")
    print(f"   Not applicable for inference")
    
    print(f"\n5. 💾 Use CPU offloading (slower but less VRAM):")
    print(f"   from accelerate import cpu_offload")
    print(f"   cpu_offload(model, execution_device='cuda:0')")
    print(f"   Expected savings: ~2-3 GB (but 2-3x slower)")
    
    print(f"=" * 80)
    
    print(f"\n🎯 RECOMMENDED CONFIGURATION FOR 12GB GPU:")
    print(f"   - num_beams=1 (greedy decoding)")
    print(f"   - max_new_tokens=100")
    print(f"   - Process videos <150 frames")
    print(f"   - Expected usage: ~6-7 GB")
    print(f"=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser('Memory Profiler for Uni-Sign')
    parser.add_argument('--rgb_support', action='store_true', help='Include RGB support')
    parser.add_argument('--dataset', default='CSL_Daily')
    parser.add_argument('--task', default='SLT')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--max_length', type=int, default=256)
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available! This profiler requires GPU.")
        return
    
    print(f"\n🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")
    
    profile_memory(args)


if __name__ == '__main__':
    main()