"""
Simple Gemma 4 + FlashMLX Integration Test

Focus: Verify core integration works.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from flashmlx.vlm_bridge import load_vlm_model, create_vlm_cache, generate_vlm


print("=" * 60)
print("Gemma 4 + FlashMLX Simple Integration Test")
print("=" * 60)

# Step 1: Load model
print("\n[1/3] Loading Gemma 4...")
model, processor = load_vlm_model("/Volumes/toshiba/models/gemma-4-E4B")

# Step 2: Create FlashMLX cache (standard - no compression)
print("\n[2/3] Creating FlashMLX cache...")
cache = create_vlm_cache(model, strategy="standard")
print(f"  Cache type: {type(cache[0]).__name__}")
print(f"  Cache layers: {len(cache)}")

# Step 3: Generate with proper settings
print("\n[3/3] Generating text...")
response = generate_vlm(
    model, processor,
    prompt="<start_of_turn>user\nWhat is 2+2?<end_of_turn>\n<start_of_turn>model\n",
    cache=cache,
    max_tokens=10,
    temperature=0.7,  # Non-zero to avoid repetition
    verbose=False
)

print("\n" + "=" * 60)
print("Results")
print("=" * 60)
print(f"Response text: {response.text}")
print(f"Tokens: {response.generation_tokens}")
print(f"Speed: {response.generation_tps:.1f} tok/s")
print(f"Memory: {response.peak_memory:.2f} MB")

print("\n✅ Integration test passed!")
