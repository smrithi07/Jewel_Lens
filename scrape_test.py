# Throwaway script — just proves OWL-ViT and DINOv2 load and run.
# Delete this after Phase 0 checkpoint passes.

from PIL import Image
import requests
import torch

# --- Test DINOv2 embedding ---
from transformers import AutoImageProcessor, AutoModel

print("Loading DINOv2...")
dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
dino_model = AutoModel.from_pretrained("facebook/dinov2-base")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = dino_processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = dino_model(**inputs)
embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
print(f"DINOv2 embedding shape: {embedding.shape}")

# --- Test OWL-ViT detection ---
from transformers import OwlViTProcessor, OwlViTForObjectDetection

print("Loading OWL-ViT...")
owl_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
owl_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

texts = [["a photo of a cat", "a photo of a remote control"]]
inputs = owl_processor(text=texts, images=image, return_tensors="pt")
with torch.no_grad():
    outputs = owl_model(**inputs)

target_sizes = torch.tensor([image.size[::-1]])
results = owl_processor.post_process_grounded_object_detection(outputs, threshold=0.1, target_sizes=target_sizes)
print(f"OWL-ViT detections found: {len(results[0]['boxes'])}")

print("\nPhase 0 checkpoint: PASS")