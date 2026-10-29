"""Download DINOv2 and DINOv3 while the Space is built, so a restart does not wait for them.
DINOv3 is gated on Hugging Face: it needs the HF_TOKEN secret of an account that accepted its licence."""
import os

from transformers import AutoModel

AutoModel.from_pretrained("facebook/dinov2-base")
print("DINOv2 cached")
if os.environ.get("HF_TOKEN"):
    try:
        AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
        print("DINOv3 cached")
    except Exception as e:
        print("DINOv3 not cached:", type(e).__name__, e)
else:
    print("No HF_TOKEN secret: DINOv3 is not cached, and the obstruction map stays off")
