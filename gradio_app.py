import os
import torch
from PIL import Image
import numpy as np
import gradio as gr

# Import from your MaskCLIP repo
from modules.modeling import SegCLIP
from modules.util_module import show_log
from modules.modeling import load_model  # or wherever your inference loader is
# (you’ll need to adapt import paths if your project structure is different)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# You can specify where your checkpoint is (you may have a default or config)
DEFAULT_CKPT = "checkpoints/segclip.bin"

def inference_fn(image: Image.Image, prompt: str):
    """
    Runs MaskCLIP inference on the input image + prompt.
    Returns: segmentation mask, or overlay.
    """
    # Preprocess input image (resize, normalize, to tensor) — adapt to your project’s preprocessing
    img_np = np.array(image).astype(np.uint8)
    # convert to torch tensor, add dims, move to device
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

    # Load model (only once ideally)
    model, test_pipeline, dev = load_model(
        model_path=DEFAULT_CKPT,
        device=device,
        dataset=None,   # you may need a dataset name or config
        vis_mode="input" # or whatever your code expects
    )

    # Perform segmentation
    with torch.no_grad():
        outputs = model(img_tensor, prompt)  # this depends on your model API
        # Suppose `outputs` is a mask tensor (1 x H x W) or soft logits
        mask = outputs.squeeze().cpu().numpy()
        # Normalize or threshold, convert to color mask
        mask_img = (mask > 0.5).astype(np.uint8) * 255  # binary mask
        mask_pil = Image.fromarray(mask_img)

        # Optionally overlay mask on original
        overlay = image.copy()
        overlay_np = np.array(overlay)
        overlay_np[..., 0] = np.where(mask > 0.5, 255, overlay_np[..., 0])  # red channel overlay
        overlay_pil = Image.fromarray(overlay_np)

    return overlay_pil, mask_pil


def build_interface():
    """Builds and returns a Gradio interface."""
    title = "MaskCLIP Segmenter"
    description = "Upload an image and supply a text prompt; outputs masked overlay + mask"

    iface = gr.Interface(
        fn=inference_fn,
        inputs=[
            gr.Image(type="pil"),
            gr.Textbox(lines=1, placeholder="Enter segmentation prompt, e.g. 'cat' or 'car'"),
        ],
        outputs=[
            gr.Image(type="pil", label="Overlay"),
            gr.Image(type="pil", label="Binary Mask"),
        ],
        title=title,
        description=description,
        allow_flagging="never",
        examples=None
    )
    return iface


if __name__ == "__main__":
    iface = build_interface()
    iface.launch(server_name="0.0.0.0", server_port=7860, share=True)
