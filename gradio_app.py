import os
import sys
import tempfile
from typing import List, Tuple

import numpy as np
import torch
import mmcv

# Ensure local mmseg package is importable when running from repo root
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mmseg.apis import init_segmentor, inference_segmentor  # noqa: E402

# Reuse prompt engineering utilities (uses OpenAI CLIP under the hood)
from tools.maskclip_utils.prompt_engineering import (  # noqa: E402
    prompt_templates as DEFAULT_TEMPLATES,
    bg_classes as DEFAULT_BG_CLASSES,
)

import gradio as gr  # noqa: E402


# ------- Helpers -------

def parse_classes(text: str) -> List[str]:
    if not text:
        return []
    # split on comma, semicolon, or newline
    raw = [t.strip() for t in text.replace("\n", ",").replace(";", ",").split(",")]
    # de-duplicate while preserving order
    seen = set()
    result = []
    for t in raw:
        if not t:
            continue
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


def resolve_paths(config_rel: str) -> str:
    return os.path.join(REPO_ROOT, config_rel)


def build_config_for_classes(
    config_file: str,
    clip_weights_path: str,
    fg_classes: List[str],
    bg_classes: List[str],
):
    import mmcv as _mmcv

    cfg = _mmcv.Config.fromfile(config_file)

    # Set dynamic classes
    num_classes = len(fg_classes) + len(bg_classes)
    cfg.model.decode_head.num_classes = num_classes
    cfg.model.decode_head.text_categories = num_classes

    # We'll provide text embeddings dynamically (no file)
    cfg.model.decode_head.text_embeddings_path = None

    # Set CLIP visual projection weights (required)
    cfg.model.decode_head.visual_projs_path = clip_weights_path

    # Provide classes to the demo dataset so palette/labels are correct
    cfg.data.test.fg_classes = fg_classes
    cfg.data.test.bg_classes = bg_classes

    return cfg


def compute_text_embeddings(
    all_classnames: List[str],
    use_prompt_engineering: bool,
    clip_model: str = "ViT-B/16",
) -> torch.Tensor:
    """Compute CLIP zeroshot weights for the provided classnames.

    Returns a CPU tensor of shape [num_classes, embed_dim]. Works on CPU or CUDA.
    """
    templates = DEFAULT_TEMPLATES if use_prompt_engineering else ["a photo of a {}."]
    import clip  # lazy import to allow startup even if not installed yet

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load(clip_model, device=device)
    model.eval()
    with torch.no_grad():
        zs = []
        for cname in all_classnames:
            texts = [t.format(cname) for t in templates]
            tokens = clip.tokenize(texts).to(device)
            class_embeds = model.encode_text(tokens)
            class_embeds = class_embeds / class_embeds.norm(dim=-1, keepdim=True)
            class_embed = class_embeds.mean(dim=0)
            class_embed = class_embed / class_embed.norm()
            zs.append(class_embed)
        w = torch.stack(zs, dim=0).float().cpu()  # [num_classes, embed_dim]
    return w


def ensure_tensor_on_device(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    if str(device).startswith("cuda"):
        return t.to(device)
    return t


# Cache last model to avoid rebuilding when class list doesn't change
_cached = {
    "classes_key": None,
    "model": None,
}


def classes_key(fg: List[str], bg: List[str], device: str, ckpt: str, clip_w: str) -> str:
    return f"fg={tuple(fg)}|bg={tuple(bg)}|dev={device}|ckpt={os.path.abspath(ckpt)}|clip={os.path.abspath(clip_w)}"


def get_or_build_model(
    fg_classes: List[str],
    bg_classes: List[str],
    checkpoint_path: str,
    clip_weights_path: str,
    device: str,
):
    key = classes_key(fg_classes, bg_classes, device, checkpoint_path, clip_weights_path)
    if _cached.get("classes_key") == key and _cached.get("model") is not None:
        return _cached["model"], False

    # Resolve config path
    config_file = resolve_paths("configs/maskclip/maskclip_vit16_512x512_demo.py")

    # Validate files exist
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.isfile(clip_weights_path):
        raise FileNotFoundError(f"CLIP visual weights not found: {clip_weights_path}")

    cfg = build_config_for_classes(config_file, clip_weights_path, fg_classes, bg_classes)

    # Build model
    try:
        model = init_segmentor(cfg, checkpoint_path, device=device)
    except Exception as e:
        # Helpful hint if CUDA is selected but not available
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but not available. Select device=cpu or install CUDA.") from e
        raise

    _cached["classes_key"] = key
    _cached["model"] = model
    return model, True


def to_bgr(image: np.ndarray) -> np.ndarray:
    # gradio provides RGB; mmseg expects BGR for visualization
    if image.ndim == 3 and image.shape[2] == 3:
        return image[:, :, ::-1].copy()
    return image


def to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] == 3:
        return image[:, :, ::-1].copy()
    return image


def run_inference(
    image: np.ndarray,
    prompt_text: str,
    bg_text: str,
    use_prompt_eng: bool,
    opacity: float,
    device_choice: str,
    checkpoint_path: str,
    clip_weights_path: str,
) -> Tuple[np.ndarray, str]:
    # Parse classes
    fg_classes = parse_classes(prompt_text)
    bg_classes = parse_classes(bg_text) if bg_text.strip() else list(DEFAULT_BG_CLASSES)

    if len(fg_classes) == 0:
        return None, "Please provide at least one foreground class in the prompt."

    # Build or reuse model with these classes
    model, rebuilt = get_or_build_model(fg_classes, bg_classes, checkpoint_path, clip_weights_path, device_choice)

    # Build text embeddings on the fly and load into head
    all_classes = fg_classes + bg_classes
    text_embed = compute_text_embeddings(all_classes, use_prompt_eng)  # [num_classes, 512]

    # Place embeddings on correct device and load into model
    device = next(model.parameters()).device
    text_embed = ensure_tensor_on_device(text_embed, device)
    with torch.no_grad():
        # Replace or copy into the existing parameter/buffer
        de = model.decode_head
        if hasattr(de, "text_embeddings"):
            if de.text_embeddings.shape != text_embed.shape:
                # Replace with a new Parameter or buffer matching type
                if isinstance(de.text_embeddings, torch.nn.Parameter):
                    de.text_embeddings = torch.nn.Parameter(text_embed)
                else:
                    # unregister old buffer by re-registering the same name
                    delattr(de, "text_embeddings")
                    de.register_buffer("text_embeddings", text_embed)
            else:
                de.text_embeddings.copy_(text_embed)
        else:
            de.register_buffer("text_embeddings", text_embed)
        # Keep book-keeping fields consistent
        if hasattr(de, "text_categories"):
            de.text_categories = text_embed.shape[0]

    # Inference: save to a temp file to be robust to mmcv.imread
    bgr = to_bgr(image)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(tmp_fd)
    try:
        mmcv.imwrite(bgr, tmp_path)
        result = inference_segmentor(model, tmp_path)
        # Use internal visualizer to overlay mask
        if hasattr(model, "module"):
            show_model = model.module
        else:
            show_model = model
        vis_bgr = show_model.show_result(tmp_path, result, show=False, opacity=float(opacity))
        vis_rgb = to_rgb(vis_bgr)
        info = (
            f"Classes: {', '.join(all_classes)}\n"
            f"Model{' rebuilt' if rebuilt else ' reused'} on device {device_choice}"
        )
        return vis_rgb, info
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


# ------- UI -------

def build_ui():
    with gr.Blocks(title="MaskCLIP Gradio Demo") as demo:
        gr.Markdown("""
        # MaskCLIP: Text-guided Segmentation
        - Enter one or more foreground classes (comma-separated). Example: `pedestrian, car, bicycle`
        - Optionally adjust background classes. Defaults to: building, ground, grass, tree, sky.
        - Put model weights in `pretrain/` or set custom paths below.
        """)
        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(type="numpy", label="Input Image")
                prompt_in = gr.Textbox(label="Foreground classes (comma-separated)", value="pedestrian, car, bicycle")
                bg_in = gr.Textbox(label="Background classes (comma-separated, optional)", value=", ".join(DEFAULT_BG_CLASSES))
                use_pe = gr.Checkbox(value=True, label="Use prompt engineering templates")
                opacity = gr.Slider(minimum=0.1, maximum=1.0, value=0.6, step=0.05, label="Overlay opacity")
            with gr.Column(scale=1):
                device = gr.Dropdown(choices=["cuda:0", "cpu"], value="cuda:0", label="Device")
                ckpt = gr.Textbox(label="Backbone checkpoint path", value=os.path.join(REPO_ROOT, "pretrain/ViT16_clip_backbone.pth"))
                clip_w = gr.Textbox(label="CLIP visual weights path", value=os.path.join(REPO_ROOT, "pretrain/ViT16_clip_weights.pth"))
                run_btn = gr.Button("Segment")
        with gr.Row():
            out_img = gr.Image(type="numpy", label="Segmentation Overlay")
        out_info = gr.Textbox(label="Info", interactive=False)

        run_btn.click(
            fn=run_inference,
            inputs=[image_in, prompt_in, bg_in, use_pe, opacity, device, ckpt, clip_w],
            outputs=[out_img, out_info],
        )
    return demo


if __name__ == "__main__":
    demo = build_ui()
    # On Windows/cmd, default server
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
