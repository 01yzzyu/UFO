"""Shared image helpers: base64 encoding (with optional downscaling) and saving."""

import base64
import io
import os

from PIL import Image


def mime_of(path):
    ext = os.path.splitext(path or "")[1].lower()
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    return "image/jpeg"


def encode_b64(path, max_size=2048):
    """Return (base64_str, mime, error). Downscales large images to JPEG."""
    if not path or not os.path.exists(path):
        return None, None, f"not_found:{path}"
    try:
        with Image.open(path) as img:
            if max_size and (img.width > max_size or img.height > max_size):
                img.thumbnail((max_size, max_size))
                buf = io.BytesIO()
                img.convert("RGB").save(buf, format="JPEG", quality=95)
                return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg", None
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8"), mime_of(path), None
    except Exception as e:  # noqa: BLE001
        return None, None, f"read_fail:{type(e).__name__}"


def encode_data_url(path, max_size=2048):
    """Return (data_url, error) for OpenAI-style image_url inputs."""
    b64, mime, err = encode_b64(path, max_size=max_size)
    if err:
        return None, err
    return f"data:{mime};base64,{b64}", None


def save_b64_image(b64_str, save_path):
    """Decode a base64 image string and write it to disk. Returns ok(bool)."""
    try:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(base64.b64decode(b64_str))
        return True
    except Exception:  # noqa: BLE001
        return False


def concat_images(pil_images, direction="horizontal", bg=(255, 255, 255)):
    """Merge multiple PIL images into one (for models that accept a single
    image). Images are resized to a common height (horizontal) or width
    (vertical) before concatenation. Returns the single PIL image (or the lone
    image if only one is given)."""
    imgs = [im.convert("RGB") for im in pil_images if im is not None]
    if not imgs:
        return None
    if len(imgs) == 1:
        return imgs[0]
    if direction == "horizontal":
        h = min(im.height for im in imgs)
        imgs = [im.resize((max(1, int(im.width * h / im.height)), h)) for im in imgs]
        total_w = sum(im.width for im in imgs)
        canvas = Image.new("RGB", (total_w, h), bg)
        x = 0
        for im in imgs:
            canvas.paste(im, (x, 0)); x += im.width
        return canvas
    w = min(im.width for im in imgs)
    imgs = [im.resize((w, max(1, int(im.height * w / im.width)))) for im in imgs]
    total_h = sum(im.height for im in imgs)
    canvas = Image.new("RGB", (w, total_h), bg)
    y = 0
    for im in imgs:
        canvas.paste(im, (0, y)); y += im.height
    return canvas


def download_to(url, save_path, timeout=120):
    """Download an image URL to disk. Returns ok(bool)."""
    import urllib.request
    try:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = r.read()
        with open(save_path, "wb") as f:
            f.write(data)
        return True
    except Exception:  # noqa: BLE001
        return False
