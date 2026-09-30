import io
import base64
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

_grad_model_cache = {}

def get_grad_model(model, target_layer_name: str):
    key = (id(model), target_layer_name)
    if key in _grad_model_cache:
        return _grad_model_cache[key]
    try:
        import keras
        grad_model = keras.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(target_layer_name).output, model.output]
        )
    except Exception:
        import tensorflow as tf
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(target_layer_name).output, model.output]
        )
    _grad_model_cache[key] = grad_model
    return grad_model

def compute_gradcam_plus_plus(model, input_tensor):
    try:
        import tensorflow as tf
    except ImportError:
        return None

    try:
        target_layer_name = None
        for name in ["top_activation", "se_block_scale", "top_conv"]:
            try:
                layer = model.get_layer(name)
                if len(layer.output.shape) == 4:
                    target_layer_name = name
                    break
            except Exception:
                continue

        if target_layer_name is None:
            for layer in reversed(model.layers):
                if len(layer.output.shape) == 4:
                    target_layer_name = layer.name
                    break

        if target_layer_name is None:
            return None

        grad_model = get_grad_model(model, target_layer_name)
        tensor = tf.cast(input_tensor, tf.float32)

        with tf.GradientTape() as tape_outer:
            with tf.GradientTape() as tape_inner:
                conv_output, preds = grad_model(tensor)
                tape_outer.watch(conv_output)
                tape_inner.watch(conv_output)
                class_score = preds[:, 0]
            grads_1st = tape_inner.gradient(class_score, conv_output)
        grads_2nd = tape_outer.gradient(grads_1st, conv_output)

        A = conv_output[0].numpy()
        g1 = grads_1st[0].numpy() if grads_1st is not None else np.zeros_like(A)

        if grads_2nd is not None:
            g2 = grads_2nd[0].numpy()
            sum_A_g2 = np.sum(A * g2, axis=(0, 1), keepdims=True)
            alpha_denom = 2.0 * g2 + sum_A_g2
            alpha = np.where(np.abs(alpha_denom) > 1e-8, g2 / (alpha_denom + 1e-8), 0.0)
            weights = np.sum(alpha * np.maximum(g1, 0), axis=(0, 1))
        else:
            weights = np.mean(np.maximum(g1, 0), axis=(0, 1))

        cam = np.sum(A * weights, axis=-1)
        cam = np.maximum(cam, 0)
        max_val = np.max(cam)
        cam = cam / (max_val + 1e-8) if max_val > 0 else cam
        return np.nan_to_num(cam)
    except Exception:
        return None

def generate_gradcam_overlay_base64(image_bytes: bytes, model=None, alpha: float = 0.45) -> str:
    try:
        orig_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Berkas citra tidak valid: {e}")
    w, h = orig_pil.size

    if model is not None:
        resized_pil = orig_pil.resize((96, 96), Image.Resampling.BILINEAR)
        inp = np.expand_dims(np.asarray(resized_pil, dtype=np.float32), axis=0)
        heatmap = compute_gradcam_plus_plus(model, inp)
    else:
        y_grid, x_grid = np.ogrid[:96, :96]
        dist_from_center = np.sqrt((x_grid - 48)**2 + (y_grid - 48)**2)
        heatmap = np.clip(1.0 - (dist_from_center / 48.0), 0.0, 1.0)

    if heatmap is None:
        heatmap = np.zeros((96, 96))

    heatmap_norm = np.clip(np.nan_to_num(heatmap), 0.0, 1.0)
    hmap_resized = np.array(
        Image.fromarray(np.uint8(255 * heatmap_norm)).resize((w, h), Image.Resampling.BILINEAR)
    ) / 255.0

    cmap = plt.get_cmap("jet")
    hmap_colored = cmap(hmap_resized)[:, :, :3]
    orig_arr = np.asarray(orig_pil, dtype=np.float32) / 255.0

    blended = (1.0 - alpha) * orig_arr + alpha * hmap_colored
    blended_u8 = np.uint8(np.clip(blended, 0.0, 1.0) * 255)

    result_pil = Image.fromarray(blended_u8)
    buf = io.BytesIO()
    result_pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
