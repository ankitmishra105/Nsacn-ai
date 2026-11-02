import os

import matplotlib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input as densenet_pre

# ------------------------------- CONFIG ---------------------------------------
# 1) Path to your trained model. Two formats are supported:
#    - full model:  "DenseNet121_brain_tumor.keras"   (saved by the notebook of the training script)
#    - weights only: "outputs/DenseNet121.weights.h5"  (weights-only)
MODEL_PATH = "DenseNet121_brain_tumor.keras"

# 2) Must be the same IMG_SIZE you trained with (the notebook used 256).
IMG_SIZE = 256

# 3) Predictions below this confidence show a warning.
CONFIDENCE_THRESHOLD = 0.60

# Class order must match training. Training used sorted(folder names), so we rebuild it the same way.
TUMOR_TYPES = ["Astrocytoma", "Ependymoma", "Glioma", "Hemangiopericytoma", "Meningioma",
               "Neurocytoma", "Normal", "Oligodendroglioma", "Other", "Schwannoma"]
WEIGHTINGS = ["T1", "T1C+", "T2"]
CLASS_NAMES = sorted(f"{t} {w}" for t in TUMOR_TYPES for w in WEIGHTINGS)
NUM_CLASSES = len(CLASS_NAMES)


# ---------------------------- MODEL LOADING -----------------------------------
def build_model():
    """Same architecture as the training script (needed only when loading a weights file)."""
    base = DenseNet121(include_top=False, weights=None, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.05),
            layers.RandomTranslation(0.03, 0.03),
        ],
        name="augmentation",
    )
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = augmentation(inputs)
    x = layers.Lambda(densenet_pre, name="preprocess")(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)
    return keras.Model(inputs, outputs, name="DenseNet121")


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if MODEL_PATH.endswith(".keras"):
        return keras.models.load_model(
            MODEL_PATH, compile=False, safe_mode=False,
            custom_objects={"preprocess_input": densenet_pre},
        )
    model = build_model()
    model.load_weights(MODEL_PATH)
    return model


# ---------------------------- PREDICTION --------------------------------------
def preprocess_image(pil_img):
    """Same steps as training: RGB -> resize (antialiased). The model scales pixels internally,
    so we pass raw 0-255 values (do NOT divide by 255)."""
    arr = np.array(pil_img.convert("RGB"))
    arr = tf.image.resize(arr, (IMG_SIZE, IMG_SIZE), antialias=True)
    arr = tf.cast(tf.round(arr), tf.uint8)
    return tf.cast(arr, tf.float32).numpy()[None, ...]      # shape (1, H, W, 3)


def predict(model, batch, use_tta):
    probs = model(batch, training=False).numpy()[0]
    if use_tta:    # average with the horizontally flipped image, as in training evaluation
        flipped = np.ascontiguousarray(batch[:, :, ::-1, :])
        probs = (probs + model(flipped, training=False).numpy()[0]) / 2.0
    return probs


def tumor_type_probs(probs):
    """Add up the T1 / T1C+ / T2 probabilities of each tumor type."""
    totals = {t: 0.0 for t in TUMOR_TYPES}
    for name, p in zip(CLASS_NAMES, probs):
        totals[name.rsplit(" ", 1)[0]] += float(p)
    return totals


# ----------------------------- GRAD-CAM ---------------------------------------
def grad_cam(model, batch, class_idx):
    """Grad-CAM: which parts of the image pushed the model toward `class_idx`.

    Idea: take the last convolutional feature maps of the backbone, measure how much the class
    score changes with each map (gradient), and use that as the map's importance weight.
    The weighted sum of the maps is the heatmap.
    """
    # Split the model into: [preprocess] -> [backbone (nested model)] -> [head layers]
    # The backbone is the big nested model (hundreds of layers); the augmentation block is a small one.
    idx = next((i for i, layer in enumerate(model.layers)
                if isinstance(layer, keras.Model) and len(layer.layers) > 20), None)
    if idx is None:
        raise RuntimeError("Could not find the DenseNet121 backbone inside the loaded model. "
                           "Top-level layers found: " + ", ".join(l.name for l in model.layers))
    backbone = model.layers[idx]
    head_layers = model.layers[idx + 1:]
    preprocess = model.get_layer("preprocess")

    x = preprocess(tf.convert_to_tensor(batch))
    with tf.GradientTape() as tape:
        conv = tf.cast(backbone(x, training=False), tf.float32)   # last conv feature maps (e.g. 8x8x1024)
        tape.watch(conv)
        y = conv
        for layer in head_layers[:-1]:                            # pooling, BN, dropout, dense ...
            y = layer(y, training=False)
        last = head_layers[-1]                                    # final softmax Dense layer
        kernel, bias = last.get_weights()
        logits = tf.matmul(tf.cast(y, tf.float32), tf.constant(kernel)) + tf.constant(bias)
        score = logits[:, class_idx]                              # use the logit (before softmax)

    grads = tape.gradient(score, conv)
    if grads is None:
        raise RuntimeError("No gradient reached the feature maps (the model graph looks disconnected).")
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)   # importance of each feature map
    raw = tf.reduce_sum(weights * conv, axis=-1)[0].numpy()
    cam = np.maximum(raw, 0)
    if cam.max() <= 0:                       # nothing positive: fall back to the raw map instead of a blank heatmap
        cam = raw - raw.min()
    return cam / (cam.max() + 1e-8)                               # scale to 0-1


def overlay_heatmap(pil_img, cam, alpha=0.4):
    """Resize the heatmap to the original image size, colour it and blend it on top."""
    img = pil_img.convert("RGB")
    heat = Image.fromarray(np.uint8(255 * cam)).resize(img.size, Image.BILINEAR)
    heat = np.array(heat) / 255.0
    colored = (matplotlib.colormaps["jet"](heat)[..., :3] * 255).astype(np.uint8)
    blended = (1 - alpha) * np.array(img) + alpha * colored
    return Image.fromarray(blended.astype(np.uint8))


# ------------------------------- UI -------------------------------------------
st.set_page_config(page_title="NeuroScan AI", page_icon="🧠", layout="wide")

st.title("🧠 NScan AI")
st.caption("Brain tumor classification from MRI using a fine-tuned DenseNet121")

with st.sidebar:
    st.header("Settings")
    use_tta = st.checkbox("Test-time augmentation (flip)", value=True,
                          help="Averages predictions on the image and its mirror. Slightly more accurate.")
    show_cam = st.checkbox("Show Grad-CAM heatmap", value=True,
                           help="Highlights the image regions the model relied on for its prediction.")
    st.markdown("---")
    st.markdown("**Model:** DenseNet121 (ImageNet transfer learning)")
    st.markdown(f"**Input size:** {IMG_SIZE}×{IMG_SIZE}")
    st.markdown(f"**Classes:** {NUM_CLASSES} ({len(TUMOR_TYPES)} tumor types × 3 MRI weightings)")
    with st.expander("See all classes"):
        st.write(CLASS_NAMES)
    st.markdown("---")
    st.warning(
        "For research and education only. This tool is not a medical device and must not be "
        "used for diagnosis or treatment decisions."
    )

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: `{MODEL_PATH}`. Put it next to app.py or edit MODEL_PATH at the top of app.py.")
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.info("Check that MODEL_PATH, IMG_SIZE and your TensorFlow/Keras version match the ones used for training.")
    st.stop()

uploaded = st.file_uploader("Upload an axial brain MRI slice (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload an MRI image to get a prediction.")
    st.stop()

image = Image.open(uploaded)
batch = preprocess_image(image)
probs = predict(model, batch, use_tta)

order = np.argsort(probs)[::-1]
top_class = CLASS_NAMES[order[0]]
top_conf = float(probs[order[0]])
type_totals = tumor_type_probs(probs)
top_type = max(type_totals, key=type_totals.get)

left, right = st.columns([1, 1.3])

with left:
    st.subheader("Uploaded image")
    st.image(image, use_container_width=True)

with right:
    st.subheader("Prediction")
    c1, c2 = st.columns(2)
    c1.metric("Predicted class", top_class, f"{top_conf:.1%} confidence", delta_color="off")
    c2.metric("Tumor type (all weightings)", top_type, f"{type_totals[top_type]:.1%}", delta_color="off")

    if top_conf < CONFIDENCE_THRESHOLD:
        st.warning("Low confidence: the model is unsure about this image. Treat the result with extra caution.")

    st.markdown("**Top 5 classes**")
    top5 = pd.DataFrame({"Class": [CLASS_NAMES[i] for i in order[:5]],
                         "Probability": [float(probs[i]) for i in order[:5]]}).set_index("Class")
    st.bar_chart(top5)

    st.markdown("**Probability by tumor type**")
    type_df = (pd.DataFrame({"Tumor type": list(type_totals), "Probability": list(type_totals.values())})
               .sort_values("Probability", ascending=False).set_index("Tumor type"))
    st.bar_chart(type_df)

if show_cam:
    st.markdown("---")
    st.subheader("Grad-CAM: where the model looked")
    explain = st.selectbox("Explain which class?", [CLASS_NAMES[i] for i in order[:5]], index=0)
    try:
        cam = grad_cam(model, batch, CLASS_NAMES.index(explain))
        g1, g2 = st.columns(2)
        g1.image(image, caption="Original", use_container_width=True)
        g2.image(overlay_heatmap(image, cam), caption=f"Grad-CAM for: {explain}", use_container_width=True)
        st.caption("Warm colours (red/yellow) mark regions that pushed the model toward this class. "
                   "This shows where the model focused. It is not a tumor outline and it is coarse.")
    except Exception as e:
        st.error("Could not compute Grad-CAM. The details are below.")
        st.exception(e)

st.caption("The model only knows these 30 classes. A non-MRI or unusual image will still get a prediction, so use axial brain MRI slices only.")