import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import os
import io

st.set_page_config(
    page_title="Visionary AI | Car Brand Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_css():
    st.markdown("""
        <style>
            .main .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            .card {
                background-color: #f0f2f6; /* A light grey for a professional feel */
                border: 1px solid #e6e6e6;
                border-radius: 10px;
                padding: 2rem;
            }
            div[data-testid="stMetric"] {
                background-color: transparent;
                border: none;
                padding: 0;
            }
            div[data-testid="stMetric"] > div:nth-child(2) > div {
                font-size: 2.5rem;
                font-weight: bold;
                color: #262730; /* Darker font for better contrast */
            }
        </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model_resources():
    """Loads the model, labels, and device, caching the result."""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_DIR = './dataset/DATA'
    if not Path(DATA_DIR).is_dir():
        return None, None, None
    num_classes = len([d for d in Path(DATA_DIR).iterdir() if d.is_dir()])
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model_path = "cars_brands.pth"
    if not Path(model_path).exists():
        return None, None, None
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    id2label = sorted([d.name for d in Path(DATA_DIR).iterdir() if d.is_dir()])
    print("Model and resources loaded successfully.")
    return model, id2label, DEVICE


def predict(image_bytes, model, id2label, device, top_k=3):
    """Processes an image and returns the Top-K predicted brands and their confidence scores."""
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean, std)
    ])
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)

    results = []
    with torch.no_grad():
        logits = model(x)
        probabilities = torch.nn.functional.softmax(logits, dim=1)

        # Get the top k predictions
        top_k_conf, top_k_indices = torch.topk(probabilities, top_k)

        # Squeeze to remove batch dimension
        top_k_conf = top_k_conf.squeeze().tolist()
        top_k_indices = top_k_indices.squeeze().tolist()

        for i in range(top_k):
            label = id2label[top_k_indices[i]]
            confidence = top_k_conf[i]
            results.append((label, confidence))

    return results


# --- Main Application ---
load_css()
model, id2label, DEVICE = load_model_resources()

# Sidebar for controls
with st.sidebar:
    st.title("Visionary AI")
    st.header("Control Panel")
    input_method = st.radio(
        "Choose your input method:",
        ("Upload an Image", "Select an Example")
    )
    st.divider()
    uploaded_file = None
    selected_image_file = None
    if input_method == "Upload an Image":
        uploaded_file = st.file_uploader(
            "Choose an image file", type=['jpg', 'jpeg', 'png']
        )
    else:
        TEST_IMG_DIR = "our_tests"
        if Path(TEST_IMG_DIR).is_dir():
            test_images = [""] + [f for f in os.listdir(TEST_IMG_DIR) if f.endswith(('jpg', 'jpeg', 'png'))]
            selected_image_file = st.selectbox(
                "Choose from our test images:", options=test_images
            )

# Main area for display
st.title("Car Brand Analyzer")
st.write("An advanced neural network for recognizing vehicle brands from images.")
st.divider()

image_bytes, caption = None, None
if uploaded_file:
    image_bytes = uploaded_file.getvalue()
    caption = "Uploaded Image"
elif selected_image_file:
    image_path = os.path.join(TEST_IMG_DIR, selected_image_file)
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    caption = f"Selected Example: {selected_image_file}"

# Display logic
if image_bytes and model:
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.image(image_bytes, caption=caption, use_container_width=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Analysis Result")
        with st.spinner("Processing image..."):
            # Get a list of predictions
            predictions = predict(image_bytes, model, id2label, DEVICE, top_k=3)

            # Display the top prediction
            top_prediction_label, top_prediction_conf = predictions[0]
            st.metric(label="Predicted Brand", value=top_prediction_label.upper())
            st.progress(top_prediction_conf, text=f"Confidence: {top_prediction_conf:.2%}")

            st.divider()

            # Display other possibilities
            st.write("Other Possibilities:")
            for label, confidence in predictions[1:]:
                st.text(f"{label.title()}:")
                st.progress(confidence)

        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Please select an input method and provide an image in the sidebar to begin.")
    if not model:
        st.error("Model could not be loaded. Please check the application configuration.")
