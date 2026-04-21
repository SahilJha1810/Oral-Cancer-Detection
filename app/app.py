import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from src.model import load_model

st.title("🧠 Oral Cancer Detection System")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
model_b1 = load_model("efficientnet_b1", "models/b1.pth", device)
model_b2 = load_model("efficientnet_b2", "models/b2.pth", device)

transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.ToTensor()
])

uploaded_file = st.file_uploader("Upload Oral Image")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        p1 = torch.softmax(model_b1(img), dim=1)
        p2 = torch.softmax(model_b2(img), dim=1)

        probs = (p1 + p2) / 2
        confidence = probs[0][1].item()
        pred = torch.argmax(probs, 1).item()

    label = "Cancer" if pred == 1 else "Non-Cancer"

    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: {confidence:.4f}")
