import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import torch
from torchvision import transforms
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        stride = 2 if in_channels != out_channels else 1

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels,
                      out_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
        ) if in_channels != out_channels else None

        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.downsample(x) if self.downsample else x

        x = self.conv1(x)
        x = self.conv2(x)
        x += residual
        x = self.relu(x)

        return x


class ResNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block1 = nn.Sequential(ResidualBlock(64, 64),
                                    ResidualBlock(64, 64))

        self.block2 = nn.Sequential(ResidualBlock(64, 128),
                                    ResidualBlock(128, 128))

        self.block3 = nn.Sequential(ResidualBlock(128, 256),
                                    ResidualBlock(256, 256))

        self.block4 = nn.Sequential(ResidualBlock(256, 512),
                                    ResidualBlock(512, 512))

        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc1(x)

        return x


def load_model():
    resnet = ResNet()
    resnet.load_state_dict(torch.load("streamlit/resources/resnet_7.pt"))
    resnet.eval()

    return resnet


def classify(image):
    image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    transformed_image = image_transforms(image).unsqueeze(0)

    model = load_model()
    with torch.no_grad():
        pred = model(transformed_image).squeeze(0)

    pred_cls = pred.softmax(0)
    result = pred_cls.argmax().item()
    return result, pred_cls[result].item()


def get_image():
    uploaded_file = st.file_uploader("Upload an image file",
                                     type=["png", "jpg", "jpeg"])

    url = st.text_input("Or enter an image URL",
                        placeholder="https://example.org")

    image = None
    col2 = None
    if uploaded_file and url:
        st.info("Please choose a single upload method.")
        return image, col2

    if uploaded_file:
        image = Image.open(uploaded_file)
    elif url:
        try:
            response = requests.get(url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
        except Exception as e:
            st.error(f"Unable to load image from URL: {e}")

    if image:
        _, col2, _ = st.columns([2, 5, 2])
        with col2:
            st.image(image, caption="Selected Image", use_container_width=True)
    else:
        st.info("Please upload an image or enter an image URL.")

    return image, col2


st.image(
    "https://imgproxy.domestika.org/unsafe/w:1200/rs:fill/plain/src://blog-post-open-graph-covers/000/011/998/11998-original.jpg?1701110208",
    use_container_width=True)

st.title("🧑‍🎨 Author Classification")

if 'success_count' not in st.session_state:
    st.session_state['success_count'] = 0

if 'success_rate' not in st.session_state:
    st.session_state['success_rate'] = None

if 'success_delta' not in st.session_state:
    st.session_state['success_delta'] = None

col1, col2, col3 = st.columns([2.2, 2, 1])
with col2:
    st.metric("Success rate", st.session_state['success_rate'],
              st.session_state['success_delta'])

image, col = get_image()

authors = [
    'Albrecht Durer', 'Alfred Sisley', 'Edgar Degas', 'Francisco Goya',
    'Pablo Picasso', 'Paul Gauguin', 'Pierre-Auguste Renoir', 'Rembrandt',
    'Titian', 'Vincent van Gogh'
]

if image is not None:
    with col:
        result, probability = classify(image)
        probability = round(probability * 100, 2)

        if probability <= 50:
            color = 'red'
        elif probability < 75:
            color = 'orange'
        else:
            color = '#3dd56d'

        st.markdown(
            f'<h3 style="text-align: center;">Author: <span style="color: {color};">{authors[result]}</span></h3>',
            unsafe_allow_html=True)
        st.markdown(
            f'<h3 style="text-align: center;">Probability: <span style="color: {color};">{probability}%</span></h3>',
            unsafe_allow_html=True)

        st.text("")
        st.markdown(
            f'<p style="text-align: center;">Was the classification successful?</p>',
            unsafe_allow_html=True)

        col1, col2 = st.columns([2, 2])
        with col1:
            if st.button("Yes", use_container_width=True):
                st.session_state['success_count'] += 1
                st.session_state['total_count'] += 1

                success_rate = round((st.session_state['success_count'] /
                                      st.session_state['total_count']) * 100,
                                     2)

                if st.session_state['success_rate']:
                    st.session_state['success_delta'] = round(
                        success_rate - st.session_state['success_rate'], 2)

                st.session_state['success_rate'] = success_rate

                st.rerun()

        with col2:
            if st.button("No", use_container_width=True):
                st.session_state['total_count'] += 1

                success_rate = round((st.session_state['success_count'] /
                                      st.session_state['total_count']) * 100,
                                     2)

                if st.session_state['success_rate']:
                    st.session_state['success_delta'] = round(
                        success_rate - st.session_state['success_rate'], 2)

                st.session_state['success_rate'] = success_rate

                st.rerun()
