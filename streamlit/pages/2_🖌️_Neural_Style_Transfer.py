import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import torch
from torchvision import transforms
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from torch import optim
from torchvision.transforms.functional import to_pil_image
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_image(header):
    st.subheader(header)
    uploaded_file = st.file_uploader("Upload an image file",
                                     type=["png", "jpg", "jpeg"],
                                     key=header + "file")

    url = st.text_input("Or enter an image URL",
                        placeholder="https://example.org",
                        key=header + "url")

    image = None
    if uploaded_file and url:
        st.info("Please choose a single upload method.")
        return image

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
        st.image(image, caption="Selected Image", use_container_width=True)
    else:
        st.info("Please upload an image or enter an image URL.")

    return image


imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


def get_hook(name, features):

    def hook(module, input, output):
        features[name] = output

    return hook


def get_image_tensor(img, img_height):
    w, h = img.size
    img_transforms = transforms.Compose([
        transforms.Resize((img_height, int(img_height * (w / h)))),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    return img_transforms(img).unsqueeze(0).to(device)


def gram_matrix(features):
    _, n_filters, height, width = features.size()
    resized_features = features.view(n_filters, height * width)

    result = torch.mm(resized_features, resized_features.t())
    return result.div(n_filters * height * width)


def register_hooks(model, layers):
    features = {}
    handles = []

    for name in layers:
        tokens = name.split(".")
        module = model
        for token in tokens:
            module = getattr(module, token)

        handle = module.register_forward_hook(get_hook(name, features))
        handles.append(handle)

    return features, handles


def get_image_features(model, layers, img_tensor):
    features, handles = register_hooks(model, layers)

    model(img_tensor)

    for handle in handles:
        handle.remove()

    return features


def normalize(img):
    imagenet_mean_tensor = torch.tensor(imagenet_mean).to(device).view(
        1, 3, 1, 1)
    imagenet_std_tensor = torch.tensor(imagenet_std).to(device).view(
        1, 3, 1, 1)

    return (img - imagenet_mean_tensor) / imagenet_std_tensor


@st.fragment
def download_generated_image(buffer):
    st.download_button(label="Download JPG",
                       data=buffer,
                       file_name="generated_image.jpg",
                       mime="image/jpeg",
                       icon=":material/download:",
                       use_container_width=True)


def train(model, input_img, epochs, content_features, style_features,
          input_features, content_weight, style_weight):
    input_img.requires_grad_(True)
    optimizer = optim.LBFGS([input_img])

    progress_bar = st.progress(0, text="Progress")
    _, col2, _ = st.columns([2, 4, 2])
    with col2:
        my_image = st.image(to_pil_image(input_img.squeeze(0).cpu()))

    col1, col2, col3 = st.columns([2, 3, 2])
    with col1:
        time_metric = st.metric("Elapsed time", None)

    with col2:
        content_loss_metric = st.metric("Content loss", None)

    with col3:
        style_loss_metric = st.metric("Style loss", None)

    start_time = time.time()

    epoch = [0]
    while epoch[0] < epochs:

        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)

            if epoch[0] % 10 == 0:
                my_image.image(to_pil_image(input_img.squeeze(0).cpu()))

            normalized_input = normalize(input_img)

            optimizer.zero_grad()
            model(normalized_input)

            content_loss = 0.0
            for layer, f in content_features.items():
                content_loss += torch.nn.MSELoss()(input_features[layer], f)

            content_loss /= len(content_features)
            content_loss *= content_weight

            style_loss = 0.0
            for layer, f in style_features.items():
                style_loss += torch.nn.MSELoss(reduction='sum')(gram_matrix(
                    input_features[layer]), f)

            style_loss /= len(style_features)
            style_loss *= style_weight

            total_loss = content_loss + style_loss

            total_loss.backward()

            epoch[0] += 1

            progress_bar.progress(epoch[0] * 1.0 / epochs, text="Progress")

            elapsed = int(time.time() - start_time)
            time_metric.metric("Elapsed time", f"{elapsed} s")

            if epoch[0] % 5 == 0:
                content_loss_metric.metric("Content loss",
                                           round(content_loss.item(), 4))
                style_loss_metric.metric("Style loss",
                                         round(style_loss.item(), 4))

            return total_loss

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    progress_bar.empty()

    img_pil = to_pil_image(input_img.squeeze(0).cpu())
    buffer = BytesIO()
    img_pil.save(buffer, format="JPEG")
    buffer.seek(0)

    download_generated_image(buffer)


def transfer_style(content_image, style_image, content_layers, style_layers,
                   content_weight, style_weight, image_height, epochs,
                   model_name):

    content_tensor = get_image_tensor(content_image, image_height)
    style_tensor = get_image_tensor(style_image, image_height)

    content_transform = transforms.Compose([
        transforms.Resize((content_tensor.shape[2], content_tensor.shape[3])),
        transforms.ToTensor()
    ])

    input_img = content_transform(content_image).unsqueeze(0).clone().to(
        device)
    input_layers = content_layers + style_layers

    model = vgg19(VGG19_Weights.DEFAULT).to(
        device).features if model_name == "VGG19" else deeplabv3_resnet101(
            DeepLabV3_ResNet101_Weights.DEFAULT).to(device).backbone

    model.eval()
    model.requires_grad_(False)

    content_features = get_image_features(model, content_layers,
                                          content_tensor)

    style_features = get_image_features(model, style_layers, style_tensor)
    for layer in style_features:
        style_features[layer] = gram_matrix(style_features[layer])

    input_features, _ = register_hooks(model, input_layers)

    train(model, input_img, epochs, content_features, style_features,
          input_features, content_weight, style_weight)


@st.cache_data
def get_vgg_layers():
    model = vgg19(weights=None).to(device).features
    return [
        f'{name}({type(module).__name__})'
        for name, module in model.named_modules() if name
    ]


@st.cache_data
def get_deeplabv3_resnet101_layers():
    model = deeplabv3_resnet101(weights=None).to(device).backbone
    return [
        f'{name}({type(module).__name__})'
        for name, module in model.named_modules() if name
    ]


st.image(
    "https://miro.medium.com/v2/resize:fit:1198/1*ant6-qGX1WP3-F_2F_UStA.png",
    use_container_width=True)

st.title("🖌️ Neural Style Transfer")

col1, col2 = st.columns(2)
with col1:
    content_image = get_image("Content image")
with col2:
    style_image = get_image("Style image")

st.divider()

content_weight = 1.0
image_height = 512

model_name = st.selectbox("Feature extractor model",
                          ["VGG19", "DeepLabV3 ResNet-101"])

if model_name == "VGG19":
    content_layers = ["22(ReLU)"]
    style_layers = ["1(ReLU)", "6(ReLU)", "11(ReLU)", "20(ReLU)", "29(ReLU)"]
    style_weight = 10.0**5
    epochs = 300
    layers = get_vgg_layers()
else:
    content_layers = ['layer4.2.relu(ReLU)']
    style_layers = [
        'relu(ReLU)', 'layer1.1.relu(ReLU)', 'layer2.1.relu(ReLU)',
        'layer2.3.relu(ReLU)', 'layer3.1.relu(ReLU)'
    ]
    style_weight = 10.0**6
    epochs = 100
    layers = get_deeplabv3_resnet101_layers()

default_config = st.checkbox("Use default configuration", value=True)
if not default_config:
    epochs = st.slider("Number of epochs",
                       min_value=0,
                       max_value=1000,
                       value=epochs,
                       step=20)
    image_height = st.slider("Output image height",
                             min_value=200,
                             max_value=1440,
                             value=image_height)

    col1, col2 = st.columns(2)
    with col1:
        content_weight = st.number_input("Content weight",
                                         min_value=0.0,
                                         value=content_weight)
        content_layers = st.multiselect("Content layers",
                                        layers,
                                        default=content_layers)

    with col2:
        style_weight = st.number_input("Style weight",
                                       min_value=0.0,
                                       value=style_weight)
        style_layers = st.multiselect("Style layers",
                                      layers,
                                      default=style_layers)

if content_image and style_image:
    if st.button("Run", use_container_width=True):
        content_layers = [cl.split("(", 1)[0] for cl in content_layers]
        style_layers = [sl.split("(", 1)[0] for sl in style_layers]

        transfer_style(content_image, style_image, content_layers,
                       style_layers, content_weight, style_weight,
                       image_height, epochs, model_name)
