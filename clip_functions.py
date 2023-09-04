import torch
import clip
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32")

def preprocess_text(text):
    text_tokens = clip.tokenize([text]).to(device)
    return text_tokens

def calculate_text_features(text_query):
    text_tokens = clip.tokenize([text_query]).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features

def calculate_image_features(image):
    image = preprocess(image).unsqueeze(0).to(device)
    
    image_features = model.encode_image(image)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    
    return image_features
