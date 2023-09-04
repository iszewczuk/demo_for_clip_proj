import gradio as gr
from custom_functions_dataset import clean_and_convert
from clip_functions import calculate_text_features, calculate_image_features
import numpy as np
import pandas as pd
from PIL import Image
import torch
import clip
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32")

fashion_data = pd.read_csv('fashion_set1.csv')

fashion_data['text_features'] = fashion_data['text_features'].apply(clean_and_convert)
fashion_data['image_features'] = fashion_data['image_features'].apply(clean_and_convert)

categories0 = ["Dress", "Top", "Pants", "Blazer", "Jacket", 
              "High heels", "Boots", "Sneakers", "Tshirt"]

categories = ["All", "Dress", "Top", "Pants", "Blazer", "Jacket", 
              "High heels", "Boots", "Sneakers", "Tshirt"]

def find_similar_images_to_text(text_query, category=None, k=3):
    query_text_features = calculate_text_features(text_query)

    similarity_scores = []
    if category != "all":
        filtered_data = fashion_data[fashion_data['category'] == category.lower()]
    else:
        filtered_data = fashion_data

    for index, row in filtered_data.iterrows():
        row_img_features = torch.tensor(row['image_features'], device=device, dtype=torch.float32)
        
        query_text_features = query_text_features.squeeze()
        row_img_features = row_img_features.squeeze()
        
        similarity = torch.dot(query_text_features, row_img_features)
        similarity_scores.append((index, similarity.item()))

    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_similar_images = similarity_scores[:k]

    similar_images_data = fashion_data.iloc[[index for index, _ in top_similar_images]]
    return similar_images_data, top_similar_images

def find_similar_images_to_img(image, needed_category, num_similar=3):
    image_pil = Image.fromarray(image)
    query_img_features = calculate_image_features(image_pil)
    similarities = []
    
    if needed_category != "all":
        category_data = fashion_data[fashion_data['category'] == needed_category]
    else:
        category_data = fashion_data

    for index, row in category_data.iterrows():
        row_img_features = torch.tensor(row['image_features'], device=device, dtype=torch.float32)

        similarity = torch.dot(row_img_features, query_img_features.squeeze())
        similarities.append((row['full_path'], similarity.item()))

    similarities.sort(key=lambda x: x[1], reverse=True)
    similar_images = similarities[:num_similar]

    return similar_images

def find_similar_images_for_file_and_text(image, text_query, category=None, k=3, alpha=0.65):
    image_pil = Image.fromarray(image)
    query_img_features = calculate_image_features(image_pil)

    query_text_features = calculate_text_features(text_query)

    similarity_scores = []
    if category != "all":
        filtered_data = fashion_data[fashion_data['category'] == category]
    else:
        filtered_data = fashion_data

    for index, row in filtered_data.iterrows():
        row_img_features = torch.tensor(row['image_features'], device=device, dtype=torch.float32)

        similarity = alpha * torch.dot(query_text_features.squeeze(), row_img_features.squeeze()) + \
            (1 - alpha) * torch.dot(query_img_features.squeeze(), row_img_features.squeeze())
        
        similarity_scores.append((index, similarity.item()))

    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_similar_images = similarity_scores[:k]

    similar_images_data = fashion_data.loc[[index for index, _ in top_similar_images]]
    top_similarity_scores = [score for _, score in top_similar_images]
    similar_images_data['similarity_score'] = top_similarity_scores
    
    return similar_images_data

def catalog(category):
    items = fashion_data[fashion_data['category'] == category.lower()]
    
    images_with_captions = []
    target_size = (185, 250) 
    
    for _, row in items.iterrows():
        img_path = row['full_path']
        img = Image.open(img_path)
        img = img.resize(target_size)
        images_with_captions.append((img))

    return images_with_captions

def search_by_text(text, category):
    similar_images_data, similarity_scores = find_similar_images_to_text(text, category.lower(), k=3)
    images_with_captions = []

    for i, (_, row) in enumerate(similar_images_data.iterrows()):
        img_path = row['full_path']
        img = Image.open(img_path)
        images_with_captions.append((img)) 

    return images_with_captions

def search_by_image(image, category):
    category = category.lower()
    similar_images = find_similar_images_to_img(image, category)
    images_with_captions = []

    for img_path, _ in similar_images:
        img = Image.open(img_path)
        images_with_captions.append((img))

    return images_with_captions

def search_by_image_and_text(text, image, category, alpha):
    similar_images_data = find_similar_images_for_file_and_text(image, text, category.lower(), alpha=alpha)
    
    images_with_captions = []

    for _, row in enumerate(similar_images_data.iterrows()):
        img_path = row[1]['full_path']
        img = Image.open(img_path)
        images_with_captions.append((img)) 

    return images_with_captions


with gr.Blocks() as demo:
    gr.Markdown("Some features that can be done using CLIP")
    with gr.Tab("Catalog"):
        catalog_dropdown_input = gr.Dropdown(categories0, label="Choose needed category please")
        catalog_image_output_plot = gr.Gallery(columns=7, show_download_button=False, object_fit="contain", height="auto")

    with gr.Tab("Search by Text description"):
        sbt_text_input = gr.Textbox(label="Add text description of interested item please")
        sbt_dropdown_input = gr.Dropdown(categories, label="Choose needed category please")
        sbt_button = gr.Button("Show")
        sbt_image_output = gr.Gallery(preview=True)

    with gr.Tab("Similar to input img"):
        sti_image_input = gr.Image()
        sti_dropdown_input = gr.Dropdown(categories, label="Choose needed category please")
        sti_button = gr.Button("Show")
        sti_image_output = gr.Gallery(preview=True)

    with gr.Tab("Similar to input img and text"):
        img_and_txt_text_input = gr.Textbox(label="Add text description of interested item please")
        img_and_txt_image_input = gr.Image()
        img_and_txt_dropdown_input = gr.Dropdown(categories, label="Choose needed category please")
        img_and_txt_slider = gr.Slider(0, 1, value=0.65 ,label="Text impact strength:")
        img_and_txt_button = gr.Button("Show")
        img_and_txt_image_output = gr.Gallery(preview=True)

    
    catalog_dropdown_input.change(catalog, inputs=catalog_dropdown_input, outputs=catalog_image_output_plot)
    sbt_button.click(search_by_text, inputs=[sbt_text_input, sbt_dropdown_input], outputs=sbt_image_output)
    sti_button.click(search_by_image, inputs=[sti_image_input, sti_dropdown_input], outputs=sti_image_output)
    img_and_txt_button.click(search_by_image_and_text, 
                             inputs=[img_and_txt_text_input, img_and_txt_image_input, img_and_txt_dropdown_input, img_and_txt_slider], 
                             outputs=img_and_txt_image_output)


if __name__ == "__main__":
    demo.launch(share=True)
