import numpy as np
import csv
import os
import streamlit as st
import cv2 as cv
import random
from tensorflow import random as tfrand
import time

from processing import set_random_seed, define_model, create_annoy_index, AnnoyIndex, preprocess_query_image, search_image
    
# Đặt hạt giống
set_random_seed(50)

# Create a feature extractor model
feature_extractor = define_model()

folder =  f"D:/HCMUS/HK9/VIR"
names_path = f"{folder}/features.csv"
feature_path = f"{folder}/features.npy"
index_path = f"{folder}/index.ann"

with st.spinner("Loading data..."):
    # Load image names from CSV file
    image_names = []
    with open(names_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            image_names.append(row[0])

    # Load precomputed features
    features = np.load(feature_path)

    # Create and load Annoy index
    if not os.path.exists(index_path):
        index = create_annoy_index(features)
        index.save(index_path)

    index = AnnoyIndex(features.shape[1], 'euclidean')
    index.load(index_path)

st.title("Social Image Retrieval System")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
btn_searching = st.sidebar.button("Search")

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv.imdecode(file_bytes, 1)
    
    # Display the uploaded image
    st.sidebar.image(img, channels="BGR", caption='Uploaded Image.', use_column_width=True)
    
    if btn_searching:
        with st.spinner("Processing image..."):
            query_img_array = preprocess_query_image(img)
            query_feature = feature_extractor.predict(query_img_array)
            
        with st.spinner("Searching..."):
            start = time.time()
            similar_images = search_image(query_feature[0], index, image_names, top_k=5)
            end = time.time()
            run_time = end - start
            
            st.sidebar.write(f'Runtime: {run_time:.2f} secs')
            
            st.write("Similar images:")
            
            # Display similar images
            for image_name in similar_images:
                image_path = f'{folder}/Dataset/Flickr/flickr30k_images/flickr30k_images/{image_name}'
                similar_img = cv.imread(image_path)
                st.image(similar_img, channels="BGR", caption=image_name, use_column_width=True)
    
