import numpy as np
import csv
import os
import streamlit as st
import cv2 as cv
import time
from dotenv import load_dotenv

from processing import set_random_seed, define_model, create_annoy_index, AnnoyIndex, preprocess_query_image, search_image
from get_data import download_database, unzip_file

img_folder = "Flickr/mirflickr"

# Đặt hạt giống
set_random_seed(50)

# Create a feature extractor model
feature_extractor = define_model()

load_dotenv() 

with st.spinner("Getting data..."):
    try:
        if not os.path.exists('Flickr'):
            with st.spinner("Downloading and unzip dataset..."):
                download_database(os.getenv("Dataset"), "Flickr.zip")
                unzip_file("Flickr.zip")
            
            with st.spinner("Downloading features..."):
                download_database(os.getenv("Index"), "database/index.ann")
                download_database(os.getenv("CSV"), "database/features.csv")
                download_database(os.getenv("Features"), "database/features.npy")
                
            st.success("Finish getting data!")
    except Exception as e:
        st.error('Failed to get database!')
        print(f'Error when getting database, ${e}')
        
    try:
        # Load image names from CSV file
        image_names = []
        with open("database/features.csv", 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                image_names.append(row[0])

        # Load precomputed features
        features = np.load("database/features.npy")

        # Create and load Annoy index
        if not os.path.exists("database/index.ann"):
            index = create_annoy_index(features)
            index.save("database/index.ann")

        index = AnnoyIndex(features.shape[1], 'euclidean')
        index.load("database/index.ann")
    except Exception as e:
        st.error('Failed to load features and index')
        print(f'Error when loading features and index, ${e}')

st.title("Social Image Retrieval System")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    top_k = st.sidebar.number_input("Number of top K results", min_value=1, max_value=100, value=5)
    btn_searching = st.sidebar.button("Search")
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
            similar_images = search_image(query_feature[0], index, image_names, top_k=top_k)
            end = time.time()
            run_time = end - start
            
            st.write(f'Runtime: {run_time:.2f} secs. Similar images:')
            
            # Display similar images
            index = 1
            for image_name in similar_images:
                image_path =  f'{img_folder}/{image_name}'
                similar_img = cv.imread(image_path)
                caption = f'Image {index}: {image_name}'
                st.image(similar_img, channels="BGR", caption=caption, use_column_width=True)
                index+=1
    
