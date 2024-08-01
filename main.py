import os
import csv
import time
import numpy as np
import streamlit as st
import cv2 as cv
from dotenv import load_dotenv

from processing import (
    set_random_seed, define_model, create_annoy_index, AnnoyIndex,
    preprocess_query_image, search_image
)
from get_data import download_database, unzip_file
from label_handle import load_label_from_path

IMG_FOLDER = "mirflickr"
LABEL_FOLDER = "mirflickr/meta/tags"
DATABASE_FOLDER = "database"
FEATURES_CSV = os.path.join(DATABASE_FOLDER, "features.csv")
FEATURES_NPY = os.path.join(DATABASE_FOLDER, "features.npy")
INDEX_FILE = os.path.join(DATABASE_FOLDER, "index.ann")


def setup_environment():
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    set_random_seed(50)
    load_dotenv()


@st.cache_resource(show_spinner=True)
def load_data():
    image_names, features, index = None, None, None
    load_time = time.time()
    try:
        with open(FEATURES_CSV, 'r') as f:
            image_names = [row[0] for row in csv.reader(f)]

        features = np.load(FEATURES_NPY)

        if not os.path.exists(INDEX_FILE) and features is not None:
            index = create_annoy_index(features)
            index.save(INDEX_FILE)

        index = AnnoyIndex(features.shape[1], 'euclidean')
        index.load(INDEX_FILE)
        
        if image_names is None or features is None or index is None:
            st.error('Failed to load features and index')
    except Exception as e:
        print(f'Error when loading features and index: {e}')
    finally:
        load_time = time.time() - load_time
        print(f'Load data time: {load_time:.2f} secs')
        return image_names, index

def createModel():
    extractor = None
    load_time = time.time()
    try:
        extractor = define_model()
    except Exception as e:
        print(f'Error when creating model: {e}')
    finally:
        load_time = time.time() - load_time
        print(f'Create model time: {load_time:.2f} secs')
        
        return extractor


@st.cache_resource(show_spinner=True)
def load_data_and_create_model():
    image_names, index = load_data()
    feature_extractor = createModel()
    return image_names, index, feature_extractor


def download_and_prepare_data():
    if not os.path.exists(IMG_FOLDER):
        with st.spinner("Downloading and unzipping dataset..."):
            download_database(os.getenv("Dataset"), "Flickr.zip")
            unzip_file("Flickr.zip")
            os.remove("Flickr.zip")

    if not os.path.exists(DATABASE_FOLDER):
        os.mkdir(DATABASE_FOLDER)

    with st.spinner("Downloading features..."):
        if not os.path.exists(INDEX_FILE):
            download_database(os.getenv("Index"), INDEX_FILE)
        if not os.path.exists(FEATURES_CSV):
            download_database(os.getenv("CSV"), FEATURES_CSV)
        if not os.path.exists(FEATURES_NPY):
            download_database(os.getenv("Features"), FEATURES_NPY)

    st.success("Finished getting data!")


def display_similar_images(similar_images, img_folder, label_folder):
    for idx, image_name in enumerate(similar_images, start=1):
        image_path = os.path.join(img_folder, image_name)
        label_path = os.path.join(
            label_folder, f'tags{image_name.replace(".jpg", ".txt").replace("im", "")}')

        similar_img = cv.imread(image_path)
        labels = load_label_from_path(label_path)
        
        st.write(f'Labels {idx}: {", ".join(labels)}')

        caption = f'Image {idx}: {image_name}'
        st.image(similar_img, channels="BGR",
                 caption=caption, use_column_width=True)


def main():
    setup_environment()

    with st.spinner("Getting data..."):
        try:
            download_and_prepare_data()
        except Exception as e:
            st.error('Failed to get database!')
            print(f'Error when getting database: {e}')
            return

    st.title("Social Image Retrieval System")
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        file_bytes = np.asarray(
            bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv.imdecode(file_bytes, 1)
    else:
        img = []

    top_k = st.sidebar.number_input(
        "Number of top K results", min_value=1, max_value=25000, value=5)

    if uploaded_file: 
        btn_searching = st.sidebar.button("Search")
        st.sidebar.image(img, channels="BGR",
                        caption='Uploaded Image.', use_column_width=True)
        
        if btn_searching and img is not None:
            with st.spinner("Processing image..."):
                start = time.time()
                query_img_array = preprocess_query_image(img)
                image_names, index, feature_extractor = load_data_and_create_model()
                
                if feature_extractor is None:
                    print('Feature extractor is None')
                    return

                query_feature = feature_extractor.predict(query_img_array)
                end = time.time()
                run_time = (end - start)
                st.write(f'Runtime: {run_time:.2f} secs. Image processed!')

            with st.spinner("Searching..."):
                start = time.time()
                similar_images = search_image(
                    query_feature[0], index, image_names, top_k=int(top_k))
                end = time.time()
                run_time = (end - start)

                st.write(f'Runtime: {run_time:.2f} secs. Similar images:')
                
            with st.spinner("Loading image..."):
                display_similar_images(
                    similar_images, IMG_FOLDER, LABEL_FOLDER)

if __name__ == "__main__":
    main()
