import os
import csv
import time
import numpy as np
import streamlit as st
import cv2 as cv
from dotenv import load_dotenv
import threading


from processing import (
    set_random_seed, define_model, create_annoy_index, AnnoyIndex,
    preprocess_query_image, search_image
)
from get_data import download_database, unzip_file
from label_handle import read_upload, load_label_from_path

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
    

def load_data_and_create_model():
    print("START load data and create model")
    load_time = time.time()
    try:
        global image_names, features, index, feature_extractor
        image_names, features, index = load_data()
        feature_extractor = define_model()
        if image_names is None or features is None or index is None:
            st.error('Failed to load features and index')
            return
    except Exception as e:
        print(f'Error when loading features and index: {e.__context__}')
    finally:
        load_time = time.time() - load_time
        print(f'Load data and create model time: {load_time:.2f} secs')

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


def load_data():
    try:
        with open(FEATURES_CSV, 'r') as f:
            image_names = [row[0] for row in csv.reader(f)]

        features = np.load(FEATURES_NPY)

        if not os.path.exists(INDEX_FILE):
            index = create_annoy_index(features)
            index.save(INDEX_FILE)

        index = AnnoyIndex(features.shape[1], 'euclidean')
        index.load(INDEX_FILE)
        return image_names, features, index
    except Exception as e:
        st.error('Failed to load features and index')
        print(f'Error when loading features and index: {e}')
        return None, None, None


def display_similar_images(similar_images, query_labels, img_folder, label_folder):
    query_labels_list = query_labels.split(", ")
    for idx, image_name in enumerate(similar_images, start=1):
        image_path = os.path.join(img_folder, image_name)
        label_path = os.path.join(
            label_folder, f'tags{image_name.replace(".jpg", ".txt").replace("im", "")}')

        similar_img = cv.imread(image_path)
        labels = load_label_from_path(label_path)
        matching_labels = [
            label for label in labels if label in query_labels_list]

        accuracy_str = f'Image {idx} - '
        accuracy_str += f'Accuracy {len(matching_labels)}/'
        accuracy_str += str(len(query_labels_list))

        st.write(accuracy_str)
        st.write(f'Labels: {", ".join(labels)}')
        st.write(f'Matching Labels: {", ".join(matching_labels)}')

        caption = f'Image {idx}: {image_name}'
        st.image(similar_img, channels="BGR",
                 caption=caption, use_column_width=True)


def main():
    setup_environment()

    # Create a thread to run load_data and create the model
    data_thread = threading.Thread(target=load_data_and_create_model)
    data_thread.start()

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
    label_file = st.sidebar.file_uploader(
        "Choose your label file...", type=["txt"])
    if uploaded_file:
        file_bytes = np.asarray(
            bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv.imdecode(file_bytes, 1)
    else:
        img = []

    query_labels = "No label"
    if label_file:
        query_labels = ", ".join(read_upload(
            label_file)) if label_file else "No label"
    top_k = st.sidebar.number_input(
        "Number of top K results", min_value=1, max_value=100, value=5)
    st.sidebar.write(f'Query labels: {query_labels}')
    print("FINISH load img and label")
    if uploaded_file and label_file:
        btn_searching = st.sidebar.button("Search")
    if uploaded_file:
        st.sidebar.image(img, channels="BGR",
                        caption='Uploaded Image.', use_column_width=True)
    if uploaded_file and label_file:
        if btn_searching and img is not None:
            with st.spinner("Processing image..."):
                data_thread.join()
                query_img_array = preprocess_query_image(img)
                query_feature = feature_extractor.predict(query_img_array)

            with st.spinner("Searching..."):
                start = time.time()
                similar_images = search_image(
                    query_feature[0], index, image_names, top_k=int(top_k))
                end = time.time()
                run_time = (end - start)

                st.write(f'Runtime: {run_time:.2f} secs. Similar images:')
                display_similar_images(
                    similar_images, query_labels, IMG_FOLDER, LABEL_FOLDER)


if __name__ == "__main__":
    main()
