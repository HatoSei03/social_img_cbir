import os
import csv
import time
import numpy as np
import streamlit as st
import cv2 as cv
from dotenv import load_dotenv
from annotated_text import annotated_text, parameters
from streamlit_tags import st_tags
import pandas as pd

from processing import (
    set_random_seed, define_model, create_annoy_index, AnnoyIndex,
    preprocess_query_image, search_image
)
from get_data import download_database, unzip_file
from label_handle import load_annotation, load_plain_annotation

IMG_FOLDER = "mirflickr"
DATABASE_FOLDER = "database"
FEATURES_CSV = os.path.join(DATABASE_FOLDER, "features.csv")
FEATURES_NPY = os.path.join(DATABASE_FOLDER, "features.npy")
INDEX_FILE = os.path.join(DATABASE_FOLDER, "index.ann")
ANNOTATIONS = os.path.join(DATABASE_FOLDER, "annotation.csv")


parameters.SHOW_LABEL_SEPARATOR = True
st.set_page_config(layout="wide")


def setup_environment():
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    set_random_seed(50)
    load_dotenv()
    pd.options.display.float_format = "{:,.4f}".format


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


@st.cache_data
def convert_result_to_csv(imgs_name, input_labels):
    isDemo = st.session_state.isDemo
    result = '"rank","index","po_annotation","re_annotation","match_labels"\n'
    for idx, img in enumerate(imgs_name, start=1):
        img_id = get_digits(img)
        po, re = load_plain_annotation(
            img_id, ANNOTATIONS)
        match_labels = "; ".join(
            [label for label in input_labels if label in re]) if isDemo else ""
        result += f"{idx},\"{img}\",\"{'; '.join(po)}\",\"{'; '.join(re)}\",\"{match_labels}\"\n"
    return result


@st.cache_data
def suggestion_label(imgs_list):
    result = []
    for img in imgs_list:
        img_id = get_digits(img)
        re = load_plain_annotation(
            img_id, ANNOTATIONS)[1]
        for label in re:
            if label == "":
                continue
            if label not in result:
                result.append(label)
    return sorted(result)


@st.cache_data(show_spinner=False)
def count_correct(img_result, input_label):
    correct_label_img = 0
    for img in img_result:
        img_id = get_digits(img)
        re = load_plain_annotation(
            img_id, ANNOTATIONS)[1]
        correct_label_img += 1
        for label in input_label:
            if label not in re:
                correct_label_img -= 1
                break
    return correct_label_img


@st.cache_data(show_spinner=False)
def count_correct_in_db(input_label):
    correct_label_img = 0
    RE_COL = 3
    with open(ANNOTATIONS, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            re_labels = row[RE_COL].split(';')
            correct_label_img += 1
            for label in input_label:
                if label not in re_labels:
                    correct_label_img -= 1
                    break
    return correct_label_img


@st.cache_data(show_spinner=False)
def calc_rr(img_result, input_label):
    for idx, img in enumerate(img_result):
        img_id = get_digits(img)
        re = load_plain_annotation(
            img_id, ANNOTATIONS)[1]
        is_correct = True
        for label in input_label:
            if label not in re:
                is_correct = False
                break
        if is_correct:
            return 1/(idx+1)
    return 0


@st.cache_data(show_spinner=False)
def calc_apk(img_result, input_label):
    pk_values = []
    k = len(img_result)
    correct_counter = 0

    for i in range(1, k+1):
        img_id = get_digits(img_result[i-1])
        re = load_plain_annotation(
            img_id, ANNOTATIONS)[1]
        is_correct = True
        for label in input_label:
            if label not in re:
                is_correct = False
                break
        if is_correct:
            correct_counter += 1
            pk_values.append(correct_counter/i)
    return np.mean(pk_values) if len(pk_values) > 0 else 0


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

    with st.spinner("Downloading features and related file..."):
        if not os.path.exists(INDEX_FILE):
            download_database(os.getenv("Index"), INDEX_FILE)
        if not os.path.exists(FEATURES_CSV):
            download_database(os.getenv("CSV"), FEATURES_CSV)
        if not os.path.exists(FEATURES_NPY):
            download_database(os.getenv("Features"), FEATURES_NPY)
        if not os.path.exists(ANNOTATIONS):
            download_database(os.getenv("Annotations"), ANNOTATIONS)

    st.success("Finished getting data!")

# get only digits from string


def get_digits(text):
    return int(''.join(filter(str.isdigit, text)))


def display_similar_images(similar_images, input_label, img_folder, row_size, max_ppage, page_indx):
    grid = st.columns(row_size)
    start_idx = (page_indx - 1) * max_ppage + 1
    end_idx = min(page_indx * max_ppage, len(similar_images))
    for idx, image_name in enumerate(similar_images, start=1):
        if idx < start_idx or idx > end_idx:
            continue
        with grid[(idx-start_idx) % row_size]:
            image_path = os.path.join(img_folder, image_name)
            img_id = get_digits(image_name)
            po_annotation, re_annotation = load_annotation(img_id, ANNOTATIONS)
            similar_img = cv.imread(image_path)
            caption = f'Image {idx}: {image_name}'
            st.image(similar_img, channels="BGR",
                     caption=caption, use_column_width=True)

            re_label = load_plain_annotation(img_id, ANNOTATIONS)[1]
            match_labels_idx = [idx for idx, label in enumerate(
                re_label) if label in input_label]
            match_annotations = [re_annotation[idx]
                                 for idx in match_labels_idx]
            annotated_text(
                [
                    f"{len(po_annotation)} potential: ",
                    po_annotation,
                ]
            )
            annotated_text(
                [
                    f"{len(re_annotation)} relevant: ",
                    re_annotation
                ]
            )
            isDemo = st.session_state.isDemo
            if isDemo:
                annotated_text(
                    [
                        f"{len(match_annotations)} match: ",
                        match_annotations
                    ]
                )

        if (idx-start_idx+1) % row_size == 0:
            grid = st.columns(row_size)


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
    isDemo = st.sidebar.toggle("Experiment mode", False)
    st.session_state.isDemo = isDemo
    ignore_first = st.sidebar.toggle("Ignore first image", False)
    st.session_state.ignore_first = ignore_first
    row_size = st.sidebar.select_slider("Row size:", range(1, 11), value=3)
    max_ppage = st.sidebar.number_input(
        "Max image per page", min_value=1, max_value=50, value=20)
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        file_bytes = np.asarray(
            bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv.imdecode(file_bytes, 1)
    else:
        img = []

    top_k = st.sidebar.number_input(
        "Number of top K results", min_value=1, max_value=25000, value=1000)

    if uploaded_file:
        btn_searching = st.sidebar.button("Search")
        st.sidebar.image(img, channels="BGR",
                         caption='Uploaded Image.', use_column_width=True)

        if btn_searching and img is not None:
            load_data_and_create_model()
            with st.spinner("Processing image..."):
                start = time.time()
                query_img_array = preprocess_query_image(img)
                image_names, index, feature_extractor = None, None, None
                query_feature = None
                success = False
                while not success:
                    try:
                        start = time.time()
                        image_names, index, feature_extractor = load_data_and_create_model()
                        query_feature = feature_extractor.predict(
                            query_img_array)
                        success = True
                    except:
                        load_data.clear()
                        load_data_and_create_model.clear()

                end = time.time()
                run_time = (end - start)
                st.session_state.processing_time = run_time

            with st.spinner("Searching..."):
                start = time.time()
                similar_images = search_image(
                    query_feature[0], index, image_names, top_k=int(top_k))
                end = time.time()
                run_time = (end - start)
                st.session_state.similar_images = similar_images
                st.session_state.run_time = run_time
    if "similar_images" in st.session_state:
        ignore_first = st.session_state.ignore_first
        data = st.session_state.similar_images[ignore_first:]
        button_grid = st.columns(3)
        isDemo = st.session_state.isDemo
        
        if isDemo:
            input_labels = st_tags(
                label='Enter labels for input image:',
                text='Press enter to add more',
                value=suggestion_label(data),
                suggestions=suggestion_label(data),
                maxtags=-1,
                key="input_labels"
            )
            input_labels = [label.lower().strip() for label in input_labels]
        else:
            input_labels = []

        csv_data = convert_result_to_csv(data, input_labels)
        button_grid[0].download_button(
            label="Download result as CSV",
            data=csv_data,
            file_name=f"{uploaded_file.name.replace('.','_')}_result.csv",
            mime="text/csv",
        )

        page_idx = button_grid[2].selectbox(
            "Page",
            [i for i in range(1, len(data)//max_ppage + 2)],
            label_visibility="collapsed",
            format_func=lambda x: "Page " + str(x)
        )
        correct_img = count_correct(data, input_labels)
        correct_db = count_correct_in_db(input_labels)

        process_time = round(st.session_state.processing_time, 3)
        search_time = round(st.session_state.run_time, 3)
        precision = round(correct_img/len(data), 3) if len(data) > 0 else 0
        recall = round(correct_img/correct_db, 3) if correct_db > 0 else 0
        rr = round(calc_rr(data, input_labels), 3)
        apk = round(calc_apk(data, input_labels), 3) if isDemo else -1

        if isDemo:
            measurements = pd.DataFrame(
                [
                    ["k", top_k],
                    ["Input labels", "; ".join(input_labels)],
                    ["Processing time (secs)",  process_time],
                    ["Searching time (secs)",  search_time],
                    ["Correct images", correct_img],
                    ["Correct images in DB", correct_db],
                    ["Precision", precision],
                    ["Recall", recall],
                    ["Reciprocal Rank (RR)", rr],
                    ["Average Precision@K (AP@K)", apk]],
                columns=["Measurements", "Values"]
            )
        else:
            measurements = pd.DataFrame(
                [
                    ["k", top_k],
                    ["Processing time (secs)",  process_time],
                    ["Searching time (secs)",  search_time],
                ],
                columns=["Measurements", "Values"]
            )
        measurements.set_index("Measurements", inplace=True)
        measurements.round(4)
        st.sidebar.table(measurements)
        st.sidebar.download_button(
            label="Save measurements as CSV",
            data=measurements.to_csv().encode("utf-8"),
            file_name=f"{uploaded_file.name.replace('.','_')}_measurements.csv",
            mime="text/csv",
        )
        with st.spinner("Loading image..."):
            display_similar_images(
                data, input_labels, IMG_FOLDER, row_size, max_ppage, page_idx)


if __name__ == "__main__":
    main()
