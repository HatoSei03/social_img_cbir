# Social Image Retrieval
In this project, we utilize the Inception ResNet V2 model to extract features from social images and index these features for efficient retrieval. 

## Dataset
The MIRFLICKR25000 dataset is a collection of 25,000 images sourced from the photo-sharing platform Flickr. It was created to support research in multimedia information retrieval, with a focus on image retrieval, annotation, and related tasks. The dataset features a diverse range of images covering various subjects, scenes, and concepts.

## Set up the project
1. Clone the repository:

   ```bash
   git clone {{repository_url}}
   ```

2. Change to the project directory:

   ```bash
   cd {{project_directory}}
   ```

3. Create a `.env` file with the following content:

	```
	Dataset= "https://drive.google.com/uc?export=download&id=1lYc4Hg80T-XKBHOn4Cge5DHKrJO2avvy"
	Index= "https://drive.google.com/uc?export=download&id=1vQ8eQNxTJVmkZXUiobqCbIjtl93xRH4Q"
	CSV= "https://drive.google.com/uc?export=download&id=1l5_X9pYsgEJWMRSiXLoFmepY7sbUuzKq"
	Features= "https://drive.google.com/uc?export=download&id=1BKTV-JNwIN7PwFIijTh8C4Fu5xPyJt3R"
	Annotations="https://drive.google.com/uc?export=download&id=1dcVjpDwun1FFeqtRaitnAIYsupsJ27AW"
	Color="https://drive.google.com/uc?export=download&id=1biHqKCnTBR6WOy8hGSOyHqDSY-DQ5vLS"
	```

4. Install the Python dependencies as in [requirements.txt](requirements.txt) file by running the code
    ```bash
    pip install -r requirements.txt
    ```

## Feature Extraction
The features are extracted using Inception Resnet V2 model. This model is a powerful convolutional neural network architecture that combines the strengths of Inception networks and Residual networks, providing a deep and efficient model for capturing rich features from images. The features can be extracted by running the [extract_features](extract_features.py) file. 

To make it simple, the features are extracted beforehand and will be automatically downloaded when the program first run together with the dataset.

## Program running
Start the app server using
```bash
streamlit run main.py
```
Open your browser and navigate to `http://localhost:8501/` to access the Web Application.

When start for the first time, it will take sometime to download the dataset as well as the extracted features and index.
