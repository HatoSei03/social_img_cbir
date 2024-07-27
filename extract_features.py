import numpy as np
import csv
from keras.preprocessing.image import ImageDataGenerator

from processing import define_model, set_random_seed

set_random_seed(50)

# Create a feature extractor model
feature_extractor = define_model()

datagen = ImageDataGenerator(rescale=1./255)
generator = datagen.flow_from_directory("Dataset/Flickr/flickr30k_images/flickr30k_images",
                                        target_size=(500, 500),
                                        batch_size=32,
                                        class_mode=None,
                                        shuffle=False)

image_names = [filename.split("/")[-1] for filename in generator.filenames]

features = feature_extractor.predict(generator, steps=len(generator), verbose=1)

np.save("database/features.npy", features)

with open("database/features.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    # Writing each item in a new row
    for item in image_names:
        writer.writerow([item])

