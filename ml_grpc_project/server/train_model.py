# train_model.py
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam

from .augment_imgs import augment_images_in_folder  # Import for augmentation

IMG_SIZE = 224

def load_images_from_folder(folder_path, label):
    images, labels = [], []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
            img_array = preprocess_input(np.array(img))
            images.append(img_array)
            labels.append(label)
    return images, labels

def train_component_model(component_name):
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "training_dataset", component_name))
        train_dir = base_dir

        good_dir = os.path.join(train_dir, "good")
        bad_dir = os.path.join(train_dir, "bad")

        if len([f for f in os.listdir(good_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) < 10:
            print(f"Augmenting 'good' images for component '{component_name}'...")
            augment_images_in_folder(good_dir, convert_to_grayscale=False)

        if len([f for f in os.listdir(bad_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) < 10:
            print(f"Augmenting 'bad' images for component '{component_name}'...")
            augment_images_in_folder(bad_dir, convert_to_grayscale=False)

        good_images, good_labels = load_images_from_folder(good_dir, 0)
        bad_images, bad_labels = load_images_from_folder(bad_dir, 1)

        if len(good_images) == 0 or len(bad_images) == 0:
            return False, f"No images found in 'good' or 'bad' folders for component '{component_name}'."

        X = np.array(good_images + bad_images)
        y = np.array(good_labels + bad_labels)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        base_model = EfficientNetB0(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
        for layer in base_model.layers:
            layer.trainable = False
        for layer in base_model.layers[-20:]:
            layer.trainable = True

        inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        x = base_model(inputs, training=True)
        x = GlobalAveragePooling2D()(x)
        outputs = Dense(2, activation='softmax')(x)

        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=8, epochs=10, verbose=0)

        model.save(os.path.join(base_dir, "fine_tuned_model.keras"))
        encoder = Model(inputs, x)
        encoder.save(os.path.join(base_dir, "fine_tuned_encoder.keras"))

        def compute_prototype(encoder, image_folder):
            images = []
            for fname in os.listdir(image_folder):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(image_folder, fname)
                    img = Image.open(path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
                    images.append(np.array(img))
            images = np.array(images)
            embeddings = encoder.predict(images, verbose=0)
            return np.mean(embeddings, axis=0)

        good_proto = compute_prototype(encoder, good_dir)
        bad_proto = compute_prototype(encoder, bad_dir)

        np.save(os.path.join(base_dir, "good_proto.npy"), good_proto)
        np.save(os.path.join(base_dir, "bad_proto.npy"), bad_proto)

        return True, f"Training completed successfully for component '{component_name}'."

    except Exception as e:
        return False, f"Training failed: {str(e)}"

if __name__ == "__main__":
    component = "pens"  # Change this to the component you want to test
    success, message = train_component_model(component)
    print(message)