# model_inference.py
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

from .augment_imgs import crop_and_convert_image  # Import for preprocessing

IMG_SIZE = 224

def preprocess_image(image_pil):
    """Preprocess image to match training format"""
    image_pil = image_pil.resize((IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(image_pil)
    # Use the same preprocessing as in training (EfficientNet preprocess_input)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def load_model_and_prototypes(component_name):
    """Load the trained model and prototypes"""
    base_train_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'training_dataset'))
    component_path = os.path.join(base_train_path, component_name)
    
    # Load the full model
    model_path = os.path.join(component_path, 'fine_tuned_model.keras')
    encoder_path = os.path.join(component_path, 'fine_tuned_encoder.keras')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder file not found at {encoder_path}")
    
    # Load models
    model = load_model(model_path)
    encoder = load_model(encoder_path)
    
    # Load prototypes
    good_proto_path = os.path.join(component_path, 'good_proto.npy')
    bad_proto_path = os.path.join(component_path, 'bad_proto.npy')
    
    if not os.path.exists(good_proto_path) or not os.path.exists(bad_proto_path):
        raise FileNotFoundError("Prototype files not found")
    
    good_proto = np.load(good_proto_path)
    bad_proto = np.load(bad_proto_path)
    
    return model, encoder, good_proto, bad_proto

def predict_defect(image_path, component_name="discs"):
    """Predict if component is defective"""
    try:
        # Load model and prototypes
        model, encoder, good_proto, bad_proto = load_model_and_prototypes(component_name)
        
        # Use augment_imgs to crop and convert the image
        processed_img_pil = crop_and_convert_image(image_path, convert_to_grayscale=False)
        
        if processed_img_pil is None:
            raise ValueError(f"Failed to process image: {image_path}")
        
        # Convert to RGB if it's grayscale
        if processed_img_pil.mode == 'L':
            processed_img_pil = processed_img_pil.convert('RGB')
        
        # Preprocess image
        processed_img = preprocess_image(processed_img_pil)
        
        # Get prediction from the main model
        preds = model.predict(processed_img, verbose=0)
        defect_prob = preds[0][1]  # Probability of defective class
        is_defective = defect_prob >= 0.5
        
        return is_defective, defect_prob
        
    except Exception as e:
        raise Exception(f"Error in prediction: {str(e)}")

def crop_and_replace_test_images(test_dir):
    """Crop and convert test images to grayscale, replacing originals"""
    print(f"Processing images in {test_dir}")
    
    for filename in os.listdir(test_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(test_dir, filename)
            
            try:
                # Crop and convert to grayscale
                processed_img = crop_and_convert_image(image_path, convert_to_grayscale=True)
                
                if processed_img is not None:
                    # Save the processed image, replacing the original
                    processed_img.save(image_path)
                    print(f"✅ Processed and replaced: {filename}")
                else:
                    print(f"⚠️ Could not process: {filename}")
                    
            except Exception as e:
                print(f"❌ Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    component_name = "discs"
    TESTING_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'testing_dataset'))
    test_dir = os.path.join(TESTING_ROOT, component_name)
    print(f"Looking for test directory at: {test_dir}")

    if not os.path.exists(test_dir):
        print(f"❌ Test folder not found: {test_dir}")
        exit(1)

    # Step 1: Crop and replace test images with grayscale versions
    print("Step 1: Processing and replacing test images...")
    crop_and_replace_test_images(test_dir)
    
    # Step 2: Test the processed images
    print("\nStep 2: Testing processed images...")
    for filename in os.listdir(test_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(test_dir, filename)
            try:
                is_defective, confidence = predict_defect(image_path, component_name)
                print(f"🔍 {filename}: {'DEFECTIVE' if is_defective else 'NOT DEFECTIVE'}, confidence={confidence:.4f}")
            except Exception as e:
                print(f"❌ Error processing {filename}: {str(e)}")