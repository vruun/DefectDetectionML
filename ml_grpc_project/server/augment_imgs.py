# augment_imgs.py
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

def estimate_background_color(image, border_size=20):
    """
    Estimate background color by sampling border pixels.
    Enhanced to handle edge cases and provide better color estimation.
    """
    h, w = image.shape[:2]
    border_size = min(border_size, min(h, w) // 4)  # Ensure border size is reasonable
    
    # Sample from all four borders
    top = image[:border_size, :, :].reshape(-1, 3)
    bottom = image[-border_size:, :, :].reshape(-1, 3)
    left = image[:, :border_size, :].reshape(-1, 3)
    right = image[:, -border_size:, :].reshape(-1, 3)
    
    # Combine all border pixels
    border_pixels = np.concatenate((top, bottom, left, right), axis=0)
    
    # Use median instead of mean for more robust color estimation
    avg_color = np.median(border_pixels, axis=0)
    return (int(avg_color[0]), int(avg_color[1]), int(avg_color[2]))

def detect_disc_with_hough(gray_image):
    """
    Detect circular disc components using HoughCircles.
    Returns the best circle (x, y, radius) or None if no suitable circle found.
    """
    h, w = gray_image.shape
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray_image, (9, 9), 2)
    
    # Try different parameters for HoughCircles to catch various disc types
    param_sets = [
        {'dp': 1, 'min_dist': min(h, w) // 4, 'param1': 50, 'param2': 30},
        {'dp': 1, 'min_dist': min(h, w) // 6, 'param1': 40, 'param2': 25},
        {'dp': 2, 'min_dist': min(h, w) // 3, 'param1': 60, 'param2': 35}
    ]
    
    best_circle = None
    best_score = 0
    image_center = (w // 2, h // 2)
    
    for params in param_sets:
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=params['dp'],
            minDist=params['min_dist'],
            param1=params['param1'],
            param2=params['param2'],
            minRadius=min(h, w) // 8,  # Minimum reasonable radius
            maxRadius=min(h, w) // 2   # Maximum reasonable radius
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                # Score based on size and centrality
                center_distance = np.sqrt((x - image_center[0])**2 + (y - image_center[1])**2)
                # Prefer larger circles that are more centered
                score = r * 2 - center_distance
                
                # Ensure circle is reasonably within image bounds
                if (r > min(h, w) // 10 and 
                    x - r >= 0 and x + r <= w and 
                    y - r >= 0 and y + r <= h and
                    score > best_score):
                    best_score = score
                    best_circle = (x, y, r)
    
    return best_circle

def find_contour_based_disc(gray_image):
    """
    Find disc component using contour analysis with circularity filtering.
    Returns bounding box (x, y, w, h) or None.
    """
    h, w = gray_image.shape
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_image)
    
    # Multiple edge detection strategies
    # Strategy 1: Canny edges
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges1 = cv2.Canny(blurred, 30, 100)
    
    # Strategy 2: Adaptive threshold
    adaptive = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    edges2 = cv2.Canny(adaptive, 50, 150)
    
    # Combine edge maps
    edges = cv2.bitwise_or(edges1, edges2)
    
    # Morphological operations to connect broken edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    image_center = (w // 2, h // 2)
    best_contour = None
    best_score = 0
    
    for contour in contours:
        # Calculate contour properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if area < 0.01 * w * h or perimeter == 0:  # Too small
            continue
            
        # Calculate circularity (4π*area/perimeter²)
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Get bounding rectangle
        x, y, cw, ch = cv2.boundingRect(contour)
        
        # Calculate aspect ratio (should be close to 1 for squares/circles)
        aspect_ratio = min(cw, ch) / max(cw, ch)
        
        # Calculate center distance from image center
        cx, cy = x + cw // 2, y + ch // 2
        center_distance = np.sqrt((cx - image_center[0])**2 + (cy - image_center[1])**2)
        
        # Comprehensive scoring: circularity + aspect ratio + size + centrality
        size_score = min(area / (0.3 * w * h), 1.0)  # Normalize size score
        centrality_score = max(0, 1 - center_distance / (min(w, h) // 2))
        
        score = (circularity * 0.4 + 
                aspect_ratio * 0.3 + 
                size_score * 0.2 + 
                centrality_score * 0.1)
        
        # Additional filters
        if (circularity > 0.3 and  # Reasonably circular
            aspect_ratio > 0.6 and  # Not too elongated
            area > 0.02 * w * h and  # Minimum size
            area < 0.8 * w * h and   # Maximum size
            score > best_score):
            best_score = score
            best_contour = (x, y, cw, ch)
    
    return best_contour
'''
#too specific for disc shapes
def crop_component(image):
    """
    Detects disc components precisely by finding compact dark objects.
    Creates a tight square crop centered exactly on the disc.
    """
    original = image.copy()
    h, w = original.shape[:2]
    
    # Convert to grayscale for processing
    if len(original.shape) == 3:
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        gray = original.copy()
    
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Strategy 1: Multi-level thresholding to find the darkest objects
    # Calculate image statistics
    mean_brightness = np.mean(filtered)
    std_brightness = np.std(filtered)
    
    # Try multiple threshold levels to find compact dark objects
    thresholds = [
        max(0, int(mean_brightness - 1.2 * std_brightness)),  # Very dark
        max(0, int(mean_brightness - 0.8 * std_brightness)),  # Dark
        max(0, int(mean_brightness - 0.5 * std_brightness))   # Moderately dark
    ]
    
    best_detection = None
    best_compactness = 0
    
    for threshold in thresholds:
        # Create binary mask for dark regions
        _, dark_mask = cv2.threshold(filtered, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Clean up the mask - remove noise and fill holes
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel_small)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel_medium)
        
        # Find contours
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
            
        # Find the most compact dark object near center
        image_center = (w // 2, h // 2)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by size - should be significant but not huge
            min_area = 0.02 * w * h  # At least 2% of image
            max_area = 0.4 * w * h   # At most 40% of image
            
            if area < min_area or area > max_area:
                continue
            
            # Get bounding rectangle and center
            x, y, cw, ch = cv2.boundingRect(contour)
            cx, cy = x + cw // 2, y + ch // 2
            
            # Calculate compactness (area/perimeter ratio) - discs should be compact
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            compactness = (4 * np.pi * area) / (perimeter * perimeter)
            
            # Calculate aspect ratio - should be roughly square
            aspect_ratio = min(cw, ch) / max(cw, ch)
            
            # Distance from center
            center_distance = np.sqrt((cx - image_center[0])**2 + (cy - image_center[1])**2)
            max_distance = np.sqrt((w//2)**2 + (h//2)**2)
            centrality = 1 - (center_distance / max_distance)
            
            # Calculate average darkness within the contour
            mask_region = np.zeros_like(gray)
            cv2.fillPoly(mask_region, [contour], (255,))
            object_pixels = gray[mask_region == 255]
            if len(object_pixels) > 0:
                avg_darkness = np.mean(object_pixels)
                darkness_score = 1 - (avg_darkness / 255)
            else:
                darkness_score = 0
            
            # Combined score emphasizing compactness and darkness
            combined_score = (compactness * 0.35 + 
                            darkness_score * 0.3 + 
                            centrality * 0.2 + 
                            aspect_ratio * 0.15)
            
            # Only consider reasonably compact and dark objects
            if (compactness > 0.3 and  # Reasonably compact
                aspect_ratio > 0.6 and  # Not too elongated
                darkness_score > 0.3 and  # Sufficiently dark
                combined_score > best_compactness):
                
                best_compactness = combined_score
                best_detection = {
                    'contour': contour,
                    'bbox': (x, y, cw, ch),
                    'center': (cx, cy),
                    'darkness': darkness_score
                }
    
    if best_detection is not None:
        # Use the centroid of the contour for more precise centering
        contour = best_detection['contour']
        M = cv2.moments(contour)
        if M['m00'] != 0:
            true_cx = int(M['m10'] / M['m00'])
            true_cy = int(M['m01'] / M['m00'])
        else:
            true_cx, true_cy = best_detection['center']
        
        # Get bounding box but use true centroid for centering
        x, y, cw, ch = best_detection['bbox']
        
        # Create tight square crop around the true center
        # Use a more conservative padding based on the actual component size
        component_size = max(cw, ch)
        padding = max(8, int(component_size * 0.12))  # 12% padding
        
        crop_size = component_size + padding * 2
        
        # Center the crop on the true centroid
        crop_x = true_cx - crop_size // 2
        crop_y = true_cy - crop_size // 2
        
        # Ensure crop stays within image bounds
        crop_x = max(0, min(crop_x, w - crop_size))
        crop_y = max(0, min(crop_y, h - crop_size))
        
        # Make sure we have valid dimensions
        if (crop_x >= 0 and crop_y >= 0 and 
            crop_x + crop_size <= w and crop_y + crop_size <= h and
            crop_size > 0):
            cropped = original[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
            return cropped
    
    # Fallback: Try edge-based detection for internal disc features
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(filtered)
    
    # Use multiple edge detection parameters
    edges1 = cv2.Canny(enhanced, 40, 120)
    edges2 = cv2.Canny(enhanced, 20, 80)
    edges = cv2.bitwise_or(edges1, edges2)
    
    # Find contours from edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the most centered contour with reasonable size
        image_center = (w // 2, h // 2)
        best_contour = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 0.015 * w * h or area > 0.5 * w * h:
                continue
                
            x, y, cw, ch = cv2.boundingRect(contour)
            cx, cy = x + cw // 2, y + ch // 2
            
            # Prefer objects closer to center with reasonable aspect ratio
            distance = np.sqrt((cx - image_center[0])**2 + (cy - image_center[1])**2)
            aspect_ratio = min(cw, ch) / max(cw, ch)
            
            score = (1.0 / (1 + distance)) * aspect_ratio * (area / (0.1 * w * h))
            
            if score > best_score:
                best_score = score
                best_contour = contour
        
        if best_contour is not None:
            x, y, cw, ch = cv2.boundingRect(best_contour)
            
            # Calculate true center using moments
            M = cv2.moments(best_contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = x + cw // 2, y + ch // 2
            
            # Create square crop
            crop_size = max(cw, ch) + max(15, min(cw, ch) // 8)
            
            crop_x = max(0, min(cx - crop_size // 2, w - crop_size))
            crop_y = max(0, min(cy - crop_size // 2, h - crop_size))
            
            if crop_x >= 0 and crop_y >= 0 and crop_size > 0:
                cropped = original[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
                return cropped
    
    # Final fallback: center crop
    crop_size = min(h, w) // 2
    x = (w - crop_size) // 2
    y = (h - crop_size) // 2
    return original[y:y+crop_size, x:x+crop_size]
'''
#absolutely non dependent on shape of the component but struggles with precropped images
def crop_component(image):
    """Generalized component cropping for arbitrary shapes"""
    original = image.copy()
    h, w = original.shape[:2]
    
    # Convert to grayscale with CLAHE normalization
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) if len(original.shape) == 3 else original.copy()
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Multi-stage thresholding
    methods = []
    
    # Adaptive threshold variants
    for block_size in [11, 21, 31]:
        thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, block_size, 10)
        methods.append(thresh)
    
    # Global threshold variants
    for thresh_val in [120, 160, 200]:
        _, thresh = cv2.threshold(enhanced, thresh_val, 255, cv2.THRESH_BINARY_INV)
        methods.append(thresh)
    
    best_contour = None
    best_area = 0
    center_x, center_y = w//2, h//2
    
    # Contour analysis with shape-agnostic criteria
    for thresh in methods:
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, 
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), 
                                 iterations=2)
        
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:  # Minimum viable area
                continue
                
            # Position-based scoring
            x, y, cw, ch = cv2.boundingRect(contour)
            contour_center = (x + cw//2, y + ch//2)
            dist_from_center = np.hypot(contour_center[0]-center_x, contour_center[1]-center_y)
            
            # Shape-agnostic selection criteria:
            # 1. Significant but non-dominant size (1-30% of image area)
            # 2. Proximity to image center
            # 3. Clear separation from image edges
            area_ratio = area / (h*w)
            edge_margin = min(x, y, w-(x+cw), h-(y+ch))
            
            if (0.01 < area_ratio < 0.3 and 
                edge_margin > 20 and 
                (area > best_area or dist_from_center < 100)):
                best_contour = contour
                best_area = area
    
    # Fallback to edge-based detection
    if best_contour is None:
        edges = cv2.Canny(enhanced, 50, 150)
        edges = cv2.dilate(edges, None, iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            best_contour = max(contours, key=cv2.contourArea)
    
    # Final crop logic
    if best_contour is not None:
        x, y, cw, ch = cv2.boundingRect(best_contour)
        pad = int(max(cw, ch) * 0.15)  # Dynamic padding
        
        # Square crop calculation
        side = max(cw, ch) + 2*pad
        cx, cy = x + cw//2, y + ch//2
        
        start_x = max(0, cx - side//2)
        start_y = max(0, cy - side//2)
        end_x = min(w, start_x + side)
        end_y = min(h, start_y + side)
        
        # Boundary checks
        start_x = end_x - side if start_x < 0 else start_x
        start_y = end_y - side if start_y < 0 else start_y
        
        return original[start_y:end_y, start_x:end_x]
    
    # Fallback to center crop
    size = min(h, w) // 2
    return original[h//2-size:h//2+size, w//2-size:w//2+size]

def rotate_image(image, angle, fill_color=(128, 128, 128)):
    """
    Rotate image by angle degrees around center and fill empty space with fill_color.
    Enhanced to handle both grayscale and color images properly.
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Get rotation matrix
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Handle different image types
    if len(image.shape) == 3:
        # Color image
        rotated = cv2.warpAffine(image, rot_matrix, (w, h), borderValue=fill_color, borderMode=cv2.BORDER_CONSTANT)
    else:
        # Grayscale image
        if isinstance(fill_color, (tuple, list)):
            fill_value = int(np.mean(fill_color))
        else:
            fill_value = fill_color
        rotated = cv2.warpAffine(image, rot_matrix, (w, h), borderValue=fill_value, borderMode=cv2.BORDER_CONSTANT)
    
    return rotated

def crop_and_convert_image(image_path, convert_to_grayscale=True):
    """
    Process a single image: crop component and optionally convert to grayscale.
    Used for inference where no augmentation is needed.
    Returns the processed image as PIL Image.
    """
    try:
        # Load image
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        
        # Crop component using enhanced detection
        cropped = crop_component(image)
        
        if cropped is None or cropped.size == 0:
            return None
        
        # Convert to grayscale if requested
        if convert_to_grayscale:
            if len(cropped.shape) == 3:
                cropped = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
                return Image.fromarray(cropped, 'L')
            else:
                return Image.fromarray(cropped, 'L')
        else:
            return Image.fromarray(cropped, 'RGB')
            
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def augment_images_in_folder(folder_path, convert_to_grayscale=False):
    """
    Process all images in folder with enhanced disc detection:
    1) Detect and crop disc components using multiple strategies
    2) Convert to grayscale if requested
    3) Augment by rotating multiple degrees with background filling
    4) Save rotated images
    5) Delete original images after augmentation
    """
    if not os.path.exists(folder_path):
        print(f"❌ Folder does not exist: {folder_path}")
        return
    
    # Get list of image files
    image_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if not image_files:
        print(f"⚠️ No image files found in {folder_path}")
        return
    
    print(f"📁 Processing {len(image_files)} images in: {folder_path}")
    
    successful_crops = 0
    failed_crops = 0
    
    for filename in tqdm(image_files, desc="Processing images"):
        file_path = os.path.join(folder_path, filename)
        
        try:
            # Load image
            image = Image.open(file_path).convert("RGB")
            image = np.array(image)
            
            # Crop component using enhanced detection
            cropped = crop_component(image)
            
            if cropped is None or cropped.size == 0:
                print(f"❌ Skipping {filename} — Cropping failed.")
                failed_crops += 1
                continue
            
            successful_crops += 1
            
            # Convert to grayscale if requested
            if convert_to_grayscale:
                if len(cropped.shape) == 3:
                    cropped = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
            
            # Estimate background color for rotation fill
            background_color = estimate_background_color(image)
            if convert_to_grayscale:
                background_color = int(np.mean(background_color))
            
            # Save the 0-degree cropped image
            if len(cropped.shape) == 2:  # Grayscale
                cropped_pil = Image.fromarray(cropped, 'L')
            else:  # Color
                cropped_pil = Image.fromarray(cropped, 'RGB')

            name, ext = os.path.splitext(filename)
            cropped_filename = f"cropped_0_{name}{ext}"
            cropped_path = os.path.join(folder_path, cropped_filename)
            cropped_pil.save(cropped_path)
            
            # Generate rotated versions
            rotation_angles = [-30, -15, 15, 30]
            for angle in rotation_angles:
                rotated = rotate_image(cropped, angle, fill_color=background_color)
                
                # Convert back to PIL and save
                if len(rotated.shape) == 2:  # Grayscale
                    rotated_pil = Image.fromarray(rotated, 'L')
                else:  # Color
                    rotated_pil = Image.fromarray(rotated, 'RGB')
                
                # Generate filename for rotated image
                rotated_filename = f"rotated_{angle}_{name}{ext}"
                rotated_path = os.path.join(folder_path, rotated_filename)
                
                rotated_pil.save(rotated_path)
            
            # Remove original image after successful processing
            os.remove(file_path)
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {str(e)}")
            failed_crops += 1
            continue
    
    print(f"✅ Processing complete!")
    print(f"   Successful crops: {successful_crops}")
    print(f"   Failed crops: {failed_crops}")
    print(f"   Generated {successful_crops * 5} total images (1 cropped + 4 rotated per original)")

if __name__ == "__main__":
    # Replace these paths manually when testing different folders
    test_folders = [
        r"D:\mechmet intern project\ml_grpc_project\training_dataset\discs\good",
        r"D:\mechmet intern project\ml_grpc_project\training_dataset\discs\bad"
    ]

    for folder in test_folders:
        print(f"Processing folder: {folder}")
        augment_images_in_folder(folder, convert_to_grayscale=True)