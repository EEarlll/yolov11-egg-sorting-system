from ultralytics import YOLO
import cv2
import numpy as np
import joblib
import argparse
import time
from utils import get_features, calculateLength, calculatePerimeterArea

def predict_egg_weight_instant_capture(output_path=None, camera_id=0):
    # Load models
    model = YOLO('yolo11n-seg.pt')
    linear_model = joblib.load('linear_weight_model.pkl')
    
    # Reference measurements
    ref_x_min, ref_x_max = 297, 354  
    real_length_of_reference = 3
    
    # Initialize webcam
    cap = cv2.VideoCapture(camera_id)
    cap.set(3, 640)
    cap.set(4, 480)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Small delay to allow camera to initialize
    time.sleep(0.3)
    
    # Capture a single frame instantly
    ret, img = cap.read()
    
    # Release webcam immediately
    cap.release()
    
    if not ret:
        print("Error: Failed to capture image")
        return
    
    # Run prediction
    res = model.predict(img, conf=0.2)
    
    predicted_weight = None
    
    for result in res:
        if result and len(result.masks) > 0:
            # Get mask
            for mask in result.masks.xy:
                points = np.int32([mask])
                pixel_perimeter = cv2.arcLength(points, True)
                pixel_area = cv2.contourArea(points)
                perimeter, area = calculatePerimeterArea(pixel_perimeter, pixel_area, ref_x_min, ref_x_max, real_length_of_reference)
            
            # Get bounding box
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                
                pixel_width = x2 - x1
                pixel_height = y2 - y1
                width, height = calculateLength(pixel_width, pixel_height, ref_x_min, ref_x_max, real_length_of_reference)
                
                shape_index = (width / height) * 100 if height > 0 else 0
                compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
                
                # Compute additional egg features
                egg_features = get_features(width, height)
                
                # Prepare input features for the model
                feature_vector = {
                    "perimeter": perimeter,
                    "area": area,
                    "height": height,
                    "width": width,
                    "shape_index": shape_index,
                    "compactness": compactness,
                    "longer_semi_major": egg_features["longer_semi_major"],
                    "shorter_semi_major": egg_features["shorter_semi_major"],
                    "semi_major_axis_ratio": egg_features["semi_major_axis_ratio"],
                    "d1": egg_features["d1"],
                    "d2": egg_features["d2"],
                    "d3": egg_features["d3"],
                    "d4": egg_features["d4"],
                }
                
                # Convert to NumPy array and reshape for prediction
                input_data = np.array(list(feature_vector.values())).reshape(1, -1)
                
                # Predict using the trained model
                predicted_weight = linear_model.predict(input_data)[0]
                
                # Draw on image
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, f"Weight: {predicted_weight:.2f}g", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Draw contour
                cv2.drawContours(img, points, -1, (0, 255, 0), 2)
    
    # Display and save results
    if predicted_weight is not None:
        print(f"Predicted Egg Weight: {predicted_weight:.2f}g")
        
        # Show the result
        cv2.imshow('Egg Weight Prediction', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        if output_path:
            cv2.imwrite(output_path, img)
            print(f"Annotated image saved to {output_path}")
    else:
        print("No egg detected in the image")
        cv2.imshow('No Egg Detected', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return predicted_weight

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict egg weight from instant webcam capture')
    parser.add_argument('--output', type=str, help='Path to save annotated output image (optional)')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    
    args = parser.parse_args()
    
    predict_egg_weight_instant_capture(args.output, args.camera)