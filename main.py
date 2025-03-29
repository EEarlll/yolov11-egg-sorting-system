from ultralytics import YOLO
import cv2
import numpy as np
import joblib
from utils import get_features, calculateLength, calculatePerimeterArea

ref_x_min, ref_x_max = 297, 354  
real_length_of_reference = 3  
model = YOLO('yolo11n-seg.pt')
linear_model = joblib.load('linear_weight_model.pkl')

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        ret, img = cap.read()
        res = model.predict(img, stream=True, conf=0.2)

        for result in res:
            if result:
                for mask in result.masks.xy:
                    points = np.int32([mask])
                    pixel_perimeter = cv2.arcLength(points, True)
                    pixel_area = cv2.contourArea(points)
                    perimeter, area = calculatePerimeterArea(pixel_perimeter, pixel_area, ref_x_min, ref_x_max, real_length_of_reference)

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

                    # Display the prediction
                    print(f"Predicted Weight: {predicted_weight:.2f}g")

                    # Draw Bounding Box
                    boxText2 = f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}"
                    boxText = f"width: {width:.2f}, height: {height:.2f}"
                    cv2.putText(img, boxText, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.putText(img, boxText2, (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, f"Weight: {predicted_weight:.2f}g", (x1, y1 - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        cv2.imshow('Webcam', img)

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()