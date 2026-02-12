import cv2
import numpy as np

def get_dominant_color(roi):
    """Extract the dominant color from the region of interest"""
    pixels = roi.reshape(-1, 3)
    pixels = np.float32(pixels)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 3
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    dominant_color = centers[np.argmax(np.bincount(labels.flatten()))]
    return dominant_color

def classify_color(bgr_color):
    """Classify color based on BGR values"""
    b, g, r = bgr_color
    
    hsv = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = hsv
    
    if s < 30:
        if v < 50:
            return "Black"
        elif v > 200:
            return "White"
        else:
            return "Gray"
    
    if h < 10 or h > 170:
        return "Red"
    elif 10 <= h < 25:
        return "Orange"
    elif 25 <= h < 35:
        return "Yellow"
    elif 35 <= h < 85:
        return "Green"
    elif 85 <= h < 130:
        return "Blue"
    elif 130 <= h < 170:
        return "Purple"
    else:
        return "Unknown"

def classify_shape(contour):
    """Classify shape including oval, round, heart, triangular, horseshoe"""
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    if perimeter == 0:
        return "Unknown"
    
    # Calculate circularity
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    # Calculate convexity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    # Get aspect ratio
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 0
    
    # Triangular shape
    if len(approx) == 3:
        return "Triangular"
    
    # Round/Circular
    if circularity > 0.85 and solidity > 0.9:
        return "Round"
    
    # Oval
    if 0.7 < circularity < 0.85 and solidity > 0.85:
        return "Oval"
    
    # Heart shape - typically has two humps on top
    if solidity < 0.85 and circularity > 0.5:
        # Check for concavity (heart has indentation at top)
        defects = cv2.convexityDefects(contour, cv2.convexHull(contour, returnPoints=False))
        if defects is not None and len(defects) > 2:
            return "Heart"
    
    # Horseshoe/U-shape - open curved shape
    if solidity < 0.75 and 0.4 < circularity < 0.7:
        return "Horseshoe"
    
    # Polygon shapes
    if len(approx) == 4:
        if 0.95 <= aspect_ratio <= 1.05:
            return "Square"
        else:
            return "Rectangle"
    elif len(approx) == 5:
        return "Pentagon"
    elif len(approx) == 6:
        return "Hexagon"
    elif 7 <= len(approx) <= 10:
        return "Polygon"
    
    return "Irregular"

def detect_crab_from_camera():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert to grayscale for better edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Use adaptive thresholding to separate crab from background
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        crab_detected = False
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area - adjust this based on your printed image size
            if 5000 < area < 100000:  # Filter out very small and very large objects
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Additional filtering: aspect ratio to avoid detecting elongated objects like fingers
                aspect_ratio = float(w) / h if h > 0 else 0
                if 0.3 < aspect_ratio < 3.0:  # Reasonable aspect ratio for crab shape
                    
                    # Draw rectangle around detected crab
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Extract ROI for color detection
                    crab_roi = frame[y:y+h, x:x+w]
                    
                    if crab_roi.size > 0:
                        # Get dominant color
                        dominant_color = get_dominant_color(crab_roi)
                        color_name = classify_color(dominant_color)
                        
                        # Get shape classification
                        shape_name = classify_shape(contour)

                        # Display detection information on frame with background for better visibility
                        label_y = y - 10 if y - 10 > 30 else y + h + 20
                        
                        # Add background rectangles for text
                        cv2.rectangle(frame, (x, label_y - 50), (x + 200, label_y), (0, 0, 0), -1)
                        
                        # Display text
                        cv2.putText(frame, f'Shape: {shape_name}', (x + 5, label_y - 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.putText(frame, f'Color: {color_name}', (x + 5, label_y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                        crab_detected = True

        if not crab_detected:
            cv2.putText(frame, 'No Crab Detected', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Display windows
        cv2.imshow("Threshold", thresh)
        cv2.imshow("Crab Detection", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the detection
detect_crab_from_camera()
