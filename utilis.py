import cv2
import numpy as np
from sahi import AutoDetectionModel  # SAHI's auto-detection model for object detection
from sahi.predict import get_sliced_prediction  # Function for slicing input images for detection


# Initialize the SAHI (Slicing Aided Hyper Inference) object detection model using YOLOv8
sahi_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',  # YOLOv8 model type
    model_path="runs/detect/train/weights/best.pt",  # Path to the pre-trained YOLOv8 model
    confidence_threshold=0.1,  # Minimum confidence threshold for detection
)

# Perform object detection on the given frame using the SAHI model
def sahi_detection(frame):
    result = get_sliced_prediction(
        frame,
        sahi_model,
        slice_height=320,  # Height of each slice to split the image for detection
        slice_width=320,  # Width of each slice
        overlap_height_ratio=0.1,  # Overlap ratio between slices (height)
        overlap_width_ratio=0.1,  # Overlap ratio between slices (width)
    )
    return result  # Return the detection result

# Extract bounding boxes, class IDs, and names from the detection result
def get_detction_info(result):
    boxes = []  # List to hold bounding box coordinates
    classes = []  # List to hold detected class IDs
    names = []  # List to hold names of detected objects
    for object_prediction in result.object_prediction_list:
        # Filter detections for vehicles (cars, buses, trucks, motorcycles)
        # if object_prediction.category.name in ["hel", "bus", "truck", "motorcycle"]:
        box = object_prediction.bbox.to_xyxy()  # Get the bounding box in XYXY format
        boxes.append(box)  # Append the bounding box to the list
        classes.append(object_prediction.category.id)  # Append the class ID
        names.append(object_prediction.category.name)  # Append the object name (class)
    return boxes, classes, names  # Return the lists of boxes, classes, and names


# To make detections and get required outputs
def YOLO_Detection(model, frame, conf=0.3):
    # Perform inference on an image
    results = model.predict(frame, conf=conf, classes = [0])
    # Extract bounding boxes, classes, names, and confidences
    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    names = results[0].names
    confidences = results[0].boxes.conf.tolist()

    return boxes, classes, names, confidences

## Draw YOLOv8 detections function
def label_detection(frame, text, left, top, bottom, right, tbox_color=(30, 155, 50), fontFace=1, fontScale=0.8,
                    fontThickness=1):
    # Draw Bounding Box
    cv2.rectangle(frame, (int(left), int(top)), (int(bottom), int(right)), tbox_color, 2)
    # Draw and Label Text
    textSize = cv2.getTextSize(text, fontFace, fontScale, fontThickness)
    text_w = textSize[0][0]
    text_h = textSize[0][1]
    y_adjust = 10
    cv2.rectangle(frame, (int(left), int(top) + text_h + y_adjust), (int(left) + text_w + y_adjust, int(top)),
                  tbox_color, -1)
    cv2.putText(frame, text, (int(left) + 5, int(top) + 10), fontFace, fontScale, (255, 255, 255), fontThickness,
                cv2.LINE_AA)

# Updated drawPolygons function to avoid changing to green if detection inside
def drawPolygons(frame, points_list, detection_in_polygon=False, blink_state=False, alpha=0.2):
    # Color for blinking red effect
    polygon_color_inside = (0, 0, 255) if blink_state else (0, 0, 0)  # Toggle red/transparent for inside polygons
    polygon_color_outside = (30, 50, 250)  # Default color for outside polygons

    # Create a transparent overlay for the polygons
    overlay = frame.copy()

    for area in points_list:
        # Reshape the flat tuple to an array of shape (4, 1, 2)
        area_np = np.array(area, np.int32)

        # Draw filled polygons for detections inside with blinking effect
        if detection_in_polygon:
            # Blinking red for polygons with detections inside
            cv2.fillPoly(overlay, [area_np], polygon_color_inside)
        else:
            # Draw the polygon boundary (no fill) for outside detections
            cv2.polylines(overlay, [area_np], isClosed=True, color=polygon_color_outside, thickness=3)

    # Blend the overlay with the original frame
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    return frame


def draw_background_rectangle(frame, rect_dimensions):
    """Draw a rectangle for the danger alert."""
    rect_width, rect_height = rect_dimensions
    frame_height, frame_width, _ = frame.shape
    top_left = (int((frame_width - rect_width) / 2), 10)
    bottom_right = (top_left[0] + rect_width, top_left[1] + rect_height)

    cv2.rectangle(frame, top_left, bottom_right, (255, 255, 255), thickness=-1)
    return top_left, rect_height


def overlay_icon(frame, icon, position):
    """Overlay the danger icon on the frame."""
    icon_x_offset, icon_y_offset = position
    frame[icon_y_offset:icon_y_offset + icon.shape[0],
    icon_x_offset:icon_x_offset + icon.shape[1]] = icon


# def setup_device():
#     """Check if CUDA is available and set the device."""
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
#     return device

