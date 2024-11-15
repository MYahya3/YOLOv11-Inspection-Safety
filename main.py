import os
import time
import cv2
from ultralytics import YOLO
from utilis import YOLO_Detection, label_detection, draw_background_rectangle, sahi_detection, get_detction_info


# Function to check if an item is within a person's bounding box, with a proximity threshold
def is_within(person_box, item_box, threshold=0):
    px1, py1, px2, py2 = person_box  # Person's bounding box
    ix1, iy1, ix2, iy2 = item_box  # Item's bounding box

    # Check if the item box is close to the person's bounding box with a margin threshold
    return (
            (ix1 >= px1 - threshold and iy1 >= py1 - threshold) and
            (ix2 <= px2 + threshold and iy2 <= py2 + threshold)
    )


# Load YOLO model for person detection
def load_yolo_model(device, model_path):
    model = YOLO(model_path)
    model.to(device)
    model.nms = 0.7
    return model

# Process each frame to detect persons and safety equipment
def process_frame(frame, person_model):
    # Detect persons using YOLO_Detection
    person_boxes, person_classes, person_names, _ = YOLO_Detection(person_model, frame)

    # Detect helmets and vests using sahi_detection
    sahi_result = sahi_detection(frame)
    item_boxes, item_classes, item_names = get_detction_info(sahi_result)

    # For each person, check if helmet and vest are within their bounding box
    detections = []
    for person_box in person_boxes:
        has_helmet = False
        has_vest = False
        for item_box, item_name in zip(item_boxes, item_names):
            if is_within(person_box, item_box, threshold=60):
                if item_name == 'helmet':
                    has_helmet = True
                elif item_name == 'vest':
                    has_vest = True
        detections.append((person_box, has_helmet, has_vest))

    return detections


# Function to draw rounded rectangle with small rounded corners
def draw_rounded_rectangle(image, top_left, bottom_right, color, thickness, radius=5):
    # Create the rounded corners by drawing ellipses at each corner
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Draw the four rounded corners
    cv2.ellipse(image, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, -1)
    cv2.ellipse(image, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, -1)
    cv2.ellipse(image, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, -1)
    cv2.ellipse(image, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, -1)

    # Draw the rectangle body (without corners)
    cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(image, (x1, y1 + radius), (x1 + radius, y2 - radius), color, -1)
    cv2.rectangle(image, (x2 - radius, y1 + radius), (x2, y2 - radius), color, -1)


# Function to draw a rounded rectangle with adjustable corner rounding
def draw_rounded_rectangle(image, top_left, bottom_right, color, thickness, radius=10):
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Draw the four rounded corners
    cv2.ellipse(image, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, -1)
    cv2.ellipse(image, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, -1)
    cv2.ellipse(image, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, -1)
    cv2.ellipse(image, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, -1)

    # Draw the rectangle body (without corners)
    cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(image, (x1, y1 + radius), (x1 + radius, y2 - radius), color, -1)
    cv2.rectangle(image, (x2 - radius, y1 + radius), (x2, y2 - radius), color, -1)


# Draw detection results with helmet and vest indicators with enhanced appearance
def draw_detections(frame, detections):
    for person_box, has_helmet, has_vest in detections:
        x1, y1, x2, y2 = map(int, person_box)

        # Draw the person's bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 144, 30), 3)

        # Calculate the height of the person box to scale the indicators
        person_height = y2 - y1
        indicator_height = int(person_height * 0.08)  # Increased height to make it more visible

        # Define the left margin for the helmet and vest indicators
        left_margin = 10  # Adjusted margin for better alignment

        # Define padding for the text rectangle (larger padding for better visibility)
        text_padding = 8
        corner_radius = 10  # Slightly larger corner radius for rounded rectangle

        # Helmet indicator (connected to the left side of the person box)
        helmet_color = (87, 139, 46) if has_helmet else (0, 0, 255)  # Green if detected, Red if not detected

        # Calculate text size for the "Helmet" label
        (text_width, text_height), baseline = cv2.getTextSize("Helmet", cv2.FONT_HERSHEY_COMPLEX, 0.7, 2)

        # Adjust rectangle height and width to fit the text and padding
        rect_width = text_width + 2 * text_padding
        rect_height = text_height + 2 * text_padding

        # Draw the background rounded rectangle for the helmet label
        draw_rounded_rectangle(frame,
                               (x1 - left_margin - rect_width, y1),
                               (x1 - left_margin, y1 + rect_height),
                               helmet_color, -1, corner_radius)
        # Add the text inside the rectangle
        cv2.putText(frame, "Helmet",
                    (x1 - left_margin - rect_width + text_padding, y1 + rect_height // 2 + text_height // 2),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)

        # Gap between the helmet and vest indicators
        gap_between_indicators = 12  # Increased gap for clarity

        # Vest indicator (connected to the left side of the person box, below helmet)
        vest_color = (87, 139, 46) if has_vest else (0, 0, 255)  # Green if detected, Red if not detected

        # Calculate text size for the "Vest" label
        (text_width, text_height), baseline = cv2.getTextSize("Vest", cv2.FONT_HERSHEY_COMPLEX, 0.7, 2)

        # Adjust rectangle height and width to fit the text and padding
        rect_width = text_width + 2 * text_padding
        rect_height = text_height + 2 * text_padding

        # Draw the background rounded rectangle for the vest label below the helmet
        draw_rounded_rectangle(frame,
                               (x1 - left_margin - rect_width, y1 + rect_height + gap_between_indicators),
                               (x1 - left_margin, y1 + 2 * rect_height + gap_between_indicators),
                               vest_color, -1, corner_radius)
        # Add the text inside the rectangle
        cv2.putText(frame, "Vest",
                    (x1 - left_margin - rect_width + text_padding,
                     y1 + rect_height + gap_between_indicators + rect_height // 2 + text_height // 2),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)


# Main function to process video frames and output results
def main(source, output_path="output_video.mp4"):
    person_model = load_yolo_model(device="cuda", model_path="yolo11m.pt")
    sahi_model = ...  # Initialize your SAHI model here

    if os.path.isfile(source) and not source.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        cap = cv2.VideoCapture(source)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detect persons and their equipment
            detections = process_frame(frame, person_model)
            draw_detections(frame, detections)  # Draw bounding boxes and indicators

            out.write(frame)  # Write processed frame to output video
            cv2.imshow('Frame', cv2.resize(frame, (920, 620)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(source="22.mp4", output_path="output_video/22_out.mp4")
