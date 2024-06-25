import cv2
import threading
import time
from ultralytics import YOLO

SAVE = True  # Set to True to save the video, False to only display
DELAY = 0.8


def process_frame():
    global current_frame, processed_frame
    while True:
        if current_frame is not None:
            try:
                # Process the current frame with YOLO
                results = model(current_frame, verbose=False)
                # Extract the annotated frame
                annotated_frame = results[0].plot()
                processed_frame = annotated_frame

            except Exception as e:
                print(f"Error processing frame: {e}")
            finally:
                time.sleep(DELAY)  # Simulate a delay


def display_video():
    global current_frame, processed_frame
    cap = cv2.VideoCapture('videos/video_1.mp4')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video = None

    if SAVE:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter('output_raspberry.mp4', fourcc, fps, (frame_width, frame_height))

    try:
        # Create a thread for processing frames
        threading.Thread(target=process_frame, daemon=True).start()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_frame = frame

            if processed_frame is not None:
                # Overlay the processed (annotated) frame onto the original frame
                frame = cv2.addWeighted(frame, 0.5, processed_frame, 0.5, 0)

            if SAVE and output_video is not None:
                output_video.write(frame)

            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        if SAVE and output_video is not None:
            output_video.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # Initialize global variables
    current_frame = None
    processed_frame = None

    # Load the YOLO model
    model = YOLO('trained_models/yolov8s.pt')
    display_video()
