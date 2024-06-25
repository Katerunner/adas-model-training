from ultralytics import YOLO
import time
import cv2

DEFAULT_MODEL_PATH = 'trained_models/yolov8s.pt'
DEFAULT_VIDEO_PATH = 'videos/video_1.mp4'


def load_model(model_path: str = DEFAULT_MODEL_PATH):
    loaded_model = YOLO(model_path)
    return loaded_model


if __name__ == '__main__':
    model = load_model(DEFAULT_MODEL_PATH)
    cap = cv2.VideoCapture(DEFAULT_VIDEO_PATH)

    prev_frame_time = 0
    new_frame_time = 0

    try:
        while cap.isOpened():
            success, frame = cap.read()

            if success:
                new_frame_time = time.time()

                results = model(frame, verbose=False)
                annotated_frame = results[0].plot()

                fps = 1 / (new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time
                fps_str = str(int(fps))

                cv2.putText(annotated_frame,
                            f'FPS: {fps_str}',
                            (7, 70), cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (100, 255, 0),
                            3, cv2.LINE_AA)

                cv2.imshow("YOLOv8 Inference", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
