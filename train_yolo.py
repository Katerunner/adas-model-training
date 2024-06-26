import argparse
from typing import Any, Dict
from ultralytics import YOLO


def train_model(
        model_name: str,
        data: str,
        imgsz: int = 512,
        epochs: int = 200,
        batch: int = 16,
        fliplr: float = 0.0,
        flipud: float = 0.0,
        mosaic: float = 0.0,
        mixup: float = 0.0,
        degrees: float = 8.0,
        scale: float = 0.3,
        perspective: float = 0.0001,
        name: str = 'cul_tld_512_yolov8n',
        **kwargs: Any
) -> Any:
    """
    Train YOLOv8 model with specified configuration.

    Parameters:
        model_name (str): The name of the YOLO model to load.
        data (str): Path to the config YAML file.
        imgsz (int): Image size for training.
        epochs (int): Number of training epochs.
        batch (int): Batch size for training.
        fliplr (float): Probability of flipping the image left-right.
        flipud (float): Probability of flipping the image up-down.
        mosaic (float): Probability of applying mosaic augmentation.
        mixup (float): Probability of applying mixup augmentation.
        degrees (float): Range of degrees for random rotation.
        scale (float): Scaling factor for image scaling.
        perspective (float): Perspective transformation probability.
        name (str): Name of the training run.
        **kwargs (Any): Additional parameters for YOLO training.

    Returns:
        Any: Results of the training process.
    """
    # Load the model.
    model = YOLO(model_name)

    # Training.
    results = model.train(
        data=data,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        fliplr=fliplr,
        flipud=flipud,
        mosaic=mosaic,
        mixup=mixup,
        degrees=degrees,
        scale=scale,
        perspective=perspective,
        name=name,
        **kwargs
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 model with specified configuration.")
    parser.add_argument('--model_name', type=str, required=True, help="Name of the YOLO model to load.")
    parser.add_argument('--data', type=str, required=True, help="Path to the config YAML file.")
    parser.add_argument('--imgsz', type=int, default=512, help="Image size for training.")
    parser.add_argument('--epochs', type=int, default=200, help="Number of training epochs.")
    parser.add_argument('--batch', type=int, default=16, help="Batch size for training.")
    parser.add_argument('--fliplr', type=float, default=0.0, help="Probability of flipping the image left-right.")
    parser.add_argument('--flipud', type=float, default=0.0, help="Probability of flipping the image up-down.")
    parser.add_argument('--mosaic', type=float, default=0.0, help="Probability of applying mosaic augmentation.")
    parser.add_argument('--mixup', type=float, default=0.0, help="Probability of applying mixup augmentation.")
    parser.add_argument('--degrees', type=float, default=8.0, help="Range of degrees for random rotation.")
    parser.add_argument('--scale', type=float, default=0.3, help="Scaling factor for image scaling.")
    parser.add_argument('--perspective', type=float, default=0.0001, help="Perspective transformation probability.")
    parser.add_argument('--name', type=str, default='cul_tld_512_yolov8n', help="Name of the training run.")

    # Capture all remaining arguments as key-value pairs
    args, unknown = parser.parse_known_args()

    # Process additional arguments
    kwargs = {}
    for i in range(0, len(unknown), 2):
        key = unknown[i].lstrip('--')
        value = unknown[i + 1]
        # Convert value to appropriate type (int, float, or str)
        if value.isdigit():
            kwargs[key] = int(value)
        else:
            try:
                kwargs[key] = float(value)
            except ValueError:
                kwargs[key] = value

    train_model(
        model_name=args.model_name,
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        fliplr=args.fliplr,
        flipud=args.flipud,
        mosaic=args.mosaic,
        mixup=args.mixup,
        degrees=args.degrees,
        scale=args.scale,
        perspective=args.perspective,
        name=args.name,
        **kwargs
    )
