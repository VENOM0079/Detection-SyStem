from ultralytics import YOLO

def train_yolo_phone(data_path, epochs=50, imgsz=640):
    """
    Instructions for dataset preparation:
    1. Organize labeled images in YOLO format:
       data/
         train/
           images/
           labels/
         val/
           images/
           labels/
    2. Create a data.yaml:
       train: ../data/train/images
       val: ../data/val/images
       nc: 2
       names: ['person', 'cell phone']
    """
    # Load a pretrained model
    model = YOLO('yolov8n.pt')

    # Train the model
    # Note: In a real scenario, you'd point to your custom data.yaml
    # results = model.train(data=data_path, epochs=epochs, imgsz=imgsz)
    print(f"To train: model.train(data='{data_path}', epochs={epochs}, imgsz={imgsz})")
    
if __name__ == "__main__":
    print("YOLOv8 Fine-tuning Script Template")
    print("Required classes: person, cell phone")
    # Example usage:
    # train_yolo_phone('path/to/custom_data.yaml')
