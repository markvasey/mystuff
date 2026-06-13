import os
import sys
import shutil

def main():
    dest_dir = "/home/markvasey/Dropbox/GitHub/mystuff/TapoViewerMonitorV2/src/main/resources"
    dest_file = os.path.join(dest_dir, "yolov8n-pose.onnx")
    
    if os.path.exists(dest_file):
        print(f"Model already exists at {dest_file}")
        return

    # Try to import ultralytics, install if missing
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics package not found. Installing via pip...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "onnx"])
        from ultralytics import YOLO

    print("Downloading and exporting YOLOv8n-pose model to ONNX...")
    model = YOLO("yolov8n-pose.pt")
    # Export with dynamic axes or standard 640x640 input shape
    path = model.export(format="onnx", opset=19)
    
    if os.path.exists(path):
        os.makedirs(dest_dir, exist_ok=True)
        shutil.move(path, dest_file)
        print(f"Successfully exported and moved model to {dest_file}")
        
        # Clean up temporary downloads
        if os.path.exists("yolov8n-pose.pt"):
            os.remove("yolov8n-pose.pt")
    else:
        print("Failed to export model to ONNX.")

if __name__ == "__main__":
    main()
