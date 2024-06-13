from typing import List
import torch
from ultralytics import YOLO, RTDETR
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, VisionTextDualEncoderModel, AutoTokenizer, AutoImageProcessor, VisionTextDualEncoderProcessor
import numpy as np
from io import BytesIO

class VLMManager:
    def __init__(self):
        # Device configuration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the RT-DETR model
        self.rtdetr_model = RTDETR("model/rtdetr_0.856_0.828.pt").to(self.device)
        
        # Load the finetuned CLIP model
        self.clip_model = VisionTextDualEncoderModel.from_pretrained("model/clip-base-32-finetune").to(self.device)
        self.clip_processor = VisionTextDualEncoderProcessor.from_pretrained("model/clip-base-32-finetune")

        
    def preprocess_image(self, image_bytes):
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return image


    def detect_objects(self, image):
        results = self.rtdetr_model(image)

        bboxes = []
        for result in results:
            for box, cls in zip(result.boxes.xywh, result.boxes.cls):
                # Convert to COCO format
                center_x, center_y, width, height = box.tolist()
                x_min = center_x - width / 2
                y_min = center_y - height / 2
                coco_bbox = [x_min, y_min, width, height]
                bboxes.append(coco_bbox)
        return bboxes

    def object_images(self, image, bboxes):
        image_arr = np.array(image)
        object_images = []
        for bbox in bboxes:
            x_min, y_min, width, height = [int(val) for val in bbox]
            x1, y1, x2, y2 = x_min, y_min, x_min + width, y_min + height
            obj_image = image_arr[y1:y2, x1:x2]
            obj_image = Image.fromarray(obj_image)
            object_images.append(obj_image)
        return object_images


    def identify_target(self, query, images):
        try:
            inputs = self.clip_processor(
                text=[query], images=images, return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                
            logits_per_image = outputs.logits_per_image

            most_similar_idx = torch.argmax(logits_per_image, dim=0).item()
            
            # Ensure the index is within the range of images
            if most_similar_idx >= len(images):
                raise IndexError("Index out of range for identified image.")
            print(most_similar_idx)
            return most_similar_idx
        except Exception as e:
            print(f"Error identifying target: {e}")
            return -1

    def identify(self, image: bytes, caption: str) -> List[int]:
        
        # Preprocess the image
        preprocessed_image = self.preprocess_image(image)

        # Detect objects and get bounding boxes
        bboxes = self.detect_objects(preprocessed_image)
        if not bboxes:
            return [0,0,0,0]

        # Get images of objects
        images = self.object_images(preprocessed_image, bboxes)
        
        # Identify target 
        idx = self.identify_target(caption, images)
        if idx == -1:
            return [0,0,0,0]
        
        # Return bounding box of best match
        return [int(val) for val in bboxes[idx]]

