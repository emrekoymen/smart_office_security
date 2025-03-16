import torch
import time
import os
import re
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
DEVICE = "cuda"
TEMPERATURE = 1.0
CONFIDENCE_THRESHOLD = 0.45

class GroundingDINO:
    def __init__(self, model_id, device):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.device = device

    def detect_objects(self, image_path, text_queries):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, text=text_queries, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits / TEMPERATURE
        confidence_scores = torch.sigmoid(logits).squeeze().cpu().numpy()
        
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, box_threshold=0.30, target_sizes=target_sizes
        )[0]
        
        return image, results

class ImageAnnotator:
    def __init__(self, output_image_folder, output_xml_folder):
        self.output_image_folder = output_image_folder
        self.output_xml_folder = output_xml_folder
        os.makedirs(output_image_folder, exist_ok=True)
        os.makedirs(output_xml_folder, exist_ok=True)

    def annotate_image(self, image, bboxes, output_path):
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
            label, score = bbox['label'], bbox['score']

            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
            draw.text((x_min, y_min - 10), f"{label}: {score:.2f}", fill="red", font=font)

        image.save(output_path)
        print(f"✅ Saved annotated image: {output_path}")

    def save_pascal_voc_xml(self, filename, width, height, bboxes):
        xml_filename = os.path.splitext(filename)[0] + ".xml"
        xml_path = os.path.join(self.output_xml_folder, xml_filename)

        annotation = ET.Element("annotation")
        ET.SubElement(annotation, "folder").text = ""
        ET.SubElement(annotation, "filename").text = filename

        source = ET.SubElement(annotation, "source")
        ET.SubElement(source, "database").text = "Unknown"

        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = "3"

        ET.SubElement(annotation, "segmented").text = "0"

        for bbox in bboxes:
            obj = ET.SubElement(annotation, "object")
            ET.SubElement(obj, "name").text = bbox['label']
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "occluded").text = "0"
            ET.SubElement(obj, "difficult").text = "0"

            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(int(bbox['xmin']))
            ET.SubElement(bndbox, "ymin").text = str(int(bbox['ymin']))
            ET.SubElement(bndbox, "xmax").text = str(int(bbox['xmax']))
            ET.SubElement(bndbox, "ymax").text = str(int(bbox['ymax']))

        tree = ET.ElementTree(annotation)
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)
        print(f"✅ Saved XML annotation: {xml_path}")

class LabelMapper:
    LABEL_MAPPING = {
        "high-visibility vest": "HV_VEST",
        "high-visibility jacket": "HV_JACKET",
        "worker": "WORKER",
        "hard hat": "HELMET",
        "bump cap": "BUMPCAP"
    }

    @staticmethod
    def map_label(detected_label):
        detected_label_lower = detected_label.lower()
        if "vest" in detected_label_lower:
            return "HV_VEST"
        elif "jacket" in detected_label_lower:
            return "HV_JACKET"
        for key, mapped_label in LabelMapper.LABEL_MAPPING.items():
            if key in detected_label_lower:
                return mapped_label
        return detected_label.upper()

class ImageProcessor:
    def __init__(self, model, annotator, text_queries):
        self.model = model
        self.annotator = annotator
        self.text_queries = text_queries

    def process_image(self, image_path):
        image, results = self.model.detect_objects(image_path, self.text_queries)
        width, height = image.size
        bboxes = []

        for score, detected_label, box in zip(results['scores'], results['labels'], results['boxes']):
            if score.item() > CONFIDENCE_THRESHOLD:
                mapped_label = LabelMapper.map_label(detected_label)
                box = [round(i, 1) for i in box.tolist()]

                bboxes.append({
                    "label": mapped_label,
                    "score": round(score.item(), 2),
                    "xmin": box[0], "ymin": box[1],
                    "xmax": box[2], "ymax": box[3]
                })

        return image, width, height, bboxes

class GroundingDinoLabellingTool:
    def __init__(self, input_folder, output_image_folder, output_xml_folder, text_queries):
        self.input_folder = input_folder
        self.model = GroundingDINO(MODEL_ID, DEVICE)
        self.annotator = ImageAnnotator(output_image_folder, output_xml_folder)
        self.processor = ImageProcessor(self.model, self.annotator, text_queries)

    def run(self):
        print("\n🚀 Starting Grounding DINO Labelling...")
        start_time = time.time()

        for filename in filter(lambda x: x.lower().endswith(".jpg"), os.listdir(self.input_folder)):
            image_path = os.path.join(self.input_folder, filename)
            print(f"\n🔍 Processing: {filename}")

            image, width, height, bboxes = self.processor.process_image(image_path)

            if bboxes:
                output_image_path = os.path.join(self.annotator.output_image_folder, filename.replace(".jpg", "_bbox.jpg"))
                self.annotator.annotate_image(image, bboxes, output_image_path)
                self.annotator.save_pascal_voc_xml(filename, width, height, bboxes)
            else:
                print(f"⚠️ No objects detected in {filename}.")

        end_time = time.time()
        print(f"\n✅ Process completed in {end_time - start_time:.2f} seconds.")

