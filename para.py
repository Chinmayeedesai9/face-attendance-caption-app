import torch
from torchvision import models, transforms
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load a pre-trained object detection model (CPU-friendly)
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load and preprocess the image
image = Image.open("tr.jpg")
transform = transforms.Compose([transforms.ToTensor()])
image_tensor = transform(image).unsqueeze(0)

# Get predictions
with torch.no_grad():
    predictions = model(image_tensor)

# Filter predictions based on confidence threshold
threshold = 0.5
boxes = [box for i, box in enumerate(predictions[0]['boxes']) if predictions[0]['scores'][i] > threshold]

print(f"Processing {len(boxes)} regions with confidence > {threshold}.")

# Load the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Collect all cropped regions
regions = []
for box in boxes:
    x_min, y_min, x_max, y_max = box.int().tolist()
    regions.append(image.crop((x_min, y_min, x_max, y_max)))

# Batch process regions for caption generation
if len(regions) > 0:
    inputs = processor(images=regions, return_tensors="pt", padding=True)  # BLIP input tensor
    with torch.no_grad():
        outputs = caption_model.generate(**inputs)
    captions = [processor.decode(output, skip_special_tokens=True) for output in outputs]
else:
    captions = []

# Remove duplicates and clean captions
identified_objects = list(set(captions))
identified_objects = [obj.strip() for obj in identified_objects if obj.strip()]

# Generate a concise paragraph based on identified objects
if len(identified_objects) > 0:
    paragraph = f"In the image, I see {', '.join(identified_objects[:3])}."
else:
    paragraph = "No identifiable objects were detected in the image."

# Print the results
print("Generated Paragraph:",paragraph)