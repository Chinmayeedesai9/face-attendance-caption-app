import torch
from torchvision import models, transforms
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load models once
det_model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
det_model.eval()

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption_paragraph(image_path, detected_names=None):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0)

    captions = []

    with torch.no_grad():
        predictions = det_model(image_tensor)

    for i, box in enumerate(predictions[0]['boxes']):
        score = predictions[0]['scores'][i]
        if score < 0.6:
            continue

        x_min, y_min, x_max, y_max = box.int().tolist()
        region = image.crop((x_min, y_min, x_max, y_max))

        inputs = blip_processor(images=region, return_tensors="pt")
        with torch.no_grad():
            output = blip_model.generate(**inputs)
        caption = blip_processor.decode(output[0], skip_special_tokens=True).strip()
        if caption and caption not in captions:
            captions.append(caption)

    # Compose paragraph with multiple lines
    if captions:
        para = "Here’s what I see in the image:\n"
        for cap in captions[:5]:  # Limit to first 5 regions
            para += f"- {cap}\n"
    else:
        para = "No clear objects were detected in the image.\n"

    # Add detected names if available
    if detected_names:
        para += "\nThe following students were identified in the image:\n"
        for name in detected_names:
            para += f"- {name}\n"

    return para
