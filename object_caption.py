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

    with torch.no_grad():
        predictions = det_model(image_tensor)

    captions = []
    for i, box in enumerate(predictions[0]['boxes']):
        if predictions[0]['scores'][i] < 0.6:
            continue
        x_min, y_min, x_max, y_max = box.int().tolist()
        region = image.crop((x_min, y_min, x_max, y_max))

        inputs = blip_processor(images=region, return_tensors="pt")
        with torch.no_grad():
            output = blip_model.generate(**inputs)
        caption = blip_processor.decode(output[0], skip_special_tokens=True)
        captions.append(caption.strip())

    unique = list(set(captions))
    if unique:
        para = f"In the image, I see {', '.join(unique[:3])}, and more in the scene."
    else:
        para = "No identifiable objects were detected in the image."

    if detected_names:
        para += f" The following students were identified: {', '.join(detected_names)}."

    return para
