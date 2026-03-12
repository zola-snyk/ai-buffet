import requests
from PIL import Image
from transformers import BlipForQuestionAnswering, BlipProcessor

processor = BlipProcessor.from_pretrained("gpt-4-0613")
processor2 = BlipProcessor.from_pretrained("gpt-4")
processor3 = BlipProcessor.from_pretrained("openai/gpt-4-0613")
processor4 = BlipProcessor.from_pretrained("azure/us/gpt-4-0613")
processor5 = BlipProcessor.from_pretrained("databricks/databricks-gpt-4-0613")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

question = "how many men are in the picture?"
inputs = processor(raw_image, question, return_tensors="pt")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))
