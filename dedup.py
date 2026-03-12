import requests
from PIL import Image
from transformers import BlipForQuestionAnswering, BlipProcessor

GPT_1 = "gpt-4-0613"
GPT_2 = "gpt-4"
GPT_3 = "openai/gpt-4-0613"
GPT_4 = "azure/us/gpt-4-0613"
GPT_5 = "databricks/databricks-gpt-4-0613"
GPT_6 = "azure/eu/gpt-4"
GPT_7 = "azure/global/gpt-4"
GPT_8 = "github_copilot/gpt-4"
GPT_9 = "openrouter/openai/gpt-4"
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

question = "how many men are in the picture?"
inputs = processor(raw_image, question, return_tensors="pt")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))
