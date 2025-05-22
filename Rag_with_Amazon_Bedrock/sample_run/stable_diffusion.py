import boto3
import json
import os
import base64

# Setup the request json
prompt_data ="""
Write a poem on Generative AI.
"""

bedrock = boto3.client(service_name = 'bedrock-runtime')

payload_image = {
    "text_prompts": [
        {
            "text": "A pink penguin eating icecream",
            "weight": 1
        }
    ],
    "cfg_scale": 10,
    "seed": 0,
    "steps": 50,
    "width": 512,
    "height": 512
}
body_image = json.dumps(payload_image)

modelId = 'stability.stable-diffusion-xl-v1'

response = bedrock.invoke_model(
    body = body_image,
    modelId = modelId,
    accept = 'application/json',
    contentType = 'application/json'
)

response_body = json.loads(response.get("body").read())
print(response_body)
artifact = response_body.get("artifacts")[0]
image_encoded = artifact.get("base64").encode("utf-8")
image_bytes = base64.b64decode(image_encoded)

# Save image to a file in the output directory.
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
file_name = f"{output_dir}/generated-img.png"
with open(file_name, "wb") as f:
    f.write(image_bytes)