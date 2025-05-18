import requests
import json
import gradio as gr

# Using the ollama REST API - https://github.com/ollama/ollama
url = 'http://localhost:11434/api/generate'

headers = {
    'Content-Type':"application/json"
}

history = []
def generate_response(prompt):
    history.append(prompt)
    final_prompt = '\n'.join(history)

    data = {
        "model": "Lil_Coder",
        "prompt": final_prompt,
        'stream': False
    }
    response = requests.post(url=url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        response  = response.text
        data = json.loads(response)
        actual_response = data['response']
        return actual_response
    else:
        print("Error!", response.text)

# Interface
interface = gr.Interface(fn = generate_response,
                         inputs = gr.Textbox(lines = 4, placeholder="Enter your prompt..."),
                         outputs = 'text')
interface.launch()