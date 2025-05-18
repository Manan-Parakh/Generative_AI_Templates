# GradioLlamaAPI

A lightweight Gradio-based web interface to interact with a custom CodeLlama model served locally via Ollama's REST API.  
This project wraps a custom CodeLlama assistant named **Lil_Coder**, designed specifically as a code teaching assistant.

---

## Overview

This repository contains:

- A **Modelfile** to create your custom Ollama CodeLlama model with a personalized system prompt and temperature setting.  
- A **Python script** to launch a Gradio UI that sends prompts and receives responses via Ollama's REST API.

---

## Modelfile Explanation

The Modelfile configures your Ollama model:

```modelfile
FROM codellama

PARAMETER temperature 1

SYSTEM """
You are a code teaching assistant named as Lil_Coder created by Manan.
Answer all the code related questions being asked.
"""
````

* **FROM codellama**: Base model is Meta's CodeLlama.
* **temperature 1**: Sets creativity level (higher means more creative, less deterministic).
* **SYSTEM prompt**: Defines the assistant's role and behavior.

---

## Setup Instructions

### Prerequisites

* Install [Ollama](https://github.com/ollama/ollama) and ensure it is running locally.
* Python 3.7+ environment.
* Install Python dependencies:

```bash
pip install gradio requests
```

### Creating Your Custom Model

Place the Modelfile in your working directory, then run:

```bash
ollama create Lil_Coder -f Modelfile
```

This registers the model `Lil_Coder` in Ollama.

---

## Running the Gradio App

Run the Python script (`app.py`):

```bash
python app.py
```

This will start a Gradio web UI (typically at [http://localhost:7860](http://localhost:7860)) for interacting with your custom assistant.

---

## Python Script Details

The script sends your input prompt (and conversation history) to the Ollama REST API at `http://localhost:11434/api/generate`, specifying the `Lil_Coder` model. Responses are shown in the web UI.

---

## File Structure

```
├── app.py              # Gradio interface script
├── modelfile           # Ollama model definition
├── README.md           # This file
```

---

## Customization

* Adjust `temperature` in the Modelfile for creativity control.
* Modify the `SYSTEM` prompt to change the assistant’s personality and focus.
* Improve the Gradio UI with streaming, syntax highlighting, multi-session support, etc.

---

## Known Issues

* Slow response times, especially for complex or lengthy prompts.
* No streaming support; output is delivered only after full generation completes.
* Minimal UI with no syntax highlighting or conversation history display.
* Limited error handling for API failures or server downtime.

---

## Future Improvements

* Add streaming API support for real-time response display.
* Improve UI
* Support multi-session chat history isolation for multiple users.
* Improve error handling.
* Add file upload support for code review and explanation.
  
---

## Contact

Created by Manan Parakh.
Feel free to open issues or contribute improvements!

```
