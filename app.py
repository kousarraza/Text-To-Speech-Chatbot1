import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForTextToWaveform
import torchaudio

# Initialize model
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
model = AutoModelForTextToWaveform.from_pretrained("facebook/mms-tts-eng")

def text_to_speech(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        audio = model.generate(**inputs)
    
    audio_file = "output.wav"
    torchaudio.save(audio_file, audio.squeeze(0), sample_rate=16000)
    return audio_file

# Create Gradio interface
iface = gr.Interface(
    fn=text_to_speech,
    inputs="text",
    outputs="audio",
    title="Text-to-Speech Chatbot",
    description="Enter text to convert it to speech."
)

iface.launch()
