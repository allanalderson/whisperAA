
import torch
import time
import whisper
import os

audio = whisper.load_audio("audioFiles/allanVoice.mp3")
device = torch.device('cuda')
# model = whisper.load_model("small")
model = whisper.load_model("small", device=device)
#

result = model.transcribe(audio)



print(result["text"])
