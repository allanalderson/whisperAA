import whisper
# import os
import tiktoken
import torch

#
# import os
#
# # Get the base directory of your project
# base_dir = os.path.dirname(os.path.abspath(__file__))
# # Specify the relative path to the audio file
# audio_file_path = os.path.join(base_dir, "audioFiles", "allanVoice")
# print (audio_file_path)
#
# model = whisper.load_model("base")  #  model = whisper.load_model("base", device="cpu")
#
# # with torch.cuda.device(device):
# result = model.transcribe(audio_file_path)
#
# print(result["text"])


model = whisper.load_model("base")
result = model.transcribe("audioFiles/allanVoice.mp3")
print(result["text"])