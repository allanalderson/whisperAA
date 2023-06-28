
import torch
import whisper
import sys

if torch.cuda.is_available():
	print("Current CUDA device:", torch.cuda.get_device_name())
	cuda_version = torch.version.cuda
	print("CUDA Toolkit version:", cuda_version)
	torch_version = torch.__version__
	print("PyTorch version:", torch_version)
	print()
	device = device = torch.device('cuda')
else:
	print ("Torch GPU unavailable")
	device = device = torch.device('cpu')


audio = whisper.load_audio("inputAudio/AustralianVoice.mp3")
model = whisper.load_model("small", device=device)
result = model.transcribe(audio, language='en')



# print(result["text"])


with open("outputText/output.txt", "w") as file:  # # Open a file for writing
	sys.stdout = file  # Redirect the standard output to the file
	print(result["text"])  # # Print the result to the file

sys.stdout = sys.__stdout__  # # Restore the standard output