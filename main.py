
import torch
import whisper

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


audio = whisper.load_audio("audioFiles/allanVoice.mp3")
model = whisper.load_model("small", device=device)

result = model.transcribe(audio, language='en')



print(result["text"])
