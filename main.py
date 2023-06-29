'''Working with Py3.10.11 and pyTorch 2.0.1+cu117 with CUDA Toolkit 11.7 on a 2080 Ti'''
import torch
import whisper
import sys
import os

input_filename = "allanVoice.mp3"


output_filename = input_filename[:11].replace(".", "").rstrip() + ".txt"
input_folder = "readFolder"
output_folder = "writeFolder"
path_to_input = os.path.join(input_folder, input_filename)
path_to_output = os.path.join(output_folder, output_filename)

if torch.cuda.is_available():
	print("Current CUDA device:", torch.cuda.get_device_name())
	cuda_version = torch.version.cuda
	print("CUDA Toolkit version:", cuda_version)
	torch_version = torch.__version__
	print("PyTorch version:", torch_version)
	print()
	device = torch.device('cuda')
else:
	print("Torch GPU unavailable")
	device = device = torch.device('cpu')

print("Input: ", path_to_input)
print("Output: ", path_to_output)

print()
print("Loading audio file... ")
try:
	audio = whisper.load_audio(path_to_input)
except:
	print ("FILE NOT FOUND.  ERROR IN FILENAME?")
	exit()

print("  Loading model... ")
model = whisper.load_model("medium.en", device=device)
print("    Transcribing ... ")
result = model.transcribe(audio)
print("      Writing Output... ")
with open(path_to_output, "w") as file:  # # Open a file for writing
	sys.stdout = file  # Redirect the standard output to the file
	print(result["text"])  # # Print the result to the file
sys.stdout = sys.__stdout__  # # Restore the standard output





