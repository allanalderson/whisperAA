import torch
import whisper
import sys
import os

initial_prompts = ["Moonbi", "Attunga", "Kootingal", "Attunga", "Quirindi", "Attunga"]

input_folder = "audio_input"
output_folder = "text_output"


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
    device = torch.device('cpu')

print(" Loading model... \n")
model = whisper.load_model("medium.en", device=device)

for input_filename in os.listdir(input_folder):
    if input_filename.endswith(".mp3") or input_filename.endswith(".wav") or input_filename.endswith(".flac"):
        path_to_input = os.path.join(input_folder, input_filename)
        output_filename = input_filename[:11].replace(".", "").rstrip() + ".txt"
        path_to_output = os.path.join(output_folder, output_filename)

        # print("Input: ", path_to_input)
        # print("Output: ", path_to_output)
        #
        # print()
        # print("Loading audio file... ")
        try:
            audio = whisper.load_audio(path_to_input)
        except:
            print("FILE NOT FOUND. ERROR IN FILENAME?")
            continue


        print("   Transcribing ", input_filename)

        # Join the initial prompts into a single string
        prompt_text = "\n".join(initial_prompts)
        # Transcribe using the model and prompts
        result = model.transcribe(audio, prompt=prompt_text)
        # print(" Writing", input_filename)
        with open(path_to_output, "w") as file:
            sys.stdout = file
            print(result["text"])
        sys.stdout = sys.__stdout__

