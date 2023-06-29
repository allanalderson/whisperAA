import os

# ...

# Iterate over the files
for /input in files:
	file_name = os.path.basename("\input")

	# Do something with the file name
	print(file_name)





model = whisper.load_model("base")
results = model.transcribe(["audio1.mp3", "audio2.mp3"])
print(results[0]['text'])
print(results[1]['text'])


