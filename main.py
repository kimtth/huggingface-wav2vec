# For managing audio file
import librosa
# Importing Pytorch
import torch
# Importing Wav2Vec tokenizer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

# Importing Wav2Vec pretrained model
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

audio, rate = librosa.load("taken_clip.wav", sr=16000)
print(audio)
print(rate)

# Taking an input value
input_values = tokenizer(audio, return_tensors="pt").input_values
print(input_values)
print(input_values)

# Storing logits (non-normalized prediction values)
logits = model(input_values).logits
print(logits)

# Storing predicted id's
prediction = torch.argmax(logits, dim=-1)
print(prediction)

# Passing the prediction to the tokenzer decode to get the transcription
transcription = tokenizer.batch_decode(prediction)[0]
print(transcription)
