from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio
import torch

# Load pre-trained model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

def speech_to_text(audio_file):
    # Load audio file
    waveform, sample_rate = torchaudio.load(audio_file)

    # Convert stereo audio to mono if necessary
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Preprocess the waveform
    input_values = processor(waveform.squeeze(), sampling_rate=sample_rate, return_tensors="pt").input_values

    # Perform inference
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode the prediction
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    return transcription[0]


# Example usage
# audio_file = "/Users/marcus/Desktop/chat_bot/audio.wav"
# print(speech_to_text(audio_file))
