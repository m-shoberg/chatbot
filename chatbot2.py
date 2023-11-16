from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
from transformers import pipeline
import soundfile as sf

# Step 1: Speech Recognition
asr = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")

def transcribe_audio(audio_file):
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()
    return asr(audio_bytes)["text"]

# Step 2: Language Processing
text_generator = pipeline("text-generation", model="gpt2")

def generate_response(text_input):
    return text_generator(text_input, max_length=50)[0]["generated_text"]

# Load FastSpeech2 model
models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
    "facebook/fastspeech2-en-ljspeech",
    arg_overrides={"vocoder": "hifigan", "fp16": False}
)
model = models[0]
TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
generator = task.build_generator([model], cfg)

def synthesize_speech(text_input):
    sample = TTSHubInterface.get_model_input(task, text_input)
    wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)
    return wav, rate

# Main function to process speech-to-speech
def speech_to_speech(audio_file):
    text_input = transcribe_audio(audio_file)
    text_response = generate_response(text_input)
    wav, rate = synthesize_speech(text_response)

    # Save the output to a file
    sf.write('response.wav', wav, rate)

    return "response.wav"

# Example usage
audio_file = "/Users/marcus/Desktop/chat_bot/audio.wav"
response_audio = speech_to_speech(audio_file)
