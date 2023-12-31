from transformers import pipeline

'''
This script is broke, go to chatbot2.py
'''

# Step 1: Speech Recognition
# Initialize the Automatic Speech Recognition (ASR) pipeline
asr = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")

def transcribe_audio(audio_file):
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()
    return asr(audio_bytes)["text"]

# Step 2: Language Processing
# Initialize the text generation pipeline
text_generator = pipeline("text-generation", model="gpt2")

def generate_response(text_input):
    return text_generator(text_input, max_length=50)[0]["generated_text"]

# Step 3: Text-to-Speech
# Initialize the TTS pipeline
tts = pipeline("text-to-speech", model="facebook/fastspeech2-en-ljspeech")

def synthesize_speech(text_input):
    return tts(text_input)

# Main function to process speech-to-speech
def speech_to_speech(audio_file):
    # Convert speech to text
    text_input = transcribe_audio(audio_file)
    
    # Generate text response
    text_response = generate_response(text_input)
    
    # Convert text response to speech
    audio_response = synthesize_speech(text_response)
    
    # Save or process the audio response
    with open("response.wav", "wb") as f:
        f.write(audio_response["stream"])

    return "response.wav"

# Example usage
audio_file = "/Users/marcus/Desktop/chat_bot/audio.wav"
response_audio = speech_to_speech(audio_file)
