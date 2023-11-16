import os
import subprocess
from speech_text import speech_to_text  # Assuming this is in speech_text.py
from text_text import get_chatbot_response  # Assuming this is in text_text.py
from text_speech import text_to_speech  # Assuming text_to_speech function is in text_speech.py
import signal

def record_audio(output_file):
    try:
        # Command for recording audio
        command = ["ffmpeg", "-f", "avfoundation", "-i", ":2", "-ar", "16000", output_file]
        
        # Start recording
        print("Starting recording... Press ENTER to stop.")
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as process:
            input()
            process.send_signal(signal.SIGINT)  # Send SIGINT signal to ffmpeg process
            process.wait()  # Wait for the process to terminate
            print(f"\nRecording stopped. File saved to {output_file}")

        # Wait for the recording to be stopped manually
        # process.wait()

    except subprocess.CalledProcessError as e:
        print(f"An error occurred during recording: {e}")

def main():
    audio_file = "/Users/marcus/Desktop/chat_bot/audio.wav"

    # Check if audio file exists
    if not os.path.exists(audio_file):
        print(f"{audio_file} not found. Starting new recording.")
        record_audio(audio_file)

    # Process the audio file
    transcribed_text = speech_to_text(audio_file)  # Transcribe the audio file
    chatbot_response = get_chatbot_response(transcribed_text)  # Get chatbot response
    text_to_speech(chatbot_response, "test_output.wav")  # Convert response to speech

if __name__ == "__main__":
    main()
