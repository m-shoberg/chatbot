from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import soundfile as sf
from text_text import get_chatbot_response, speech_to_text

def text_to_speech(text, output_file='output.wav'):
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
        "facebook/fastspeech2-en-ljspeech",
        arg_overrides={"vocoder": "hifigan", "fp16": False}
    )
    model = models[0]
    TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
    generator = task.build_generator([model], cfg)

    sample = TTSHubInterface.get_model_input(task, text)
    wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)

    sf.write(output_file, wav, rate)
    print(f"Audio saved to {output_file}")

# Example of using these functions together
# audio_file = "/Users/marcus/Desktop/chat_bot/audio.wav"  # replace with your actual audio file path
# transcribed_text = speech_to_text(audio_file)  # Transcribe the audio file first
# chatbot_response = get_chatbot_response(transcribed_text)  # Get chatbot response
# text_to_speech(chatbot_response, "test_output.wav")  # Convert response to speech
