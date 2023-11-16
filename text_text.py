import requests
from speech_text import speech_to_text

API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
headers = {"Authorization": "Bearer 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"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def get_chatbot_response(text_input):
    payload = {
        "inputs": {
            "text": text_input
        }
    }

    response = query(payload)
    if 'generated_text' in response:
        generated_text = response['generated_text']
        print(generated_text)  # This will print the generated text
        return generated_text
    else:
        print("Response not found.")  # This will print if 'generated_text' is not in response
        return "Response not found."

# This part needs to be modified to take text input from the user or another source
if __name__ == "__main__":
    text_input = speech_to_text('/Users/marcus/Desktop/chat_bot/audio.wav')
    get_chatbot_response(text_input)
