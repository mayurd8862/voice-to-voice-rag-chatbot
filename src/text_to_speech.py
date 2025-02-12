from smallest import Smallest
import os 

def text_to_speech_file(text):
    client = Smallest(api_key=os.getenv("SMALLEST_API_KEY"))
    client.synthesize(
        text,
        sample_rate=24000,
        speed=1.0,
        save_as="output.wav"
    )

    return "output.wav"


