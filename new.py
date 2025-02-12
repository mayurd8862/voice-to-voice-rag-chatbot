import streamlit as st
from src.speech_to_text import transcribe_audio
from src.text_to_speech import text_to_speech_file
from dotenv import load_dotenv
load_dotenv()
from audio_recorder_streamlit import audio_recorder
from groq import Groq
from langchain_groq import ChatGroq
llm = ChatGroq(model_name="Llama3-8b-8192")
# Initialize the Groq client
import io
client = Groq()


audio_value = st.audio_input("Record a voice message")

# if audio_value:
    # st.audio(audio_value)


if audio_value:
    with st.spinner("Transcribing..."):
        # Write the audio bytes to a temporary file
        webm_file_path = "temp_audio.mp3"
        with open(webm_file_path, "wb") as f:
            f.write(audio_value.read())

        # Convert the audio to text using the speech_to_text function
        transcript = transcribe_audio(client,webm_file_path)
        # output = llm.invoke(transcript)
        output = llm.invoke(f"Generate short answer for the query given below:(strictly 3-4 lines):\n\n{transcript}")

        st.write(output.content)

        if output:
            # audio_buffer = text_to_audio_buffer(output.content)
            file = text_to_speech_file(output.content)
            # Play the generated audio
            st.write(file)
            st.audio(file, format="audio/wav", autoplay=True)
        else:
            st.warning("Please enter some text!")


# if __name__ == "__main__":
    
#     st.audio("output.wav", format="audio/wav", autoplay=True)
    