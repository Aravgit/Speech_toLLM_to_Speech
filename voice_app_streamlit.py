import streamlit as st
import subprocess
import pyaudio
import sys
import ipywidgets as widgets
from IPython.display import display
import wave
import threading
import time 
import librosa
import soundfile as sf
from IPython.display import Audio
import os
from gtts import gTTS
import IPython.display as ipd
import asyncio
import requests
from gtts import gTTS
import librosa
import soundfile as sf
hostname= 'hostname where the seamless models are hosted'
API_URL = f"http://{hostname}/translate/"
API_URL_Dload =  f"http://{hostname}/download/"


def adjust_speed_librosa(audio_path, output_path, speed_factor):
    y, sr = librosa.load(audio_path, sr=None)
    y_fast = librosa.effects.time_stretch(y,rate= speed_factor)
    sf.write(output_path, y_fast, sr)


def send_file_for_translation(input_data, mode, tgt_lang, src_lang):
    if mode == "t2st" or mode == "t2tt":  # text to speech translation or text to text translation
        data = {
            'input_data': input_data,  # this is the text input
            'mode': mode,
            'tgt_lang': tgt_lang,
            'src_lang': src_lang
        }
        response = requests.post(API_URL, data=data)

    else:  # mode == "s2tt" for speech to text translation
        with open(input_data, 'rb') as f:  # here input_data is the file path
            files = {'file': (input_data, f, 'audio/wav')}
            data = {
                'mode': mode,
                'tgt_lang': tgt_lang,
                'src_lang': src_lang
            }
            response = requests.post(API_URL, files=files, data=data)
    response_json = response.json()  # Parse the JSON response
    if mode in ["s2tt", "t2tt"]:
        return response_json.get('translated_text', None)
    elif mode == "t2st":
        return response_json.get('audio_link', None)
            

def text_to_audio(text, lang='hi'):
#     translator = Translator()
    """Converts given text to audio and plays it."""
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save("output_audio.mp3")
#     return "output_audio.mp3" #ipd.Audio("output_audio.mp3")

CHUNK = 1024
WIDTH = 2
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 15
OUTPUT_FILENAME = "input_check.wav"

frames = []
stop_flag = threading.Event()
p = pyaudio.PyAudio()
stream = None

def record_audio():
    global frames, stream, stop_flag
    
    frames.clear()
    stream = p.open(format=p.get_format_from_width(WIDTH),
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    st.write("* recording")
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        if stop_flag.is_set():
            break
        data = stream.read(CHUNK)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()

    with wave.open(OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(p.get_format_from_width(WIDTH)))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
    st.write(f"* saved to {OUTPUT_FILENAME}")
    
    
import streamlit as st
from audio_recorder_streamlit import audio_recorder

# if audio_bytes:
#     st.audio(audio_bytes, format="audio/wav")

def main():
    st.title("KnowledgeX Voice Chat")
    st.header("Ask question in any language")
    audio_bytes = audio_recorder(pause_threshold=10.0, sample_rate=41000)
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        with open(OUTPUT_FILENAME, 'wb') as f:
            f.write(audio_bytes)
        st.write("recorded your question. Processing .......")
        ##### Voice translation
        y, sr = librosa.load(f'{OUTPUT_FILENAME}', sr=None)  # Load the file
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=16000)  # Resample to 16000Hz
        sf.write('inputre.wav', y_resampled, 16000)   # Save the resampled audio
        file_path = 'inputre.wav'
        translation = send_file_for_translation(file_path,'s2tt','eng',None)
        st.write(f"Question: {translation}")
        ### LLM Processing
        subprocess.run(['python', 'subprocess_fn.py', OUTPUT_FILENAME], capture_output=True, text=True)
#         answer = completed_process.stdout.strip()
        with open("output.txt", "r") as file:
            answer = file.read()
        st.write(f'LLM output : {answer}')
        ### Text to speech translation
        final_response = send_file_for_translation(input_data=str(answer),mode='t2tt',tgt_lang='hin',src_lang="eng")
        st.write(f'Translated response : {final_response}')
        ###Text to Audio 
        text_to_audio(text=final_response, lang='hi')
        audio_path = "output_audio.mp3"
        if os.path.abspath("output_audio.mp3"):
            print('file_exists')
        # Usage
        input_path = 'output_audio.mp3'
        output_path = 'output_audio_fast.mp3'
        adjust_speed_librosa(input_path, output_path, 1.4)

        audio_file = open(f"{output_path}", "rb")
        audio_bytes_out = audio_file.read()
        st.audio(audio_bytes_out, format="audio/mp3", start_time=0)
#         if st.button('Clear'):
#             st.experimental_rerun()
        
main()
p.terminate()
