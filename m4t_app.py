from fastapi import FastAPI, UploadFile, HTTPException
from scripts.m4t.predict.predict import main as m4t_predict
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from argparse import Namespace
import os
import tempfile
import torch
import whisper
import transformers
from transformers import pipeline
import torch
from transformers import WhisperProcessor

def _save_temp_file(data: bytes) -> str:
    """Saves data to a temporary file and returns the path to the file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(data)
        return tmp.name


app = FastAPI()

UPLOAD_DIR = "uploaded_files"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
device = torch.device("cuda:0")

# print(torch.device())
@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='audio/wav', filename=filename)
    else:
        raise HTTPException(status_code=404, detail="File not found")

        
        
@app.post("/translate/")
async def translate_audio(file: UploadFile = File(None),
                          input_data: str = Form(None), 
                          mode: str = Form(...),
                          tgt_lang: str = Form(...),
                          src_lang: str = Form(None),
                         model:str = Form(None)):
    
    if mode == "s2tt" and file:  # speech to text translation and file is provided
        temp_file = _save_temp_file(await file.read())
        input_path = temp_file
    elif mode == "t2st" and input_data:  # text to speech translation and text is provided
        input_path = input_data
    elif mode == "t2tt" and input_data:
        input_path = input_data
    else:
        raise ValueError("Invalid mode or input not provided!")

    # Set up arguments for m4t_predict
    output_filename = "output_t2st.wav" 
    if mode in ['t2st','s2tt']:
        if mode == "t2st":
            output_filename = "output_t2st.wav" 
        elif mode == "s2tt":
            output_filename = f"output_{file.filename}"
        args = Namespace(
            input=input_path,
            task=mode,
            tgt_lang=tgt_lang,
            src_lang=src_lang,
            output_path=os.path.join(UPLOAD_DIR, output_filename),
            model_name="seamlessM4T_large",
            vocoder_name="vocoder_36langs",
            ngram_filtering=False)
    else:
        args = Namespace(
            input=input_path,
            task=mode,
            tgt_lang=tgt_lang,
            src_lang=src_lang,
            model_name="seamlessM4T_large",
            vocoder_name="vocoder_36langs",
            ngram_filtering=False)


    # Call the prediction function
    translated_text = m4t_predict(args)

    if (mode == "s2tt"):
        model_w = whisper.load_model("large")
        
        audio = whisper.load_audio(input_path)
        audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(model_w.device)
        # detect the spoken language
        _, probs = model_w.detect_language(mel)
        print(f"Detected language: {max(probs, key=probs.get)}")
        src_lang = {max(probs, key=probs.get)}
        if file:
            os.remove(temp_file)
        
        return {"translated_text": translated_text,
                'src_lang' :src_lang} 
    elif mode == 't2tt':
        if file:
            os.remove(temp_file)
        return {"translated_text": translated_text} 
        # Use translated_text
    else:  # mode == "t2st"
        print(f"./{os.path.basename(args.output_path)}")
        return {"audio_link": f"./{os.path.basename(args.output_path)}"}  # Use args.output_path
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5003)
