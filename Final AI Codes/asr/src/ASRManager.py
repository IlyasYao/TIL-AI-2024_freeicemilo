from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import librosa
from io import BytesIO

class ASRManager:
    def __init__(self):
        # Initialize the model here
        model_save_path = "whisper-small/model"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = WhisperProcessor.from_pretrained(model_save_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_save_path)
        self.model.to(self.device)

    def preprocess_audio(self, audio_bytes: bytes):
        # Load audio from bytes and preprocess it
        audio_input, sample_rate = librosa.load(BytesIO(audio_bytes), sr=16000)
        input_features = self.processor(audio_input, return_tensors="pt", sampling_rate=16000).input_features
        input_features = input_features.to(self.device)
        return input_features
    
    def transcribe(self, audio_bytes: bytes) -> str:
        # Preprocess the audio
        input_features = self.preprocess_audio(audio_bytes)

        # Perform inference
        with torch.no_grad():
            generated_ids = self.model.generate(input_features)

        # Decode the output
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return transcription