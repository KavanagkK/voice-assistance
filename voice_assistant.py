import whisper
import sounddevice as sd
import scipy.io.wavfile
import requests
import pyttsx3

# Configuration
OLLAMA_MODEL = "tinyllama"     # LLM model (change if using "phi", etc.)
WHISPER_MODEL = "tiny"         # Whisper STT model
DURATION = 5                   # Recording duration in seconds
AUDIO_FILE = "audio.wav"       # Temporary file

# Load Whisper STT model
print("🔄 Loading Whisper model...")
stt_model = whisper.load_model(WHISPER_MODEL)

# TTS (text-to-speech)
tts_engine = pyttsx3.init()

def record_audio():
    print("🎙️ Recording...")
    fs = 44100
    recording = sd.rec(int(DURATION * fs), samplerate=fs, channels=1)
    sd.wait()
    scipy.io.wavfile.write(AUDIO_FILE, fs, recording)
    print("✅ Done recording.")

def transcribe():
    print("🔎 Transcribing...")
    result = stt_model.transcribe(AUDIO_FILE)
    print("📝 You said:", result["text"])
    return result["text"]

def ask_ollama(prompt):
    print("🤖 Talking to Ollama...")
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    )
    reply = response.json()["response"]
    print("💬 Ollama says:", reply)
    return reply

def speak(text):
    print("🗣️ Speaking...")
    tts_engine.say(text)
    tts_engine.runAndWait()

def main():
    record_audio()
    prompt = transcribe()
    if prompt.strip():
        answer = ask_ollama(prompt)
        speak(answer)
    else:
        print("⚠️ Didn't hear anything.")

# Start
if __name__ == "__main__":
    while True:
        input("\n👉 Press Enter to speak...")
        main()
