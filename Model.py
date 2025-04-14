import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from deep_translator import GoogleTranslator
import torchaudio.transforms as T
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pronouncing

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the processor and model for speech-to-text
processor = Wav2Vec2Processor.from_pretrained("ai4bharat/indicwav2vec-hindi")
model = Wav2Vec2ForCTC.from_pretrained("ai4bharat/indicwav2vec-hindi")
model.to(device)

# Open a file dialog for the user to select an audio file
def select_audio_file():
    Tk().withdraw()  # Close the root window
    file_path = askopenfilename(filetypes=[("Audio Files", "*.wav")])
    return file_path

def transcribe_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != 16000:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    input_values = processor(waveform.squeeze(), return_tensors="pt", sampling_rate=16000).input_values
    with torch.no_grad():
        logits = model(input_values.to(device)).logits
    transcription = processor.batch_decode(torch.argmax(logits, dim=-1))[0]
    return transcription

# Define diphthongs with ARPAbet phoneme sequences
diphthong_phonemes = {
    'ɑι': ['AY'],  # Example: "high", "night", "I"
    'ɑʊ': ['AW'],  # Example: "cow", "loudly"
    'ɔι': ['OY'],  # Example: "boy"
    'eι': ['EY'],  # Example: "say"
    'əʊ': ['OW'],  # Example: "go"
    'eə': ['EH', 'R'],  # Example: "lair"
    'ιə': ['IH', 'R'],  # Example: "idea"
    'ʊə': ['UH', 'R'],  # Example: "sure"
}

# Function to strip stress markers (e.g., AY1 -> AY)
def strip_stress(phoneme):
    return ''.join([char for char in phoneme if not char.isdigit()])

# Function to detect syllables and diphthongs in a word
def detect_syllables_and_diphthongs(word):
    pronunciations = pronouncing.phones_for_word(word)
    if not pronunciations:
        return {"Word": word, "Syllables": [], "Diphthongs": []}
    
    detected_syllables = []
    detected_diphthongs = []
    
    for pronunciation in pronunciations:
        phonemes = pronunciation.split()
        phonemes_no_stress = [strip_stress(phoneme) for phoneme in phonemes]
        syllables = [strip_stress(p) for p in phonemes if any(char.isdigit() for char in p)]
        detected_syllables.extend(syllables)
        
        for diphthong, phoneme_sequence in diphthong_phonemes.items():
            # Ensure the detected diphthongs are actual diphthongs, not just adjacent vowels
            if len(phoneme_sequence) == 1 and phoneme_sequence[0] in phonemes_no_stress:
                detected_diphthongs.append(diphthong)
            elif len(phoneme_sequence) > 1 and ' '.join(phoneme_sequence) in ' '.join(phonemes_no_stress):
                detected_diphthongs.append(diphthong)
    
    return {
        "Word": word,
        "Syllables": detected_syllables,
        "Diphthongs": list(set(detected_diphthongs))
    }

def process_sentence(transcription):
    words = transcription.split()
    results = [detect_syllables_and_diphthongs(word.lower()) for word in words]
    return results

def main():
    audio_file = select_audio_file()
    if not audio_file:
        print("No file selected. Exiting...")
        return
    
    transcription = transcribe_audio(audio_file)
    print("Transcription:", transcription)
    
    translated_text = GoogleTranslator(source='auto', target='en').translate(transcription)
    print("Translated Text:", translated_text)
    
    results = process_sentence(translated_text)
    for result in results:
        print(f"Word: {result['Word']}")
        print(f"Syllables: {', '.join(result['Syllables']) if result['Syllables'] else 'None'}")
        print(f"Diphthongs Detected: {', '.join(result['Diphthongs']) if result['Diphthongs'] else 'None'}")
        print("-" * 40)  # Separator for readability

if __name__ == "__main__":
    main()
