# hindi-diphthong-detector
This project detects diphthongs from speech inputs, with support for **Hindi** as the input language. It extends an earlier English-only diphthong detector by integrating speech-to-text in Hindi and machine translation to English, followed by phonetic analysis.
## 🚀 Features

- Accepts **Hindi speech input**
- Converts Hindi speech to text using `wav2vec2`
- Translates Hindi text to English using [translation method/service]
- Performs **diphthong detection** using CMU Pronouncing Dictionary
- Command-line interface (or web/GUI if applicable)
- Outputs detected diphthongs, syllables, and phonetic information

## 📦 Installation

```bash
git clone https://github.com/Samiran-Maulick/hindi-diphthong-detector.git
cd hindi-diphthong-detector
pip install -r requirements.txt
