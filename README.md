import os
import librosa
import whisper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt_tab')


# Load Whisper model
model = whisper.load_model("base")

# === Step 1: Audio to Text ===
def transcribe_audio(audio_path):
    result = model.transcribe(audio_path)
    return result['text'], result['segments'], result['language']

# === Step 2: Audio Feature Extraction ===
def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path)
    duration = librosa.get_duration(y=y, sr=sr)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitches = pitches[magnitudes > np.median(magnitudes)]
    pitch_std = np.std(pitches) if len(pitches) > 0 else 0
    return duration, pitch_std

# === Step 3: Text Feature Extraction ===
def extract_text_features(text, duration_sec):
    words = word_tokenize(text.lower())
    sentences = sent_tokenize(text)

    pause_per_sentence = len(sentences) / duration_sec if duration_sec > 0 else 0
    hesitation_count = sum([words.count(hm) for hm in ['uh', 'um', 'er', 'ah']])
    word_count = len(words)
    speech_rate = word_count / duration_sec * 60 if duration_sec > 0 else 0

    incomplete_sentences = sum(1 for s in sentences if not s.endswith(('.', '!', '?')))
    recall_issue_heuristic = sum(1 for w in words if w in ['thing', 'stuff', 'that'])  # generic substitutions

    return {
        'pauses_per_sentence': pause_per_sentence,
        'hesitation_markers': hesitation_count,
        'speech_rate_wpm': speech_rate,
        'incomplete_sentences': incomplete_sentences,
        'recall_substitutes': recall_issue_heuristic
    }

# === Step 4: Process a Batch of Audio Files ===
def process_files(audio_files):
    feature_list = []

    for file in audio_files:
        text, _, _ = transcribe_audio(file)
        duration, pitch_std = extract_audio_features(file)
        text_feats = extract_text_features(text, duration)

        text_feats.update({
            'pitch_variability': pitch_std,
            'filename': os.path.basename(file)
        })

        feature_list.append(text_feats)

    return pd.DataFrame(feature_list)

# === Step 5: Anomaly Detection with Isolation Forest ===
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def detect_anomalies(df):
    # Select features for modeling
    features = ['speech_rate_wpm', 'pauses_per_sentence', 'pitch_variability', 
                'hesitation_markers', 'recall_substitutes', 'incomplete_sentences']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    # Initialize and fit the model
    iso = IsolationForest(contamination=0.2, random_state=42)
    iso.fit(X_scaled)  # ✅ This is what was missing

    # Predict anomalies
    df['anomaly'] = iso.predict(X_scaled)
    df['anomaly_score'] = iso.decision_function(X_scaled)

    # Convert -1 (anomaly) and 1 (normal) to 1 and 0 for better readability
    df['anomaly'] = df['anomaly'].apply(lambda x: 1 if x == -1 else 0)
    
    return df


# === Example Usage ===
audio_files = ['/content/hindi.mp3', '/content/common_voice_en_41910504.mp3']  # Update with your files
df = process_files(audio_files)
df = detect_anomalies(df)

print(df)

# === Optional: Visualization ===
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='speech_rate_wpm', y='pitch_variability', hue='anomaly', palette='coolwarm')
plt.title('Cognitive Risk Detection via Speech Patterns')
plt.xlabel('Speech Rate (wpm)')
plt.ylabel('Pitch Variability')
plt.show()

