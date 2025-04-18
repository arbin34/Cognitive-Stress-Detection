# MemoTag Cognitive Speech Analysis – Proof of Concept Report

## 1. Introduction

In this proof-of-concept study, we explore whether early signs of cognitive stress or mild decline can be inferred from short voice clips. Leveraging natural speech characteristics and subtle linguistic patterns, the analysis uses audio processing, NLP, and unsupervised machine learning techniques to flag potentially abnormal speech behavior.

## 2. Methodology

### Audio Processing & Transcription
We employed **OpenAI's Whisper (base)** model to convert raw voice clips into accurate transcripts.

### Audio Feature Engineering
From the audio signals, we computed:
- **Duration (sec)**
- **Pitch variability (std)**

### Text & Linguistic Feature Extraction
We extracted:
- **Pauses per sentence**
- **Hesitation markers**
- **Incomplete sentences**
- **Recall issues**
- **Speech rate (wpm)**

## 3. Anomaly Detection Approach
To flag potentially abnormal patterns:
- Applied **StandardScaler** for feature normalization.
- Used **Isolation Forest** to identify outliers.

## 4. Results & Observations
Sample extracted feature set:
| Filename                            | Speech Rate (wpm) | Pitch StdDev | Hesitations | Incomplete Sentences | Anomaly |
|------------------------------------|-------------------|--------------|-------------|----------------------|---------|
| hindi.mp3                          | 97.21             | 134.22       | 4           | 1                    | ✅      |
| common_voice_en_41910504.mp3      | 138.50            | 89.35        | 0           | 0                    | ❌      |

## 5. Insights
### Most Informative Features:
- **Hesitation markers** were a strong verbal indicator.
- **Speech rate** combined with **pitch variability** helped isolate unusual emotional tone.
- **Recall-related word usage** gave early signals of memory-linked stress.

### Modeling Choice:
**Isolation Forest** was chosen due to its interpretability and ability to handle multivariate outliers.

## 6. Limitations & Next Steps
- Small audio dataset (30–50 samples).
- Integrate **real-time stream input** for predictions.
- **Clinical validation** needed for final model accuracy.

## Technical Stack
- Python, Librosa, Whisper, NLTK, Sklearn

## Project Deliverables
- Python script for pipeline (audio → features → anomaly).
- DataFrame with extracted features and anomaly scores.
- Visualization of feature-space risk clusters.
- 
