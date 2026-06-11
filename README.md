---
title: MoodMusic
emoji: 🎵
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.8.0
app_file: app.py
pinned: false
---

<div align="center">

# 🎵 MoodMusic

### AI-Powered Music Recommendation System

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Gradio](https://img.shields.io/badge/Gradio-App-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)](https://huggingface.co/spaces/Garv-codes/MoodMusic)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Spotify](https://img.shields.io/badge/Dataset-Spotify-1DB954?style=for-the-badge&logo=spotify&logoColor=white)](https://www.kaggle.com/datasets/vatsalmavani/spotify-dataset)

**[🚀 Try the Live Demo on Hugging Face Spaces!](https://huggingface.co/spaces/Garv-codes/MoodMusic)**

*Discover music you'll love — powered by machine learning and Spotify's audio features.*

---

</div>

## 📖 Overview

**MoodMusic** is an intelligent music recommendation system that analyzes audio features from Spotify's extensive dataset to suggest songs tailored to your taste. Using clustering algorithms and content-based filtering, it maps the musical landscape and finds tracks that match your preferences based on acoustic properties like energy, danceability, valence, tempo, and more.

Built with **170,000+ tracks** spanning a century of music (1921–2020), MoodMusic doesn't just recommend popular songs — it understands the *DNA* of music.

## ✨ Features

- 🎯 **Content-Based Filtering** — Recommends songs based on audio feature similarity
- 📊 **Exploratory Data Analysis** — Deep visual analysis of music trends across decades
- 🔬 **Cluster Analysis** — Groups songs into meaningful clusters using K-Means and genre pipelines
- 📈 **Trend Visualization** — Interactive charts showing how music has evolved over 100 years
- 🎼 **Genre Profiling** — Audio fingerprints for 100+ genres
- 🔗 **Spotify Integration** — Uses Spotipy to fetch and analyze playlist data

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.8+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn (K-Means, PCA, Pipeline) |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Spotify API** | Spotipy |
| **Scientific Computing** | SciPy |
| **Environment** | Jupyter Notebook |

## 📁 Project Structure

```
MoodMusic/
├── Music_Recommendation_System.ipynb   # Main notebook — analysis & recommendation engine
├── data.csv                            # 170K+ tracks with audio features (28 MB)
├── data_by_genres.csv                  # Aggregated audio features by genre
├── data_by_year.csv                    # Aggregated audio features by year (1921–2020)
├── requirements.txt                    # Python dependencies
├── LICENSE                             # MIT License
├── CONTRIBUTING.md                     # Contribution guidelines
└── README.md                           # You are here
```

## 📊 Dataset

The project uses the [Spotify Dataset](https://www.kaggle.com/datasets/vatsalmavani/spotify-dataset) from Kaggle, containing:

| File | Records | Description |
|------|---------|-------------|
| `data.csv` | 170,653 tracks | Individual tracks with 19 audio features |
| `data_by_genres.csv` | 2,973 genres | Audio features aggregated by genre |
| `data_by_year.csv` | 100 years | Audio features aggregated by year |

**Key audio features used:**

`acousticness` · `danceability` · `energy` · `instrumentalness` · `liveness` · `loudness` · `speechiness` · `tempo` · `valence` · `popularity`

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/garv-codes/MoodMusic.git
   cd MoodMusic
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS/Linux
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Spotify API Credentials** (Required for the web app)
   Set the following environment variables with your Spotify Developer credentials:
   - `SPOTIPY_CLIENT_ID`
   - `SPOTIPY_CLIENT_SECRET`

5. **Run the Gradio Web App**
   ```bash
   python app.py
   ```
   *(The app will be available locally at http://127.0.0.1:7860)*

6. **Or Launch the Jupyter Notebook**
   ```bash
   jupyter notebook Music_Recommendation_System.ipynb
   ```

## 🧠 How It Works

MoodMusic uses an end-to-end unsupervised machine learning pipeline:

1. **Data Engineering & Preprocessing** — Cleans, normalizes, and scales 170K+ tracks across 19 audio dimensions (e.g., valence, acousticness, energy) using Scikit-Learn's `StandardScaler`.
2. **Exploratory Data Analysis (EDA)** — Visualizes historic music trends, genre distributions, and latent feature correlations using Seaborn and Plotly.
3. **Dimensionality Reduction** — Applies PCA (Principal Component Analysis) and t-SNE to reduce the high-dimensional feature space while preserving maximum variance, allowing for 2D/3D visual mapping of musical evolution.
4. **Unsupervised Clustering** — Groups songs into mathematically similar segments using K-Means clustering algorithms.
5. **Recommendation Engine** — Given a target song or playlist, the engine isolates the nearest neighbor clusters and calculates top recommendations utilizing cosine similarity and Euclidean distance metrics.

## 🗺️ Roadmap

- [x] 🌐 **Streamlit/Gradio Web Interface** — Build an interactive, Python-native web application to easily demo the model.
- [ ] 🎭 **Mood-Based Playlists** — Generate robust playlists based on real-time user mood selection.
- [ ] 🔄 **Real-Time Spotify Integration** — Connect to live Spotify API for real-time inference and personalized recommendations.
- [ ] 🤖 **Advanced Modeling** — Experiment with Neural Collaborative Filtering and Autoencoders for improved vector representations.
- [ ] 📊 **Analytics Dashboard** — Visual dashboard mapping user listening statistics against the KMeans clusters.

## 🤝 Contributing

Contributions are welcome! Please read the [Contributing Guidelines](CONTRIBUTING.md) before getting started.

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [Spotify](https://www.spotify.com/) for the audio features API
- [Kaggle](https://www.kaggle.com/) for hosting the dataset
- The open-source community for the incredible Python ML ecosystem

---

<div align="center">

**Built with ❤️ by [Garv](https://github.com/garv-codes)**

⭐ Star this repo if you found it useful!

</div>
