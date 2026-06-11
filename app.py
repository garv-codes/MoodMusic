import os
import numpy as np
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
import gradio as gr
import warnings

warnings.filterwarnings("ignore")

# Initialize Spotify client
SPOTIPY_CLIENT_ID = os.environ.get('SPOTIPY_CLIENT_ID')
SPOTIPY_CLIENT_SECRET = os.environ.get('SPOTIPY_CLIENT_SECRET')

if not SPOTIPY_CLIENT_ID or not SPOTIPY_CLIENT_SECRET:
    raise ValueError("Spotify API credentials are missing. Please set SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET environment variables.")

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET
))

# Load data and prepare the scaler
print("Loading data...")
data = pd.read_csv("data.csv")

number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
               'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

print("Fitting scaler...")
scaler = StandardScaler()
scaler.fit(data[number_cols])
scaled_data = scaler.transform(data[number_cols])

def find_song(name, year):
    song_data = defaultdict(list)
    results = sp.search(q=f'track: {name} year: {year}', limit=1)
    if not results['tracks']['items']:
        return None

    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = [name]
    song_data['year'] = [year]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)

def get_song_data(song, spotify_data):
    try:
        song_data = spotify_data[(spotify_data['name'].str.lower() == song['name'].lower())
                                & (spotify_data['year'] == song['year'])].iloc[0]
        return song_data
    except IndexError:
        return find_song(song['name'], song['year'])

def get_mean_vector(song_list, spotify_data):
    song_vectors = []
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print(f"Warning: {song['name']} does not exist in Spotify or in database")
            continue
        song_vector = song_data[number_cols].values
        # Handle cases where song_data from spotify API might be 2D
        if len(song_vector.shape) > 1:
            song_vector = song_vector[0]
        song_vectors.append(song_vector)

    if not song_vectors:
        return None

    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)

def recommend_songs_gradio(song_name, year, n_songs=10):
    try:
        year = int(year)
    except ValueError:
        return "Invalid year format. Please enter a valid number."
        
    song_list = [{'name': song_name, 'year': year}]
    
    metadata_cols = ['name', 'year', 'artists']
    
    song_center = get_mean_vector(song_list, data)
    
    if song_center is None:
        return f"Could not find data for the song '{song_name}' ({year}). Please try another song."
        
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])

    rec_songs = data.iloc[index]
    # Filter out the input song itself
    rec_songs = rec_songs[~rec_songs['name'].str.lower().isin([song_name.lower()])]
    
    # Format artists column if it looks like a list
    def clean_artists(artist_str):
        if isinstance(artist_str, str) and artist_str.startswith('[') and artist_str.endswith(']'):
            import ast
            try:
                artists = ast.literal_eval(artist_str)
                return ", ".join(artists)
            except:
                pass
        return artist_str
        
    rec_songs['artists'] = rec_songs['artists'].apply(clean_artists)
    
    # Format output for Gradio Dataframe
    output_df = rec_songs[metadata_cols].copy()
    output_df.columns = ['Song Name', 'Release Year', 'Artists']
    
    return output_df

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(primary_hue="emerald", secondary_hue="gray")) as demo:
    gr.Markdown("# 🎵 MoodMusic Recommendation Engine")
    gr.Markdown("Discover music you'll love! Enter a song you like and its release year to get recommendations based on Spotify's audio features (valence, acousticness, energy, etc.).")
    
    with gr.Row():
        with gr.Column(scale=1):
            song_input = gr.Textbox(label="Song Name", placeholder="e.g. Bloody Sweet")
            year_input = gr.Number(label="Release Year", value=2023, precision=0)
            rec_count = gr.Slider(minimum=5, maximum=20, value=10, step=1, label="Number of Recommendations")
            submit_btn = gr.Button("Get Recommendations", variant="primary")
            
        with gr.Column(scale=2):
            output_df = gr.Dataframe(label="Recommended Songs", headers=['Song Name', 'Release Year', 'Artists'])
            
    submit_btn.click(
        fn=recommend_songs_gradio,
        inputs=[song_input, year_input, rec_count],
        outputs=output_df
    )
    
    gr.Markdown("### 🛠️ How it works")
    gr.Markdown("This app analyzes the 'DNA' of the song you entered by fetching its audio features (or using local data if available) and calculates the cosine distance across a dataset of ~170,000 tracks to find the closest matches.")

if __name__ == "__main__":
    demo.launch()
