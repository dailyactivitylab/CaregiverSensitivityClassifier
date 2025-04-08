#! /Users/priyanka/anaconda3/bin/python

import os
import pandas as pd
from pydub import AudioSegment

# This file takes in the raw audio inputs for the distress episodes. It uses the annotated distress
# to keep the distress and remove the non-distress. Outputs are .wav episodes with non-distress removed.

import os
import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm

# Paths
ANNOTATION_FILE = "../Example_Inputs/Episode_Annotations.csv"
RAW_AUDIO_DIR = "../Example_Inputs/Raw_Audio"
OUTPUT_DIR = "../Example_Inputs/Audio_Nondistress_Removed"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the annotation data
df = pd.read_csv(ANNOTATION_FILE)

# Filter only rows marked with "Distress" in GT_Detail
df_distress = df[df["GT_Detail"] == "Distress"]

# Group by each episode
for episode_id, group in tqdm(df_distress.groupby("Dist_Ep_IDs")):
    audio_path = os.path.join(RAW_AUDIO_DIR, f"{episode_id}.wav")
    
    if not os.path.exists(audio_path):
        print(f"Warning: Audio file not found for episode {episode_id}")
        continue
    
    try:
        audio = AudioSegment.from_wav(audio_path)
    except Exception as e:
        print(f"Error loading audio for {episode_id}: {e}")
        continue
    
    # Concatenate all distress segments
    distress_segments = []
    for _, row in group.iterrows():
        start_ms = row["Begin_res"] * 1000
        end_ms = row["End_res"] * 1000
        segment = audio[start_ms:end_ms]
        distress_segments.append(segment)
    
    if distress_segments:
        concatenated = sum(distress_segments)
        output_path = os.path.join(OUTPUT_DIR, f"{episode_id}.wav")
        concatenated.export(output_path, format="wav")
    else:
        print(f"No distress segments found for {episode_id}")
