# This file extracts the frame-level featurs from each of the distress episodes

import os
import numpy as np
import pandas as pd
from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import ShortTermFeatures
from tqdm import tqdm

# Paths
DISTRESS_LIST_CSV = "../Example_Inputs/Distress_Episode_List.csv"
AUDIO_DIR = "../Example_Inputs/Audio_Nondistress_Removed"
OUTPUT_CSV = "../Features/Frame_level_features.csv"

# Read list of distress episodes
df_list = pd.read_csv(DISTRESS_LIST_CSV)

# Create output dataframe
columns = ["Dist_Ep_IDs"] + [f"F_{i}" for i in range(1, 77)]  # 19 features × 4 stats = 76
all_features = pd.DataFrame(columns=columns)

for _, row in tqdm(df_list.iterrows(), total=len(df_list)):
    episode_id = row['Dist_Ep_IDs']
    audio_file = episode_id + ".wav"
    audio_path = os.path.join(AUDIO_DIR, audio_file)
    
    try:
        Fs, audio = aIO.read_audio_file(audio_path)
        audio = audio.astype(np.float32)
        dur = len(audio) / Fs

        if dur >= 0.4:
            # Extract short-term features with deltas
            F, _ = ShortTermFeatures.feature_extraction(audio, Fs, 0.4 * Fs, 0.2 * Fs, deltas=True)

            # Keep 19 relevant derivatives/deltas (exclude chroma/std/rolloff/flux)
            F = np.concatenate((F[34:40], F[42:55]), axis=0)

            # Aggregate: median, IQR, min, max
            F_out = np.concatenate((
                np.median(F, axis=1),
                np.percentile(F, 75, axis=1) - np.percentile(F, 25, axis=1),
                np.min(F, axis=1),
                np.max(F, axis=1)
            ), axis=0)

            final_features = [episode_id] + F_out.tolist()

            # Append to output CSV
            pd.DataFrame([final_features]).to_csv(OUTPUT_CSV, mode='a', header=not os.path.exists(OUTPUT_CSV), index=False)

    except MemoryError:
        print(f"Skipped {audio_file} due to memory error (likely too long).")
        continue
    except FileNotFoundError:
        print(f"Audio file not found: {audio_file}")
        continue
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        continue
