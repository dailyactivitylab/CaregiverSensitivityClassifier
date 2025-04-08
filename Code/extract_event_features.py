# This file extracts the event-level features from each of the distress episodes

import librosa
import numpy as np
import pandas as pd
from pydub import AudioSegment
from itertools import compress
import scipy.stats as sp

EPISODE_ANNOTATIONS = "../Example_Inputs/Episode_Annotations.csv"
OUTPUT_CSV = "../Features/Event_level_features.csv"

# Get annotations
episode_annotations = pd.read_csv(EPISODE_ANNOTATIONS)

# Initialize output DataFrame
output_df = pd.DataFrame(columns=["Dist_Ep_IDs", "Overall_latency_of_response", "Latency_of_response_median", "Latency_of_response_iqr", 
                                  "Caregiver_resp_prop", "Warmth_median_FAN", "Warmth_iqr_FAN", "Warmth_median_MAN", "Warmth_iqr_MAN"])

# Process each distress episode
for dist_ep_id, group in episode_annotations.groupby("Dist_Ep_IDs"):

    # Initialize lists to store all F0 values for FAN and MAN within the distress episode
    fan_f0_values = []
    man_f0_values = []
    
    # Initialize variables to compute latencies and caregiver response proportions
    latencies = []
    num_with_response = 0

    # Track the current vocalization status
    vocalizing = False
    first_fan_or_man_time = None

    # Iterate through each row in the group to extract relevant information
    for i, row in group.iterrows():
        # Check for a "Distress" vocalization
        if row['GT_Detail'] == "Distress":
            if not vocalizing:
                vocalizing = True
                first_fan_or_man_time = None  # Reset when a new vocalization starts
                
            # Process FAN response (if exists and co-occurs with distress)
            if pd.notna(row['*FAN']):
                if first_fan_or_man_time is None:
                    first_fan_or_man_time = row['Begin_res']  # Record first FAN response time
                # Get the corresponding audio of the distress episode
                audio = AudioSegment.from_file(f"../Example_Inputs/Raw_Audio/{dist_ep_id}.wav", format="wav")
                dist_ep = audio[row['Begin_res']*1000:row['End_res']*1000]
                dist_ep = dist_ep.get_array_of_samples()
                dist_ep = np.array(dist_ep).astype(np.float32)

                # Extract warmth (F0) from the FAN audio segment
                f0, voiced_flag, voiced_probs = librosa.pyin(y=dist_ep, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
                # Only compute F0 averages for voiced parts
                f0_sub = np.array(list(compress(f0, voiced_flag)))
                fan_f0_values.extend(f0_sub)
                
            # Process MAN response (if exists and co-occurs with distress)
            if pd.notna(row['*MAN']):
                if first_fan_or_man_time is None:
                    first_fan_or_man_time = row['Begin_res']  # Record first MAN response time
                # Get the corresponding audio of the distress episode
                audio = AudioSegment.from_file(f"../Example_Inputs/Raw_Audio/{dist_ep_id}.wav", format="wav")
                dist_ep = audio[row['Begin_res']*1000:row['End_res']*1000]
                dist_ep = dist_ep.get_array_of_samples()
                dist_ep = np.array(dist_ep).astype(np.float32)

                # Extract warmth (F0) from the MAN audio segment
                f0, voiced_flag, voiced_probs = librosa.pyin(y=dist_ep, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
                # Only compute F0 averages for voiced parts
                f0_sub = np.array(list(compress(f0, voiced_flag)))
                man_f0_values.extend(f0_sub)
            
        else:
            # End of vocalization; compute latencies and response proportions
            if vocalizing:
                if first_fan_or_man_time is not None:
                    latency = first_fan_or_man_time - row['Begin_res']
                    latencies.append(latency)
                vocalizing = False

            # Proportion of responses
            if row['GT_Detail'] == "Distress" and (pd.notna(row['*FAN']) or pd.notna(row['*MAN'])):
                num_with_response += 1

    # Calculate overall latency of response (first FAN or MAN) for the episode
    if latencies:
        overall_latency_of_response = min(latencies)
    else:
        overall_latency_of_response = np.nan

    # Calculate median and IQR of latency of responses
    if latencies:
        latency_median = np.median(latencies)
        latency_iqr = np.percentile(latencies, 75) - np.percentile(latencies, 25)
    else:
        latency_median, latency_iqr = np.nan, np.nan

    # Calculate the proportion of responses
    total_distress_vocalizations = group[group['GT_Detail'] == "Distress"].shape[0]
    if total_distress_vocalizations > 0:
        caregiver_resp_prop = num_with_response / total_distress_vocalizations
    else:
        caregiver_resp_prop = np.nan

    # Calculate median and IQR for FAN and MAN F0 within the distress episode
    if fan_f0_values:
        fan_median = np.median(fan_f0_values)
        fan_iqr = np.percentile(fan_f0_values, 75) - np.percentile(fan_f0_values, 25)
    else:
        fan_median, fan_iqr = np.nan, np.nan

    if man_f0_values:
        man_median = np.median(man_f0_values)
        man_iqr = np.percentile(man_f0_values, 75) - np.percentile(man_f0_values, 25)
    else:
        man_median, man_iqr = np.nan, np.nan

    # Append the results to the output DataFrame
    output_df = output_df.append({
        "Dist_Ep_IDs": dist_ep_id,
        "Overall_latency_of_response": overall_latency_of_response,
        "Latency_of_response_median": latency_median,
        "Latency_of_response_iqr": latency_iqr,
        "Caregiver_resp_prop": caregiver_resp_prop,
        "Warmth_median_FAN": fan_median,
        "Warmth_iqr_FAN": fan_iqr,
        "Warmth_median_MAN": man_median,
        "Warmth_iqr_MAN": man_iqr
    }, ignore_index=True)

# Save the output to a CSV
output_df.to_csv(OUTPUT_CSV, index=False)
