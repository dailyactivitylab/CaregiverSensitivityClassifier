#! /usr/bin/env python

import os
import librosa
import numpy as np
import pandas as pd
from pydub import AudioSegment

# This file extracts the event-level features from each of the distress episodes.
#
# For each episode we compute (all distress annotation rows and the non-distress rows around them are used, so
# a caregiver response that only comes after the infant has stopped is still counted):
#   - Latency of response: for each distress vocalization (an uninterrupted stretch of distress rows), the time
#     from its onset to the first caregiver (FAN or MAN) vocalization before the next distress vocalization
#     begins. The median and IQR over the episode's vocalizations are saved (the IQR needs at least two).
#   - Proportion of response: the share of distress vocalizations that a caregiver vocalized during.
#   - Overall latency of response (Caregiver_first_begin): time from the start of the episode to the first
#     caregiver vocalization.
#   - Warmth: median and IQR of the F0 of FAN and MAN vocalizations during distress.

EPISODE_ANNOTATIONS = "../Example_Inputs/Episode_Annotations.csv"
RAW_AUDIO_DIR = "../Example_Inputs/Raw_Audio"
OUTPUT_CSV = "../Features/Event_level_features.csv"

# Rows closer together than this (seconds) are treated as touching
TIME_TOLERANCE = 0.01

OUTPUT_COLUMNS = ["Dist_Ep_IDs", "Caregiver_first_begin", "Overall_latency_of_response",
                  "Latency_of_response_median", "Latency_of_response_iqr",
                  "Caregiver_resp_prop", "Warmth_median_FAN", "Warmth_iqr_FAN", "Warmth_median_MAN", "Warmth_iqr_MAN"]


def has_value(value):
    """True if an annotation cell (*FAN / *MAN) is marked."""
    return pd.notna(value) and str(value).strip() != ""


def get_distress_stretches(rows):
    """Split an episode's rows (sorted by time) into distress vocalizations: uninterrupted runs of distress rows.
    Returns a list of lists of rows."""
    stretches, current, previous_end = [], [], None
    for _, row in rows.iterrows():
        if row["GT_Detail"] == "Distress":
            if current and abs(row["Begin_res"] - previous_end) <= TIME_TOLERANCE:
                current.append(row)
            else:
                if current:
                    stretches.append(current)
                current = [row]
            previous_end = row["End_res"]
        else:
            if current:
                stretches.append(current)
            current, previous_end = [], None
    if current:
        stretches.append(current)
    return stretches


def get_response_features(group):
    """Latencies of response, proportion of distress vocalizations with a response, and the time of the first
    caregiver vocalization in the episode."""
    rows = group.sort_values("Begin_res")
    stretches = get_distress_stretches(rows)

    caregiver = rows["*FAN"].apply(has_value) | rows["*MAN"].apply(has_value)
    caregiver_begins = rows.loc[caregiver, "Begin_res"].to_numpy(dtype=float)

    latencies, n_responded = [], 0
    onsets = [stretch[0]["Begin_res"] for stretch in stretches]
    for k, stretch in enumerate(stretches):
        onset = onsets[k]
        next_onset = onsets[k + 1] if k + 1 < len(onsets) else np.inf
        # First caregiver vocalization from this onset until the next distress vocalization starts
        in_window = caregiver_begins[(caregiver_begins >= onset - TIME_TOLERANCE) & (caregiver_begins < next_onset - TIME_TOLERANCE)]
        if len(in_window) > 0:
            latencies.append(max(0.0, float(in_window.min() - onset)))
        # A response only counts toward the proportion if the caregiver vocalizes during the distress itself
        if any(has_value(r["*FAN"]) or has_value(r["*MAN"]) for r in stretch):
            n_responded += 1

    response_prop = n_responded / len(stretches) if stretches else np.nan
    first_begin = float(caregiver_begins.min() - rows["Begin_res"].min()) if len(caregiver_begins) > 0 else np.nan
    return latencies, response_prop, first_begin


def get_f0(audio, begin, end):
    """Voiced F0 values (Hz) in the audio between begin and end (seconds), as in the original script."""
    segment = audio[begin * 1000:end * 1000]
    samples = np.array(segment.get_array_of_samples()).astype(np.float32)
    if len(samples) == 0:
        return np.array([])
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(y=samples, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    except Exception as e:  # e.g. segment too short to analyse
        print(f"  Could not extract F0 for {begin:.2f}-{end:.2f}s: {e}")
        return np.array([])
    # Only keep the voiced parts
    return f0[voiced_flag]


def median_and_iqr(values):
    if len(values) == 0:
        return np.nan, np.nan
    return float(np.median(values)), float(np.percentile(values, 75) - np.percentile(values, 25))


def main():
    episode_annotations = pd.read_csv(EPISODE_ANNOTATIONS)

    rows = []
    for dist_ep_id, group in episode_annotations.groupby("Dist_Ep_IDs"):
        distress_rows = group[group["GT_Detail"] == "Distress"].sort_values("Begin_res")

        # Latency and proportion of caregiver response, and time of the first caregiver vocalization
        latencies, caregiver_resp_prop, caregiver_first_begin = get_response_features(group)
        overall_latency = min(latencies) if latencies else np.nan
        latency_median = float(np.median(latencies)) if latencies else np.nan
        latency_iqr = float(np.percentile(latencies, 75) - np.percentile(latencies, 25)) if len(latencies) > 1 else np.nan

        has_fan = distress_rows["*FAN"].apply(has_value)
        has_man = distress_rows["*MAN"].apply(has_value)

        # Warmth (F0) of FAN and MAN vocalizations during distress. The audio is loaded once per episode.
        fan_f0, man_f0 = [], []
        if has_fan.any() or has_man.any():
            audio_path = os.path.join(RAW_AUDIO_DIR, f"{dist_ep_id}.wav")
            try:
                audio = AudioSegment.from_file(audio_path, format="wav")
            except FileNotFoundError:
                print(f"Warning: audio file not found for episode {dist_ep_id}, warmth features will be empty")
                audio = None
            if audio is not None:
                for (_, row), fan, man in zip(distress_rows.iterrows(), has_fan, has_man):
                    if fan:
                        fan_f0.extend(get_f0(audio, row["Begin_res"], row["End_res"]))
                    if man:
                        man_f0.extend(get_f0(audio, row["Begin_res"], row["End_res"]))
        fan_median, fan_iqr = median_and_iqr(fan_f0)
        man_median, man_iqr = median_and_iqr(man_f0)

        rows.append({
            "Dist_Ep_IDs": dist_ep_id,
            "Caregiver_first_begin": caregiver_first_begin,
            "Overall_latency_of_response": overall_latency,
            "Latency_of_response_median": latency_median,
            "Latency_of_response_iqr": latency_iqr,
            "Caregiver_resp_prop": caregiver_resp_prop,
            "Warmth_median_FAN": fan_median,
            "Warmth_iqr_FAN": fan_iqr,
            "Warmth_median_MAN": man_median,
            "Warmth_iqr_MAN": man_iqr,
        })

    output_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    output_df.to_csv(OUTPUT_CSV, index=False)


if __name__ == "__main__":
    main()
