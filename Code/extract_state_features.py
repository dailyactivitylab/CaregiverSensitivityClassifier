#! /usr/bin/env python

import os
import numpy as np
import pandas as pd

# This file extracts the state-level features from each of the distress episodes.
#
# As in remove_nondistress.py, non-distress is removed, so the episode timeline is the distress
# annotation rows laid end to end. Each distress row is in one of two states:
#   Resp   - a caregiver (FAN or MAN) is vocalizing (the "response region")
#   NoResp - no caregiver vocalization
# Consecutive rows in the same state form a single visit, so a FAN row followed by a MAN row is one
# response visit, and the features do not depend on how the annotation file is split into rows.
# As described in the paper, features are computed for the attended (response) state only, because the
# unattended state is its inverse and would be redundant.

EPISODE_ANNOTATIONS = "../Example_Inputs/Episode_Annotations.csv"
OUTPUT_CSV = "../Features/State_level_features.csv"

OUTPUT_COLUMNS = ["Dist_Ep_IDs", "Episode_dur",
                  "Prop_in_Resp_region", "Visits_Resp_region",
                  "Median_return_time_resp_region", "First_entry_resp"]


def has_value(series):
    """True where an annotation column (*FAN / *MAN) is marked, False where it is empty."""
    as_text = series.astype(str).str.strip().str.lower()
    return series.notna() & (as_text != "") & (as_text != "nan")


def get_visits(group):
    """Turn one episode's distress rows into an ordered list of [in_response_region, duration] visits."""
    distress = group[group["GT_Detail"] == "Distress"].sort_values("Begin_res")
    durations = (distress["End_res"] - distress["Begin_res"]).to_numpy(dtype=float)
    responded = (has_value(distress["*FAN"]) | has_value(distress["*MAN"])).to_numpy()

    visits = []
    for is_resp, dur in zip(responded, durations):
        if dur <= 0:
            continue
        if visits and visits[-1][0] == bool(is_resp):
            visits[-1][1] += dur  # same state as the previous row: same visit
        else:
            visits.append([bool(is_resp), dur])
    return visits


def compute_features(visits):
    """Compute the state-level features for one episode from its list of visits."""
    ep_duration = sum(dur for _, dur in visits)
    resp_time = sum(dur for is_resp, dur in visits if is_resp)
    n_resp_visits = sum(1 for is_resp, _ in visits if is_resp)

    # Seconds of distress before the first response (NaN if the caregiver never responded)
    first_entry_resp = np.nan
    elapsed = 0.0
    for is_resp, dur in visits:
        if is_resp:
            first_entry_resp = elapsed
            break
        elapsed += dur

    # Median return time: median length of the no-response gaps between the first and last response.
    # Needs at least two response visits, otherwise NaN.
    median_return_time = np.nan
    resp_idx = [i for i, (is_resp, _) in enumerate(visits) if is_resp]
    if len(resp_idx) > 1:
        gaps = [visits[i][1] for i in range(resp_idx[0], resp_idx[-1] + 1) if not visits[i][0]]
        median_return_time = float(np.median(gaps))

    return {
        "Episode_dur": ep_duration,
        "Prop_in_Resp_region": resp_time / ep_duration,
        # Transitions into the state (visits), normalised by episode duration (visits per second)
        "Visits_Resp_region": n_resp_visits / ep_duration,
        "Median_return_time_resp_region": median_return_time,
        "First_entry_resp": first_entry_resp,
    }


def main():
    episode_annotations = pd.read_csv(EPISODE_ANNOTATIONS)

    rows = []
    for dist_ep_id, group in episode_annotations.groupby("Dist_Ep_IDs"):
        visits = get_visits(group)
        if not visits:
            print(f"Warning: no distress segments found for episode {dist_ep_id}, skipping")
            continue
        features = compute_features(visits)
        features["Dist_Ep_IDs"] = dist_ep_id
        rows.append(features)

    output_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    output_df.to_csv(OUTPUT_CSV, index=False)


if __name__ == "__main__":
    main()
