# CaregiverSensitivityClassifier
This repository contains the code for the real-world binary caregiver sensitivity classification model, developed for audio data collected using LENA devices. The model is described in detail in our [paper](https://ieeexplore.ieee.org/document/11204456).

Given raw audio recordings of infant distress episodes, along with distress annotations and LENA-derived adult speech tags, the pipeline extracts relevant features and predicts the caregiver’s sensitivity level (_High_ vs _Low_) for each episode.

If you use this code or model in your work, please cite our paper (see below).

## Citation Information
Khante, P., Madden-Rusnak, A., & de Barbaro, K. (2025). Real-world Classification of Caregiver Sensitivity to Infant Distress. 2025 IEEE International Conference on Development and Learning (ICDL).

## Repository structure

```
CaregiverSensitivityClassifier/
├── Code/                      all scripts (run them from inside this folder)
├── Example_Inputs/            Distress_Episode_List.csv, Episode_Annotations.csv, Raw_Audio/
│   └── Audio_Nondistress_Removed/   created by Step 1
├── Features/                  created by Steps 2-4
├── Model/                     Sensitivity_RF.joblib (the trained model)
├── Results/                   created by predict.py 
```

All scripts use relative paths (`../Example_Inputs`, `../Features`, and so on), so run them from inside `Code/`.

## Installation

```
pip install -r requirements.txt
```

The saved model should be loaded with the same scikit-learn version it was trained with. The versions used are
pinned in `requirements.txt`.

## Input Requirements

Before running any code, make sure the following input files and folders are prepared as shown in the `Example_Inputs` folder:

### 1. `Distress_Episode_List.csv`
A `.csv` file containing a list of all distress episode IDs to be processed. It should have a single column with the header `Dist_Ep_IDs`. 

### 2. `Episode_Annotations.csv`
A `.csv` file containing annotations for all distress episodes. To use the trained model, every distress episode must be annotated for **distress** — either through manual labeling or using publicly available models — and must include pre-extracted **LENA adult speech tags** (`FAN` and `MAN`). The file must include the following columns:

- `Dist_Ep_IDs`: Unique identifier for each distress episode (e.g., `P1_Dist_Ep1`). These IDs must match those in `Distress_Episode_List.csv`.
- `Begin_res`: Start time of an annotation. All new distress episodes start from 0 (in seconds).
- `End_res`: End time of an annotation (in seconds).
- `Duration - ss.msec`: Duration of the annotation (in seconds).
- `GT_Detail`: Distress marked during the episode using `Distress`. Non-distress is left unmarked.  
- `*FAN`: LENA-detected female adult speech during the episode marked using `FAN`.
- `*MAN`: LENA-detected male adult speech during the episode marked using `MAN`.

> ⚠️ The column names and structure must exactly match the format shown above.

Keep the non-distress rows (rows with an empty `GT_Detail`) in the file. The event-level features use them, so that a
caregiver response that only comes after the infant has stopped is still counted. Each annotation row should be a
stretch of time during which nothing changes, so a new row starts whenever distress, `*FAN` or `*MAN` starts or stops.

### 3. `Raw_Audio/`
This folder should contain the original raw `.wav` files, one for each distress episode. The filenames must match the distress episodes IDs in `Distress_Episode_List.csv`.

## Code

### Step 1: Preprocess Distress Episodes (Remove Non-distress)
The script `Code/remove_nondistress.py` reads `Example_Inputs/Episode_Annotations.csv` along with the corresponding audio files in the `Raw_Audio` folder. It removes all non-distress segments and retains only the portions labeled as `Distress`. The resulting audio for each episode is saved in the `Audio_Nondistress_Removed` folder, with all distress segments concatenated into a single `.wav` file per episode.

```
python remove_nondistress.py
```

### Step 2: Extract Frame-level Features
The script `Code/extract_frame_features.py` reads each distress episode listed in `Distress_Episode_List.csv`, loads the corresponding `.wav` file from the `Audio_Nondistress_Removed` folder, extracts acoustic features, and saves their summary statistics (median, IQR, min, max) to `Features/Frame_level_features.csv`.

Short-term acoustic features are computed in 0.4 second windows with a 0.2 second step. Nineteen features are kept,
each summarised by its median, IQR, minimum and maximum, giving 76 columns (`F_1` to `F_76`). Episodes with less than
0.4 seconds of distress audio are skipped.

```
python extract_frame_features.py
```

### Step 3: Extract Event-level Features
The script `Code/extract_event_features.py` reads each distress episode from `Episode_Annotations.csv`, locates corresponding `.wav` files in the `Raw_Audio` folder, and extracts event-level features summarizing caregiver responsiveness and vocal tone. These include response latencies, response proportions, and vocal tone warmth (F0) statistics for FAN and MAN vocalizations during distress periods. Results are saved to `Features/Event_level_features.csv`.

A *distress vocalization* is an uninterrupted stretch of distress rows. A *caregiver vocalization* is a row marked
`FAN` or `MAN`.

| Feature | Meaning |
|---|---|
| `Caregiver_first_begin` | Overall latency of response: seconds from the start of the episode to the first caregiver vocalization. Empty if the caregiver never vocalizes. |
| `Latency_of_response_median`, `Latency_of_response_iqr` | For each distress vocalization, the time from its onset to the first caregiver vocalization before the next distress vocalization begins (responses after the distress has ended count). Median and IQR over the episode. The IQR is empty with fewer than two latencies. |
| `Caregiver_resp_prop` | Share of distress vocalizations during which a caregiver vocalized (0 to 1) |
| `Warmth_median_FAN`, `Warmth_iqr_FAN`, `Warmth_median_MAN`, `Warmth_iqr_MAN` | Median and IQR of the fundamental frequency (F0) of FAN and MAN vocalizations during distress. Empty if there are none. |
| `Overall_latency_of_response` | Smallest latency of response in the episode (saved, but not used by the model) |

```
python extract_event_features.py
```

## Step 4: Extract state-level features

```
python extract_state_features.py
```

Reads `Example_Inputs/Episode_Annotations.csv` and writes `Features/State_level_features.csv`.

Non-distress is removed, so each episode is the distress rows laid end to end. Each distress row is either in the
**attended (response) state** (a caregiver, `*FAN` or `*MAN`, is vocalizing) or the unattended state. Consecutive
rows in the same state form one visit, so a FAN row followed by a MAN row is a single response visit. As in the
paper, features are computed for the attended state only, because the unattended state is its inverse.

| Feature | Meaning |
|---|---|
| `Prop_in_Resp_region` | Share of the episode spent in the attended state |
| `Visits_Resp_region` | Number of visits (transitions into the attended state) divided by episode duration |
| `Median_return_time_resp_region` | Median length of the gaps between the first and last response. Needs at least two separate responses, otherwise empty. |
| `Episode_dur`, `First_entry_resp` | Saved but not used by the model |

### Step 5: Predict Caregiver Sensitivity
The script `Code/predict.py` combines the frame, event and state features (matched on `Dist_Ep_IDs`), applies the
trained model in `Model/Sensitivity_RF.joblib`, and writes `Results/Predictions.csv`.

```
python predict.py
```

| Column | Meaning |
|---|---|
| `Dist_Ep_IDs` | Distress episode |
| `Predicted_sensitivity` | `High` or `Low` |
| `Prob_High`, `Prob_Low` | Model probabilities |

Episodes missing from any of the three feature files are left out, and a warning lists them. Missing latencies
(no caregiver response) are treated as a very late response, filled with three times the largest latency seen in
the training data. Other missing values are filled with -1, as in training.

## About the model

A random forest classifier trained on 387 distress episodes from 32 participants. It uses 85 features: the 76
frame-level acoustic features, the event-level features (`Caregiver_first_begin`, the latency median and IQR,
`Caregiver_resp_prop` and the four warmth features) and the state-level features (`Prop_in_Resp_region`,
`Visits_Resp_region` and `Median_return_time_resp_region`). In leave-one-participant-out cross-validation it reached
macro precision 0.731, macro recall 0.720 and macro F1 0.725.

## Notes and limitations

- Features for new data must be extracted with the scripts in this repository, because the model was trained on
  features computed in the same way. In particular, the F0 values for warmth are computed with `librosa.pyin` and its
  default sample rate, and frame features use 0.4 second windows with a 0.2 second step.
- The model was trained on LENA-recorded home audio, with adult speech tags from LENA. Performance on other kinds of
  recordings is not known.

## Contact
Contact Priyanka Khante and Kaya de Barbaro should you have any question/suggestion or if you have any problems running the code at [priyanka.khante@utexas.edu](mailto:priyanka.khante@utexas.edu) and [kaya@austin.utexas.edu](mailto:kaya@austin.utexas.edu).
