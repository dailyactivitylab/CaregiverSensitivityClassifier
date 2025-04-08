# CaregiverSensitivityClassifier
This repository contains the code for the real-world binary caregiver sensitivity classification model, developed for audio data collected using LENA devices. The model is described in detail in our [paper]()(In Submission).

Given raw audio recordings of infant distress episodes, along with distress annotations and LENA-derived adult speech tags, the pipeline extracts relevant features and predicts the caregiver’s sensitivity level (_High_ vs _Low_) for each episode.

If you use this code or model in your work, please cite our paper (see below).

## Citation Information
Khante, P., Madden-Rusnak, A., & de Barbaro, K. (2025). Real-world Classification of Caregiver Sensitivity to Infant Distress. _In Submission_.

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

### 3. `Raw_Audio/`
This folder should contain the original raw `.wav` files, one for each distress episode. The filenames must match the distress episodes IDs in `Distress_Episode_List.csv`.

## Code
### Step 1: Preprocess Distress Episodes (Remove Non-distress)
The script `Code/remove_nondistress.py` processes each `.csv` file in the `Participant_Annotations` folder along with its corresponding audio file in the `Raw_Audio` folder. It removes all non-distress segments and retains only the portions labeled as `Distress`. The resulting audio for each episode is saved in the `Audio_Nondistress_Removed` folder, with all distress segments concatenated into a single `.wav` file per episode.

```
python remove_nondistress.py
```

### Step 2: Extract Frame-level Features
The script `Code/extract_frame_features.py` reads each distress episode listed in `Distress_Episode_List.csv`, loads the corresponding `.wav` file from the `Audio_Nondistress_Removed` folder, extracts acoustic features, and saves their summary statistics (median, IQR, min, max) to `Features/Frame_Features.csv`.

```
python extract_frame_features.py
```

### Step 3: Extract Event-level Features
The script `Code/extract_event_features.py` reads each distress episode from `Episode_Annotations.csv`, locates corresponding `.wav` files in the `Raw_Audio` folder, and extracts event-level features summarizing caregiver responsiveness and vocal tone. These include response latencies, response proportions, and vocal tone warmth (F0) statistics for FAN and MAN vocalizations during distress periods. Results are saved to `Features/Event_Features.csv`.

```
python extract_event_features.py
```

### Step 4: Extract State-level Features

```
python extract_state_features.py
```

### Step 5: Predict Sensitivity using Multiscale Temporal Model

```
python predict.py
```

## Contact
Contact Priyanka Khante and Kaya de Barbaro should you have any question/suggestion or if you have any problems running the code at [priyanka.khante@utexas.edu](mailto:priyanka.khante@utexas.edu) and [kaya@austin.utexas.edu](mailto:kaya@austin.utexas.edu).
