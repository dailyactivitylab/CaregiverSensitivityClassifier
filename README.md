# CaregiverSensitivityClassifier
This repository contains the code for the real-world binary caregiver sensitivity classification model, developed for audio data collected using LENA devices. The model is described in detail in our [paper]()(In Submission).

Given raw audio recordings of infant distress episodes, along with distress annotations and LENA-derived adult speech tags, the pipeline extracts relevant features and predicts the caregiver’s sensitivity level (_High_ vs _Low_) for each episode.

If you use this code or model in your work, please cite our paper (see below).

## Citation Information
Khante, P., Madden-Rusnak, A., & de Barbaro, K. (2025). Real-world Classification of Caregiver Sensitivity to Infant Distress. _In Submission_.

## Models and Main Package Versions 
Trained Random Forest model can be found in this repository: Multiscale_Temporal_Model.h5.

### Versions

## Input Requirements

Before running any code, make sure the following input files and folders are prepared as shown in the `Example_Inputs` folder:

### 1. `Distress_Episode_List.csv`
A `.csv` file containing a list of all distress episode IDs to be processed. It should have a single column with the header `Dist_Ep_IDs`.

### 2. `Episode_Annotations.csv`
This folder should contain one `.csv` file per participant, with annotations covering all of their distress episodes. Each file should be named using the format `<ParticipantID>.csv` (e.g., `P1.csv`). To use the trained model, every distress episode must be annotated for **distress** — either through manual labeling or using publicly available models — and must include pre-extracted **LENA adult speech tags** (`FAN` and `MAN`). Each file must include the following columns:

- `GT_Episode`: Unique identifier for each distress episode (e.g., `Dist_Ep1`)
- `Begin_res`: Start time of an annotation. All new distress episodes start from 0 (in seconds).
- `End_res`: End time of an annotation (in seconds).
- `Duration - ss.msec`: Duration of the annotation (in seconds).
- `GT_Detail`: Distress marked during the episode using `Distress`. Non-distress is left unmarked.  
- `*FAN`: LENA-detected female adult speech during the episode marked using `FAN`.
- `*MAN`: LENA-detected male adult speech during the episode marked using `MAN`.

> ⚠️ The column names and structure must exactly match the format shown above. See `Participant_Annotations/P1.csv` for a working example.

### 3. `Raw_Audio/`
This folder should contain the original raw `.wav` files, one for each distress episode. Each file should be named as: <ParticipantID>_<GT_Episode>.wav (e.g, `P1_Dist_Ep1.wav`)

The filenames must match the distress episodes IDs in `Distress_Episode_List.csv`.

## Code
### Step 1: Preprocess Distress Episodes (Remove Non-distress)
The script `Code/remove_nondistress.py` processes each `.csv` file in the `Participant_Annotations` folder along with its corresponding audio file in the `Raw_Audio` folder. It removes all non-distress segments and retains only the portions labeled as `Distress`. The resulting audio for each episode is saved in the `Audio_Nondistress_Removed` folder, with all distress segments concatenated into a single `.wav` file per episode.

### Step 2: Extract Frame-level Features


### Step 3: Extract Event-level Features

### Step 4: Extract State-level Features

### Step 5: Predict Sensitivity using Multiscale Temporal Model
