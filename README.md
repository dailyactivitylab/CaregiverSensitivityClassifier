# CaregiverSensitivityClassifier
This repository contains the code for the real-world binary caregiver sensitivity classification model, developed for audio data collected using LENA devices. The model is described in detail in our [paper]()(In Submission).

Given raw audio recordings of infant distress episodes, along with distress annotations and LENA-derived adult speech tags, the pipeline extracts relevant features and predicts the caregiver’s sensitivity level (_High_ vs _Low_) for each episode.

If you use this code or model in your work, please cite our paper (see below).

## Citation Information
Khante, P., Madden-Rusnak, A., & de Barbaro, K. (2025). Real-world Classification of Caregiver Sensitivity to Infant Distress. _In Submission_.

## Models and Main Package Versions 
Trained Random Forest model can be found in this repository: Final_Sensitivity_Model.

### Versions

# Code
## Input Requirements

This code assumes that each distress episode has been annotated for **distress** — either through manual labeling or using publicly available models — and includes pre-extracted **LENA adult speech tags** (`FAN` and `MAN`).

Each participant should have a single `.csv` file containing annotations and ground truth labels for all their distress episodes. These files should be placed in the `Participant_Annotations` folder in `Example_inputs`. The format must **exactly match** the structure shown in `P1.csv` (more info below).

Raw audio files for each distress episode should be stored as individual `.wav` files in the `Raw_Audio` folder, with filenames matching the format: `<Participant_ID>_<GT_Episode>`, where `Participant_ID` is the identifier for the participant and `GT_Episode` corresponds to the episode identifier in the CSV.

### 📄 Input CSV File Format

Each `.csv` file in the `Participant_Annotations` folder should contain the following columns:

- `GT_Episode`: Unique identifier for each distress episode (e.g., `Dist_Ep1`)
- `Begin_res`: Start time of an annotation. All new distress episodes start from 0 (in seconds).
- `End_res`: End time of an annotation (in seconds).
- `Duration - ss.msec`: Duration of the annotation (in seconds).
- `GT_Detail`: Distress marked during the episode using `Distress`. Non-distress is left unmarked.  
- `*FAN`: LENA-detected female adult speech during the episode marked using `FAN`.
- `*MAN`: LENA-detected male adult speech during the episode marked using `MAN`.
- `Sensitivity`: Ground truth label for caregiver sensitivity (`High` or `Low`).

> ⚠️ **Important:** Column names and annotation labels must match exactly for the code to run successfully.

## Preprocessing distress episodes (removing non-distress)



## Extract frame-level features

## Extract event-level features

## Extract state-level features

## Combine features and predict using multiscale temporal model
