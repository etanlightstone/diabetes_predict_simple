# Diabetes Prediction Project

This project trains a simple neural network model to predict the likelihood of diabetes based on synthetic patient data.

## Project Structure

```
.
├── diabetes_dataset.csv       # Generated dataset
├── diabetes_inference.py      # Script for running inference (TODO)
├── diabetes_model.py          # Defines the PyTorch model architecture
├── diabetes_trainer.py        # Script for training the model
├── generate_fake_data.py      # Script to generate synthetic data
├── requirements.txt           # Project dependencies
└── README.md                  # This file
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd diabetes_predict
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```


## Data Generation

(I recommend skipping data generation, the included csv file was already generated and is fine)

The synthetic dataset (`diabetes_dataset.csv`) is generated using the `generate_fake_data.py` script. This script creates a mix of clearly defined diabetic/non-diabetic profiles and some overlapping data to make the classification task more challenging.

To regenerate the dataset, run:
```bash
python generate_fake_data.py
```
This will overwrite the existing `diabetes_dataset.csv` file.


*(Note: The Domino command above is illustrative. You might need to adjust it based on your specific Domino CLI setup and project configuration in `domino_project_settings.md`)*

## Training

The model is trained using the `diabetes_trainer.py` script. This script performs the following steps:
1.  Loads data from `diabetes_dataset.csv`.
2.  Preprocesses the data (scaling features).
3.  Splits the data into training (80%) and validation (20%) sets.
4.  Initializes the model defined in `diabetes_model.py`.
5.  Trains the model using specified hyperparameters (epochs, batch size, learning rate, hidden layer dimensions).
6.  Logs parameters, metrics (training loss, validation loss, accuracy), and the trained model using MLflow.
7.  Saves the trained model state dictionary locally (e.g., `diabetes_model_YYYYMMDD_HHMMSS.pt`).

**To run training locally:**
```bash
python diabetes_trainer.py --epochs 50 --batch_size 64 --learning_rate 0.001
```
You can adjust the hyperparameters using command-line arguments. See the script for available options:
```bash
python diabetes_trainer.py --help
```

## Inference

The `diabetes_inference.py` script is intended for loading a trained model and making predictions on new data. (This script is currently empty and needs implementation).

## MLflow Tracking

The training script integrates with MLflow to track experiments. Ensure your MLflow tracking server is configured correctly (either locally or within your Domino environment) to capture the runs. Model artifacts, parameters, and metrics will be logged, allowing for comparison and management of different training runs.

