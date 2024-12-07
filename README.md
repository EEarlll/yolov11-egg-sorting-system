# yolov11-egg-sorting-system

## Overview
This project is designed for sorting eggs using a YOLOv11-based object detection system. The system is trained to detect different types of eggs and sort them accordingly.

## Project Structure
```
.gitignore
dataset/
   Images1/
   Images2/
   telur.v1i.yolov11/
	data.yaml
	README.dataset.txt
	README.roboflow.txt
	test/
	  images/**
          labels/
	train/
          images/
          labels/
	valid/
	  images/
	  labels/
FastSAM-s.pt
fastSAM.ipynb
LICENSE
main.ipynb
README.md
requirements.txt
runs/
    segment/
yolo11n-seg.pt
yolo11s-seg.pt
```

### Key Files and Directories

- `dataset`: Contains the dataset used for training, validation, and testing.
  - `Images1/`, `Images2/`: Directories containing images.
  - `telur.v1i.yolov11/`: Contains the dataset in YOLOv11 format.
	- `data.yaml`: Configuration file for the dataset.
	- `README.dataset.txt`: Information about the dataset.
	- `README.roboflow.txt`: Information about the dataset export from Roboflow.
	- `train/`, `valid/`, `test/`: Directories containing images and labels for training, validation, and testing.

- `FastSAM-s.pt`: Pre-trained model file.
- `fastSAM.ipynb`: Jupyter notebook for running the FastSAM model.
- `LICENSE`: License file for the project.
- `main.ipynb`: Main Jupyter notebook for the project.
- `README.md`: This file.
- `requirements.txt`: List of dependencies required for the project.
- `runs`: Directory for storing model run outputs.
- `yolo11n-seg.pt`, `yolo11s-seg.pt`: Pre-trained YOLOv11 model files.

## Setup

1. Clone the repository:
	```sh
	git clone https://github.com/EEarlll/yolov11-egg-sorting-system.git
	cd yolov11-egg-sorting-system
	```

2. Install the required dependencies:
	```sh
	pip install -r requirements.txt
	```

## Usage

### Running the Model

1. Open the `main.ipynb` or `fastSAM.ipynb` notebook in Jupyter.
2. Execute the cells to load the model and run the object detection on the images in the dataset.

### Dataset

The dataset is organized in the `telur.v1i.yolov11` directory. The `data.yaml` file contains the configuration for the dataset, including the paths to the training, validation, and test images.

## License

This project is licensed under the terms of the `LICENSE` file.

## Acknowledgements

This project uses the Roboflow platform for dataset management and export. For more information, visit [Roboflow](https://roboflow.com).

For state-of-the-art Computer Vision training notebooks, visit [Roboflow Notebooks](https://github.com/roboflow/notebooks).

For more datasets and pre-trained models, visit [Roboflow Universe](https://universe.roboflow.com).
