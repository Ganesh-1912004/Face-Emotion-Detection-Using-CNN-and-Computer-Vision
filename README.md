# Face Emotion Detection using CNN and Computer Vision

This project implements a facial emotion detection system using Convolutional Neural Networks (CNN) and computer vision techniques. The model is trained to recognize seven different emotions: happy, sad, angry, surprise, neutral, disgust, and fear.

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## Features

- CNN-based emotion classification
- Real-time face detection using OpenCV
- Support for 7 different emotion categories
- Trained on a diverse dataset from Kaggle
- Data augmentation for improved model robustness

## Technologies Used

- Python
- TensorFlow / Keras for CNN implementation
- OpenCV for face detection and image processing
- NumPy for numerical operations
- Imgaug or similar library for data augmentation

## Project Structure

```
face-emotion-detection/
│
├── data/
│   ├── train/              # Original dataset files
│   └── test/        # Augmented dataset files
├── models/
│   └── best_model.h5      # Trained CNN model
├── src/
│   ├── facial_emotion_detection.py           # Script for training the CNN model 
│   └── test_video.py  # Script for real-time emotion detection
├── haarcascade_frontalface_default.xml  # OpenCV face detection model
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── .gitignore             # Git ignore file
```

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Ganesh-1912004/face-emotion-detection.git
   cd face-emotion-detection
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the `haarcascade_frontalface_default.xml` file from the OpenCV repository and place it in the project root directory.

## Usage

1. To train the model:
   ```
   python src/train.py
   ```

2. To run the real-time emotion detection:
   ```
   python src/test_video.py
   ```

## Dataset

The model was trained on a facial emotion dataset obtained from Kaggle. The dataset includes images of faces expressing seven different emotions: happy, sad, angry, surprise, neutral, disgust, and fear. Data augmentation techniques were applied to increase the diversity and size of the training set, improving the model's robustness and generalization capabilities.

## Model Performance

The CNN model was trained for 100 epochs using GPU acceleration, with data augmentation applied to the training set. The final model achieved an accuracy of 80% on the test set.

## Future Improvements

- Experiment with different CNN architectures
- Add support for emotion detection in video files
- Develop a user-friendly GUI for the application
- Implement ensemble methods to combine multiple models for improved accuracy

## Contributing

Contributions to this project are welcome. Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
