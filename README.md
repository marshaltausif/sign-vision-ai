# SignVisionAI

**Real-time Sign Language Detection and Recognition System**

---

## Overview

SignVisionAI is an advanced real-time sign language detection system that uses computer vision and machine learning to translate hand gestures into readable text. This project aims to bridge communication gaps for the hearing impaired by capturing sign language gestures via webcam or video input and processing them into accurate, context-aware text output.

Unlike some projects, SignVisionAI does **not** rely on MediaPipe for hand tracking. Instead, it uses custom computer vision techniques combined with machine learning models to detect and recognize hand gestures.

---

## Features

- Real-time detection and recognition of sign language alphabets and words.
- Custom hand detection and gesture recognition without external hand-tracking libraries.
- Smooth, stable gesture recognition with a configurable delay to ensure accurate word formation.
- User-friendly web interface built with **Streamlit** for easy interaction and visualization.
- Clear/reset functionality to start new word input.
- Lightweight Python-based implementation using OpenCV and ML frameworks.
- Modular, extensible code structure for easy improvements.

---

## How It Works

1. **Video Capture:** The application captures live video input from a webcam.

2. **Hand Detection:** Using image processing techniques (like skin color segmentation, contour detection, and shape analysis), the system isolates the hand region from each video frame.

3. **Feature Extraction:** From the detected hand region, relevant features are extracted (e.g., finger positions, contours, key shape descriptors).

4. **Gesture Classification:** Extracted features are fed into a trained machine learning model (e.g., CNN or other classifier) that predicts the corresponding sign language alphabet or gesture.

5. **Temporal Analysis:** To avoid misclassification due to quick movements or noise, the system checks if the same sign is held steadily for a configurable time period (e.g., 4 seconds) before confirming it.

6. **Word Formation:** Confirmed signs are appended to build words, shown on the interface.

7. **User Controls:** Users can clear the current input word to start fresh.

---

## Technologies Used

- **Python** — Core programming language  
- **OpenCV** — For real-time video capture and image processing  
- **scikit-learn / TensorFlow / PyTorch/resnet18/CNN** — For building and running the gesture classification model  
- **NumPy** — Numerical computations and array manipulations  
- **Streamlit** — For building the web-based user interface  

---

## Contribution

Contributions are welcome! If you'd like to improve SignVisionAI, please feel free to:

- Fork the repository
- Create a new feature branch
- Make your changes and improvements
- Submit a pull request for review

You can also report issues or suggest features using GitHub Issues.

---

## Contact

For questions, suggestions, or collaboration, reach out to:

**Tausif Abdullah Md**  
Email: [juniorscientist3@gmail.com](mailto:juniorscientist3@gmail.com)



