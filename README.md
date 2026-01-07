\# Self-Driving Car – Behavioral Cloning using CNN (Udacity Simulator)



This project implements a \*\*CNN-based behavioral cloning model\*\* to predict steering angles in real-time using images from the Udacity self-driving car simulator. The model learns driving behavior from human demonstration and enables smooth, stable autonomous lane navigation.



This is an end-to-end deep learning project covering \*\*data collection, preprocessing, model design, training, and deployment in a simulator environment.\*\*



---



\## 🚗 Project Overview



Behavioral cloning is a supervised learning approach where a model is trained to imitate human driving behavior. In this project, images from the center, left, and right cameras are used as input, and the corresponding steering angles are used as labels.



The trained model is capable of:

\- Following lanes smoothly

\- Handling curves without oscillation

\- Generalizing to unseen track sections



---



\## 🎯 Key Features



\- End-to-end CNN architecture (NVIDIA model inspired)

\- Trained on \*\*90,000+ driving images\*\*

\- Uses \*\*center, left, and right camera images\*\*

\- Data augmentation for better generalization

\- Real-time deployment in Udacity simulator



---



\## 🧠 Model Architecture



The model is based on NVIDIA’s end-to-end self-driving car architecture.



\*\*Architecture Summary:\*\*

\- Input: 160x320 RGB images

\- Cropping layer (remove sky \& car hood)

\- Normalization layer

\- 5 Convolutional layers with ReLU activation

\- Fully connected layers

\- Dropout for regularization

\- Output: Single steering angle value



This architecture is effective for learning spatial features and road patterns directly from images.



---



\## 📊 Data Preprocessing \& Augmentation



To improve performance and reduce overfitting, the following preprocessing techniques were applied:



\### 1. Cropping

Removed irrelevant regions (sky and car hood) to focus only on the road.



\### 2. Normalization

Scaled pixel values to the range \[-0.5, 0.5] to improve convergence.



\### 3. Data Augmentation

\- Horizontal flipping

\- Brightness adjustment

\- Steering angle correction for left \& right cameras

\- Random translation



These techniques help the model generalize better and handle real-world variations.



---



\## 🗂 Dataset



The dataset was generated using the Udacity simulator in training mode.



Each record contains:

\- Center image path

\- Left image path

\- Right image path

\- Steering angle

\- Throttle

\- Brake

\- Speed



Due to size limitations, the full dataset is not uploaded to GitHub.



---



\## 🛠 Tech Stack



\- \*\*Python 3.7\*\*

\- \*\*TensorFlow 1.15\*\*

\- \*\*Keras 2.2.4\*\*

\- OpenCV

\- NumPy

\- SciPy

\- Pillow

\- Flask (for simulator communication)

\- Eventlet \& SocketIO



---



\## 📁 Project Structure





