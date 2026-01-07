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

📁 Project Structure
text
Self-Driving-Car-Behavioral-Cloning/
├── data/                          # Dataset info / samples
│   ├── IMG/                       # Sample images
│   └── driving_log.csv            # Sample driving log
├── notebooks/                     # Jupyter notebooks
│   └── model.ipynb                # Complete model development notebook
├── src/                           # Source code
│   ├── train.py                   # Training script
│   ├── drive.py                   # Driving script for simulator
│   ├── model.py                   # Model architecture definition
│   ├── utils.py                   # Utility functions
│   └── preprocessing.py           # Data preprocessing functions
├── models/                        # Trained models
│   └── model.h5                   # Trained model weights
├── outputs/                       # Demo videos
│   └── demo.mp4                   # Autonomous driving demo
├── requirements.txt               # Dependencies list
├── LICENSE                        # MIT License
└── README.md                      # Project documentation
⚙️ Installation & Setup
1. Clone the Repository
bash
git clone https://github.com/rohitR87/Self-Driving-Car-Behavioral-Cloning.git
cd Self-Driving-Car-Behavioral-Cloning
2. Create Virtual Environment (Optional but Recommended)
bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For Linux/Mac
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
bash
pip install -r requirements.txt
Note: TensorFlow 1.15 requires Python 3.7. If you have a different Python version, consider using a virtual environment with Python 3.7.

▶️ How to Run the Project
Step 1 – Train the Model
If you want to train the model with your own data:

Collect data using the Udacity simulator in Training Mode

Place your data in the data/ directory

Run the training script:

bash
python src/train.py
Alternatively, you can use the pre-trained model in the models/ directory.

Step 2 – Download and Run the Simulator
Download the Udacity Self-Driving Car Simulator from the official repository

Choose the appropriate version for your OS

Run the simulator and select Autonomous Mode

Step 3 – Start Driving Script
bash
python src/drive.py models/model.h5
The car will now drive autonomously using the trained model.

🎥 Demo
A demo video of the autonomous driving performance is available in the outputs/ folder. You can also view it here.

📈 Results
The trained model achieves:

Smooth lane following on straight roads

Stable steering on curves without oscillation

No zig-zag behavior

Good generalization on unseen track sections

Real-time inference at 30+ FPS

Training Metrics:

Training Loss: 0.012

Validation Loss: 0.015

Inference Time: ~30ms per frame

🚀 Future Improvements
Advanced Architectures

Implement DenseNet or EfficientNet-based models

Add attention mechanisms for better focus on road features

Data Enhancement

Collect data from more challenging scenarios (sharp turns, intersections)

Add weather and lighting variations

Include recovery maneuvers

Model Improvements

Implement ensemble learning with multiple models

Add temporal information using LSTM/GRU layers

Incorporate speed as an additional input

Simulation Environment

Migrate to advanced simulators like CARLA or MetaDrive

Add traffic sign and obstacle detection

Implement multi-agent scenarios

Real-world Applications

Transfer learning to real-world driving datasets

Deploy on embedded systems (NVIDIA Jetson, Raspberry Pi)

Integrate with ROS (Robot Operating System)

👨‍💻 Author
Rohith R

Final Year Computer Science & Engineering Student

Specialization: AI / Machine Learning / Computer Vision

GitHub: rohitR87

LinkedIn: Rohith R

📌 Note
This project is built purely for educational and research purposes using the Udacity simulator. No real-world driving deployment is involved. The model is trained on simulated data and may not perform well in real-world conditions without further training and safety considerations.

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the repository

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Udacity for providing the Self-Driving Car Simulator

NVIDIA for the inspiration from their end-to-end driving model

The open-source community for various tools and libraries



