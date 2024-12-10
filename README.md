# IMU-Video-Upper-Limb-Exoskeleton
This repository is a multimodal user intention prediction algorithm based on MoViNet and LSTM. 
# Dataset
The dataset for user intention prediction with IMUs and a Camera is provided at the following link:
https://uweacuk-my.sharepoint.com/:u:/g/personal/mohammadhadi_sarajchi_uwe_ac_uk/Ea0sPpPfUBNEi2MIu_ilGQUBgd7g7J9-S4oxE1R0-LXdWw?e=rKdLnW
or
https://drive.google.com/file/d/1tn91HX9y28Xy6x3cmq3V3b6SHgiwPRpx/view?usp=sharing

# Sliding Window
A sliding window of length 10 is used for Video and IMU data. They move forward in both the Video stream and the IMU data stream to ensure that the data fed into the model is synchronised and maintains the same length.
![Fig2_New13 (2)](https://github.com/user-attachments/assets/b0b32010-b240-471b-9919-28a564c2e7b2)

# Model
A new multimodal user intention prediction model for an upper limb exoskeleton is proposed, which uses MoVinet to process the video data and LSTM to extract features from the IMU data, and finally fuses the above two features to classify the three actions of grasping, walking and placing during the process of carrying a box.
![Fig4 (1)](https://github.com/user-attachments/assets/4c561323-f942-4d44-ad9c-bb4513139d40)

# Supplementary Material:
A video demonstrating the study is available at the following link: https://youtu.be/PU3ySZ0spoI.
# Start
  pip install -r requirements.txt
  
  python train_eval.py
