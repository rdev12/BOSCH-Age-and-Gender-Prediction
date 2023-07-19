# Bosch - Age and Gender Prediction

Given an input image or video, our model predicts the age (range) and gender for the identified people in each frame. First, it uses FASTER-RCNNs for body detection in each of the frame. Then, it makes use of ESPCN super-resolution to increase the quality of the detected persons followed by a custom model with RESNET-50 backbone to predict the age and gender attributes.

If the input is a video, it ouputs a video, a frame-by-frame image folder with each of the *bounding boxes of bodies* and *person id*. Along with it, a *predictions.csv* folder is outputed according to the prescribed format. (If the video is too large, you may choose to output only the csv folder - See Code for Instructions)

The project was developed and tested on Google Colab platform. It is recommended that GPU environment is used for faster training and prediction. The links for the original notebooks are mentioned in the following subsections.

## Instructions

Download the repository folder `Age_and_Gender_Prediction` containing the code

```
cd Age_and_Gender_Prediction
```

## Installing the dependencies

Install all the dependencies required for our code by running the following command

```
cd MP_BO_T4_CODE\Age and Gender Prediction
python -m venv bosch
cd bosch/Scripts
activate.bat
# return back to the original directory
cd ..
cd ..
```
![Screenshot 2022-03-25 222302](https://user-images.githubusercontent.com/20983723/160167592-4908a630-b93b-4c2c-95b3-a22035ee7e4d.jpg)

Install the necessary requirements in order using pip
```
pip install -r requirements.txt
pip install opencv-python
pip install -q opencv-contrib-python
pip install -q --upgrade opencv-python
pip install -q --upgrade opencv-contrib-python

```

### Possible Errors
1. Try running the above commands in Anaconda Prompt instead of shell
2. Make sure the video/image is put in the 'video' or 'img' folder and this name of the testing video file is entered exactly when predict.py is run and "Enter the filename" prompt is asked
3. Make sure the environment is activated and the packages mentioned above are installed in it
4. If doesn't run in the GPU, try with CPU! During the Prediction time, it will tell if it is using GPU or CPU

NOTE: If the predict.py or train.py doesn't work, follow the instructions to run it on Google Colab. We have trained and tested on Colab. It is highly recommended use Google Colab directly for prediction or training.

### Prediction

Download and place the weights from [here](https://drive.google.com/drive/folders/1aZSDp9loAU5viqccP0RRmyQKSCWOIi2G?usp=drive_link) under necessary folders: 'pedestrian_detection/pretrained_weights/pretrained.pt', 'exp_result/PETA/PETA/img_model/ckpt_max.pth'.

To get the prediction for any video or image, simply run the `Predict.py` file  and follow the instructions mentioned in it. It is recommended to use GPU for faster prediction.

If you want to make prediction for a video,follow the below metioned steps

1) Place the video to be predicted in the `./video/` folder

2. Run the following command :
   ```
   python Predict.py --type video
   ```
3. It will ask for a filename so paste the exact file name you want prediction for .
4. The prediction will be stored in the `./prediction/{Video Name}`. It contains
5. - Outputted video with bounding boxes 
   - `images` folder containing the frame by frame ID and bounding box 
   - `predictions.csv` the csv file in the desired format
NOTE: In the code, toggle the following variable to True if you want ouput with images and video for each frame
```
pred_frame_by_frame = False # Toggle it to False if frame by frame prediction image of the video is not required
pred_video = False # Toggle it to False if the outputted video with bounding box and person id is not required
```

![Screenshot 2022-03-25 223117](https://user-images.githubusercontent.com/20983723/160167648-265115cd-d8f2-4c57-bfdd-82d29cd2a8db.jpg)
NOTE: Ignore the subsequent warning. Wait for it to process all the frames of the video (this may take some time in CPU)

If you want to make prediction for a image,follow the below metioned steps

1) Place the video to be predicted in the `./img/` folder

2. Run the following command :

   ```
   python Predict.py --type image
   ```
   
### Train

Create a `data` directory in the main folder and place the PETA datset according to the given format:

```
Age and Gender Prediction/
    data/
        PETA/
            images/[00001.png...19000.png]
            dataset.pkl
```

`images` folder can be found in:  [PETA Google Drive](https://drive.google.com/open?id=1q4cux17K3zNBgIrDV4FtcHJPLzXNKfYG).
`dataset.pkl` can be found in: ([dataset.pkl](https://drive.google.com/open?id=1q4cux17K3zNBgIrDV4FtcHJPLzXNKfYG).)

To train our age-gender prediction and the body detection, run the `Train.py` file by the following command.

```
python Train.py
```

## Using Google Colab
## Instructions

Upload the repository folder `Age and Gender Prediction` containing the code on GDrive 

### Train
[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ShVtkHfyiKB_ogBFjqAQ-8IQABcGQC5a?usp=sharing)

To train the age-gender prediction and the body detection, run the `Train.ipynb` notebook.
Create a `data` directory in the main folder and place the PETA datset according to the given format: 
```
Age and Gender Prediction/
    data/
        PETA/
            images/[00001.png...19000.png]
            dataset.pkl
```
`images` folder can be found in:  [PETA Google Drive](https://drive.google.com/open?id=1q4cux17K3zNBgIrDV4FtcHJPLzXNKfYG).
`dataset.pkl` can be found in: ([dataset.pkl](https://drive.google.com/open?id=1q4cux17K3zNBgIrDV4FtcHJPLzXNKfYG).)

### Prediction
[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18XrC1buyXGjnzZywPAUmiqLTnbex9QTk?usp=sharing)

To get the prediction for any video or image, simply run the `Predict.ipynb` file and follow the instructions mentioned in it. It is recommended to use GPU for faster prediction.

1. Place the video to be predicted in the `./video/` folder
2. Run the 'Predict for Video' subsection in the `Predict.ipynb` Notebook and all the preceding cells
3. The prediction will be stored in the `./prediction/{Video Name}`. It contains
    - Outputted video with bounding boxes
    - `images` folder containing the frame by frame ID and bounding box
    - `predictions.csv` the csv file in the desired format

## References and Description

### Github References

- https://github.com/aajinjin/Strong_Baseline_of_Pedestrian_Attribute_Recognition
- https://github.com/fannymonori/TF-ESPCN
- https://github.com/Chang-Chia-Chi/Pedestrian-Detection

### Paper References

1. Jian Jia , Houjing Huang , Wenjie Yang , Xiaotang Chen , and Kaiqi Huang:Rethinking of Pedestrian Attribute Recognition: Realistic Datasets and A Strong Baseline.

This paper proposes the PETA dataset and gives a strong baseline based on ResNet50 for Pedestrian Attribute Recognition as a multi -label learning problem. Our architecture is a result of inspiration by their strong baseline.

2. Shuai Shao , Zijian Zhao,  Boxun Li, Tete Xiao, Gang Yu, Xiangyu Zhang, Jian Sun, Megvii Inc.  : CrowdHuman: A Benchmark for Detecting Humans in a Crowd.

This paper provides a benchmark for human body detection in crowds. Baseline detectors like Faster RCNN  were tested on their annotated dataset in this paper.This paper is referred to for detecting the full body .

3. Wenzhe Shi , Jose Caballero , Ferenc Huszar´  , Johannes Totz , Andrew P. Aitken , Rob Bishop , Daniel Rueckert , Zehan Wang : Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network

This paper uses  Efficient sub-pixel convolution network for increasing the resolution of the image in LR space unlike other deep learning models which do it in HR space. Features are extracted in LR space and super resolved to HR space without a deconvolution layer.
