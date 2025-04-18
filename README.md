# Video to Music Retrieval (VMR)

## Description  
This repository contains scripts and code for video-to-music retrieval, including feature extraction and VMNET model adaptation.

---

## 1. Feature Extraction  
This folder includes several scripts to perform feature extraction as described in our paper

### **Feature Extraction Details**  
- **Video Features**: We use 3 different models.  Resnet, CLIP and I3D.  Resnet produces a 2048 vector/second of video.  CLIP produces a 512 length vector per second of video.  And i3d results in 1024 rgb feature vector and a 1024 rgb optical flow vector per second of video.
- **Music Features**: Uses VGGish to convert audio into a 128-dimensional vector per second.  

### **Included Files**  
- `downloadYt.py` - Uses `yt_dlp`, and a proxy rotation strategy to download youtube videos
- `flowProcessor.py` - This converts raw *.mp4 files to the flow based features described in our paper that are used for segmenting videos.
- `vggishProcessor.py` - This converts raw *.mp3, .mp4a and .webm files to the vggish features
- `clip.sl`, `flow.sl`, `i3d.sl`, `resnet.sl` - These are slurm scripts which take *.mp4 files and convert to video features.  We utilize this [library]([https://github.com/csehong/VM-NET](https://v-iashin.github.io/video_features/)) and followed their instructions to setup an environment.
- `HIMV200.txt` - List of YouTube URLs corresponding to the HIMV-200K.
- `SymMV.csv` - List of YouTube URLs corresponding to the SymMV-200K.

---

## 2. VMNET  
This folder contains code adapted from the official [VMNET repository](https://github.com/csehong/VM-NET). The original code was incomplete and required modifications, including dataloaders, to make it functional.  

### **How to Run VMNET**  

#### **a. Organizing Features**  
Organize the extracted features in the following structure:  

- **train_data_dir/**  
  - **audio/**  
    - `id.npy` (features of audio)  
  - **video/**  
    - `id.npy` (features of video)  

- **test_data_dir/**  
  - **audio/**  
    - `id.npy` (features of audio)  
  - **video/**  
    - `id.npy` (features of video)  


#### **b. Creating CSV Files**  
Create CSV files for training and testing, formatted as follows:  

```
ID
id_1
id_2
id_3
id_4
...
```

#### **c. Updating `train.py`**  
Modify the following flags in `train.py`:  
- `train_data_dir`  
- `train_csv_path`  
- `test_data_dir`  
- `test_csv_path`  

---

## 3. Notebooks  
This folder contains various Jupyter notebooks:  

- **`SegmentingDemo.ipynb`** - Playground for experimenting with different segmentation strategies.  This shows some of the files generated in the report 
- **`Transformer.ipynb`** - Notebook implementation of the full proposed model.  
- **`DataLoader.ipynb`** - Notebook implementation the data loader class to be used in the final script.  
