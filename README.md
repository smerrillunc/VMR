# Video to Music Retrieval (VMR)

## Description  
This repository contains scripts and code for video-to-music retrieval, including feature extraction and VMNET model adaptation.

---

## 1. Feature Extraction  
This folder includes scripts to perform feature extraction as described in [this paper](https://arxiv.org/abs/1609.08675).  

### **Feature Extraction Details**  
- **Video Features**: Uses Inception-v3 and PCA whitening to extract a 1024-dimensional vector per second.  
- **Music Features**: Uses VGGish to convert audio into a 128-dimensional vector per second.  

### **Included Files**  
- `FeatureExtraction.py` - Downloads videos using `yt_dlp`, converts audio to MP3, and extracts Inception and VGGish features.  
- `pca_transform.ipynb` - Converts 2048-dimensional Inception features to PCA-whitened 1024-dimensional features.  
- `Youtube_ID.txt` - List of YouTube URLs to download.  
- `submit_feature_extraction.py` - Submits multiple SLURM jobs to collect data in parallel.  

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

- **`SegmentingDemo.ipynb`** - Playground for experimenting with different segmentation strategies.  
- **`Transformer.ipynb`** - Notebook implementation of the full proposed model.  
