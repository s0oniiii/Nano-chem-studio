# Nano-Chem Studio

A Streamlit web application designed for SEM image particle analysis.

## Features

**SEM Particle Analysis (Computer Vision Module)**:
An automated image processing module powered by OpenCV and NumPy. It performs semantic segmentation on Scanning Electron Microscope (SEM) images to quantify nanoparticle dispersion.

- Key Technologies:
Image Processing Pipeline: Implements Gaussian Blur for noise reduction and Otsu's Binarization for adaptive thresholding.
Contour Detection: Utilizes OpenCV (cv2.findContours) to calculate particle area and morphology.
Interactive Visualization: Features dynamic histograms using Plotly for particle size distribution analysis.
Data Engineering: Automates statistical calculation (Mean, Std Dev) and generates downloadable CSV reports using Pandas.


**3D Chemical Lab (Molecular Visualization Module)**:
A web-based 3D molecular viewer built with Streamlit, PubChemPy, and py3Dmol. This module fetches 3D coordinates (SDF format) via REST API and renders them interactively using WebGL.

- Key Features:
Real-time Data Fetching: Seamless integration with PubChem API for instant access to molecular structures.
Dynamic Legend Generation: Automatically detects unique elements in the molecule and generates a context-aware legend.
Interactive Analysis: Supports 3D rotation, zooming, and toggleable atom labels with standard CPK coloring.

## Installation 

1. Install required libraries:
```bash
pip install -r requirements.txt
```

## How to Run 

```bash
streamlit run app.py
```

The browser will open automatically. The default address is http://localhost:8501.

## Usage 

1. Select the "SEM Particle Analysis" menu from the left sidebar.
2. Upload a JPG or PNG image in the "Upload SEM Image" section.
3. The uploaded image will be processed automatically, and the particle count will be displayed.
