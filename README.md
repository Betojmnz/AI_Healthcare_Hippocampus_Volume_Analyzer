# AI_Healthcare_Hippocampus_Volume_Analyzer
## Overview
Creation of an end-to-end AI system that integrates into a clinical-grade viewer and automatically measures hippocampal volumes of new patients to identify Alzheimer disease progression

## Section 1
Exploratory Data Analysis:
* Inspected a dataset of MRI scans and related segmentations,
* Represented as NIFTI files,
* Visualized image slices,
* and understood the layout of the data

## Section 2
Training of a Segmentation CNN:
* Functional code for training the segmentation model
* Test report with Dice scores on test set.
* Trained model PyTorch parameter file (model.pth)

## Section 3
Integrating into a Clinical Network:
* Code that runs inference on a DICOM volume and produces a DICOM report
* A report.dcm file with a sample report
* Screenshots of the report shown in the OHIF viewer
* Validation Plan
