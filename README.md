# Dual-LVT: A Dual Attention Language-Vision Transformer for Tumor Segmentation - MICCAI 2025 (under review)

## Overview
Dual-LVT is a multimodal segmentation model that fully integrates large language models (LLMs) to enhance medical image analysis. It employs a dual attention mechanism, where language features from LLM-generated clinical notes serve as queries to guide segmentation, enabling a more effective fusion of vision and text. Additionally, the Adaptive Hounsfield Unit (AdaHU) Clipping Module dynamically adjusts CT preprocessing, improving robustness across imaging devices. Evaluated on the RADCURE CT dataset, Dual-LVT outperforms existing methods in accuracy and reliability.

## Dataset
Dual-LVT is evaluated on the RADCURE dataset, a publicly available collection of CT scans for head and neck tumor segmentation. The dataset includes:
- High-resolution CT scans from diverse imaging devices.
- Expert-annotated tumor segmentation masks for training and evaluation.
- Clinical notes from radiation oncologists, which are utilized by LLMs to improve segmentation accuracy.
For more information and to download the dataset, please visit the **[official RADCURE website](https://www.cancerimagingarchive.net/collection/radcure/)**.
