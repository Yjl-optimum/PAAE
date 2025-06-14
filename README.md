# PAAE

## Environment
The code is developed under the following environment: [PoseFormerV2](https://github.com/QitaoZhao/PoseFormerV2)

## Rep-Fitness Dataset

We propose a clear, human-centric repetitive fitness action test dataset, **Rep-Fitness**. The videos are sourced from YouTube and Keep fitness platforms, and the dataset contains 141 action categories. You can access the Rep-Fitness Dataset from Google Drive via the following link:

[Rep-Fitness Dataset](https://drive.google.com/file/d/1GFPxQo5e5eQUy4h6_6-1K-rsZbKxF1Gq/view?usp=drive_link).(*For educational and research purposes only.*)

Click the link above to open the dataset on Google Drive.

<img src="image/category.png" alt="Categories Image" width="300"/>
<img src="image/Visual examples.png" alt="Example Image" width="600" height="auto"/>

## Skeleton Extraction and Diffusion-based Generation of Skeleton Sequences
Skeleton Extraction: [PoseFormerV2](https://github.com/QitaoZhao/PoseFormerV2)

[two-stage generation process](https://github.com/li-ronghui/LODGE)

Our program converts the SMPL format into the Human3.6M joint index as follows: ./Diffusion-based Generation of Skeleton Sequences/`process.py and smpl_process.py`

## Acknowledgment
Our training code is mainly based on [RepNet-Pytorch](https://github.com/confifu/RepNet-Pytorch/tree/main) and [TransRAC](https://github.com/SvipRepetitionCounting/TransRAC). 
