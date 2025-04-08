# MedSAM2
Segment Anything in 3D Medical Images and Videos

<div align="center">
 <table align="center">
   <tr>
     <td><a href="https://arxiv.org/abs/2504.03600" target="_blank"><img src="https://img.shields.io/badge/arXiv-Paper-FF6B6B?style=for-the-badge&logo=arxiv&logoColor=white" alt="Paper"></a></td>
     <td><a href="https://medsam2.github.io/" target="_blank"><img src="https://img.shields.io/badge/Project-Page-4285F4?style=for-the-badge&logoColor=white" alt="Project"></a></td>
     <td><a href="https://huggingface.co/wanglab/MedSAM2" target="_blank"><img src="https://img.shields.io/badge/HuggingFace-Model-FFBF00?style=for-the-badge&logo=huggingface&logoColor=white" alt="HuggingFace Model"></a></td>
   </tr>
   <tr>
     <td><a href="https://medsam-datasetlist.github.io/" target="_blank"><img src="https://img.shields.io/badge/Dataset-List-00B89E?style=for-the-badge" alt="Dataset List"></a></td>
     <td><a href="https://huggingface.co/datasets/wanglab/CT_DeepLesion-MedSAM2" target="_blank"><img src="https://img.shields.io/badge/Dataset-CT__DeepLesion-28A745?style=for-the-badge" alt="CT_DeepLesion-MedSAM2"></a></td>
     <td><a href="https://huggingface.co/datasets/wanglab/LLD-MMRI-MedSAM2" target="_blank"><img src="https://img.shields.io/badge/Dataset-LLD--MMRI-FF6B6B?style=for-the-badge" alt="LLD-MMRI-MedSAM2"></a></td>
   </tr>
   <tr>
     <td><a href="https://github.com/bowang-lab/MedSAMSlicer/tree/MedSAM2" target="_blank"><img src="https://img.shields.io/badge/3D_Slicer-Plugin-000000?style=for-the-badge" alt="3D Slicer"></a></td>
     <td><a href="https://github.com/bowang-lab/MedSAM2/blob/main/app.py" target="_blank"><img src="https://img.shields.io/badge/Gradio-Demo-F9D371?style=for-the-badge&logo=gradio&logoColor=white" alt="Gradio App"></a></td>
     <td><a href="https://colab.research.google.com/drive/1MKna9Sg9c78LNcrVyG58cQQmaePZq2k2?usp=sharing" target="_blank"><img src="https://img.shields.io/badge/Colab-Notebook-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Colab"></a></td>
   </tr>
 </table>
</div>

Welcome to join our [mailing list](https://forms.gle/bLxGb5SEpdLCUChQ7) to get updates. We’re also actively looking to collaborate on annotating new large-scale 3D datasets. If you have unlabeled medical images or videos and want to share them with the community, let’s connect!


## Installation 

- Create a virtual environment: `conda create -n medsam2 python=3.12 -y` and `conda activate medsam2` 
- Install [PyTorch](https://pytorch.org/get-started/locally/): `pip3 install torch torchvision` (Linux CUDA 12.4)
- Download code `git clone https://github.com/bowang-lab/MedSAM2.git && cd MedSAM2` and run `pip install -e ".[dev]"`
- Download checkpoints: `sh download.sh`
- Optional: Please install the following dependencies for gradio

```bash
sudo apt-get update
sudo apt-get install ffmpeg
pip install gradio==3.38.0
pip install numpy==1.26.3 
pip install ffmpeg-python 
pip install moviepy
```

## Download annotated datasets

- [CT_DeepLesion-MedSAM2](https://huggingface.co/datasets/wanglab/CT_DeepLesion-MedSAM2)



- [LLD-MMRI-MedSAM2](https://huggingface.co/datasets/wanglab/LLD-MMRI-MedSAM2) 

Note: Please also cite the raw [DeepLesion](https://doi.org/10.1117/1.JMI.5.3.036501) and [LLD-MMRI](https://www.sciencedirect.com/science/article/pii/S0893608025001078) dataset paper when using these datasets. 

- [RVENET](https://rvenet.github.io/dataset/): Waiting for authors' approval to release the mask.  


## Inference

### 3D medical image segmentation

- [CoLab](https://colab.research.google.com/drive/1MKna9Sg9c78LNcrVyG58cQQmaePZq2k2?usp=sharing): [MedSAM2_inference_CT_Lesion_Demo.ipynb](notebooks/MedSAM2_inference_CT_Lesion.ipynb)

- CMD

```bash
python medsam2_infer_3D_CT.py -i CT_DeepLesion/images -o CT_DeepLesion/segmentation
```

### Medical video segmentation

- [CoLab](https://colab.research.google.com/drive/1hyVyGh7qjTbFMuv568YcRoccdkNuDTaW?usp=sharing): [MedSAM2_Inference_Video_Demo.ipynb](notebooks/MedSAM2_Inference_Video.ipynb)


- CMD

```bash
python medsam2_infer_video.py -i input_video_path -m input_mask_path -o output_video_path 
```




### Gradio demo

```bash
python app.py
```

## Training

Specify dataset path in `sam2/configs/sam2.1_hiera_tiny_finetune512.yaml`

```bash
sbatch multi_node_train.sh
```

## Acknowledgements

- We highly appreciate all the challenge organizers and dataset owners for providing the public datasets to the community.
- We thank Meta AI for making the source code of [SAM2](https://github.com/facebookresearch/sam2) publicly available. Please also cite this paper when using MedSAM2. 


## Bibtex

```bash
@article{MedSAM2,
    title={MedSAM2: Segment Anything in 3D Medical Images and Videos},
    author={Ma, Jun and Yang, Zongxin and Kim, Sumin and Chen, Bihui and Baharoon, Mohammed and Fallahpour, Adibvafa and Asakereh, Reza and Lyu, Hongwei and Wang, Bo},
    journal={arXiv preprint arXiv:2504.03600},
    year={2025}
}
```
Please also cite SAM2
```
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}
```

