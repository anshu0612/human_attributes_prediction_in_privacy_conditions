### ACM MM'21: Human Attributes Prediction UNder Privacy-preserving Conditions

We built the Context-guided Human Attributes Prediction Network (CHAPNet) guided by out human findings. We utilized the psychophysics observations for emotion, age, and gender prediction to design CHAPNet, an end-to-end multi-tasking human attributes classification depp learning model. The advantage of basing our model design on human behaviour is that it makes the network architecture explainable.

![Model](https://chapnetgit.s3.ap-southeast-1.amazonaws.com/Model_v3.jpg)

<!-- <div align="center">
<img src="https://chapnetgit.s3.ap-southeast-1.amazonaws.com/Model_v3.jpg" width="100%" height="100%">
</div> -->

Paper: Human Attributes Prediction Uuder Privacy-preserving Conditions, accepted at [ACM Multimedia 2021](https://2021.acmmm.org/#)

--------------------------------------------------------------------------------

## Content

<!-- toc -->
- [Arguments](#arguments)
- [Dependencies](#features)
- [](#)
- [Contact](#contact)
- [References](#references)
<!-- - [Citation](#citation) -->
- [License](#license)
<!-- tocstop -->

## Usage

Training

```bash
#training with default settings on gpu_device 1 if GPU is available
python3 train.py --gpu_device 1

#training on images with head obfuscation for all people   
python3 train.py --ob_face_region heads --ob_people AO 

#training on images with all the faces obfuscated (default ob_face_region argument = 'face')
python3 train.py --ob_people AO 

#training on images with all the faces obfuscated (default ob_face_region argument = 'face')
python3 train.py --ob_people AO 

```

## Arguments

| Argument | Description | Default
| ---- | --- | --- |
| num_epochs | Set the number of epochs | 40 |
| batch_size | Set the batch size | 16 |
| lr | Set the initial learning rate | 0.01 |
| weight_decay | Set the weight decay in the range [0, 1] | 5e-4 |
| ob_face_region | Set the face region to obfuscated. Valid values are { None, eye, lower, face, head}  | None |
| ob_people | Set whether to obfuscate all the detected faces (AO) or only the targets (TO). Valid values are { None, TO, AO } | None |
| gpu_device | Set the GPU device to train the model on | 0 |

## Setup

### Expected directory structure of the data

```
├── data
│   ├── data.json
│   ├── splits.json
│   ├── privacy
│   |    ├── eye 
|   |    |     ├── images 
|   |    |     ├── pose
│   |    ├── lower
|   |    |     ├── images 
|   |    |     ├── pose
│   |    ├── face
|   |    |     ├── images 
|   |    |     ├── pose
│   |    ├── head 
|   |    |     ├── images 
|   |    |     ├── pose
│   ├── intact
│   |    ├── images
│   |    ├── pose
├──models
├──utils
......
```

### Diversity in Context and People Dataset (DPaC)

Out dataset with images containing obfuscated faces are available [online](https://bit.ly/3ak6uVE). Dataset with images of intact faces can be provided upon request.

### Pose generation 

Pose folder expects .npy file of pose guided heatmaps for each image.

Steps to  generate them:  

1. For cropped target pose landmarks generation use:

https://github.com/CMU-Perceptual-Computing-Lab/openpose

It will generate the pose landmarks json 

2. For heatmaps generation use the use the generate_heatmaps.py file.

```bash
python3 generate_heatmaps.py --cropped_targets_imgs_path "/targets/" --pose_data_path '/pose_landmarks.json' --save_path '/pose/' 
```

### Pose heatmaps generation

## Dependencies

PyTorch: > 1.7.1
Python: > 3.6

Testing

```bash
python3 test.py --cp_path "//" --gpu_device 0
```





<!-- 
## Citation
If you find this work or code is helpful in your research, please cite our work:
```
@inproceedings{wang2020score,
  title={Score-CAM: Score-weighted visual explanations for convolutional neural networks},
  author={Wang, Haofan and Wang, Zifan and Du, Mengnan and Yang, Fan and Zhang, Zijian and Ding, Sirui and Mardziel, Piotr and Hu, Xia},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops},
  pages={24--25},
  year={2020}
}
``` -->

## Contact

If you have any questions, feel free to open an issue or directly contact me via: `anshu@nus.edu.sg OR anshu@comp.nus.edu.sg`

## References

**Pose-guided target branch inspired by**:
Miao, Jiaxu, et al. "Pose-guided feature alignment for occluded person re-identification." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.

## License

PyTorch has a BSD-style license, as found in the [LICENSE](LICENSE) file.
