## VITON: An Image-based Virtual Try-on Network
Code and dataset for the CVPR 2018 paper "VITON: An Image-based Virtual Try-on Network"

### Person representation extraction
The person representation used in this paper are extracted by a 2D pose estimator and a human parser:
* [Realtime Multi-Person Pose Estimation](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)
* [Self-supervised Structure-sensitive Learning](https://github.com/Engineering-Course/LIP_SSL)

### Dataset

The dataset can be downloaded on [Google Drive](https://drive.google.com/drive/folders/1-RIcmjQKTqsf3PZsoHT4hivNngx_3386?usp=sharing).

This dataset is crawled from women's tops on [Zalando](https://www.zalando.co.uk/womens-clothing-tops/). These images can be downloaded on Google Drive. The results of pose estimation and human parsing are also included. Note that number of the images/poses/segmentation maps are more than that reported in the paper, since the ones with bad pose estimations (too few keypoints are detected) or parsing results (parsed upper clothes regions only occupy a small portion of the image).

Put all folder and labels in the ```data``` folder:

```data/women_top```: reference images (image name is ID_0.jpg) and clothing images (image name is ID_1.jpg). For example, the clothing image on reference image 000001_0.jpg is 000001_1.jpg. The resolution of these images is 1100x762.

```data/pose.pkl```: a pickle file containing a dictionary of the pose keypoints of each reference image. Please refer to [this demo](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/blob/master/testing/python/demo.ipynb) for how to parse the stored results, and [OpenPose output](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md) to understand the output format.

```data/pose/```: folder containing the pose keypoints of each reference image.

```data/segment```: folder containing the segmentation map of each reference image. In a segmentation map, label 5 corresponds to the regions of tops (used as pseudo ground truth of clothing region). label 1 (hat), 2 (hair), 4 (sunglasses), and 13 (face) are merged as the face/hair representation. All other non-background regions are merged for extracting the body representation. The colormap for visualization can be downloaded [here](https://github.com/Engineering-Course/LIP_SSL/blob/master/human_colormap.mat). Due to padding operations of the parser, these segmentation maps are 641x641, you need to crop them based on the aspect ratio of the original reference images.

```data/tps/```: TPS control points between product image and its corresponding reference image.

```data/viton_train_images.txt```: training image list.

```data/viton_train_pairs.txt```: 14,221 reference image and clothing training image pairs.

```data/viton_test_pairs.txt```: 2,032 reference image and target clothing testing image pairs. Note that these pairs are only used for the evaluation in our paper, one can choose any reference image and target clothing to generate the virtual try-on results.

### Test

#### First stage
Download pretrained models on [Google Drive](https://drive.google.com/drive/folders/1qFU4KmvnEr4CwEFXQZS_6Ebw5dPJAE21?usp=sharing). Put them under ```model/``` folder.

Run ```test_stage1.sh``` to do the inference.
The results are in ```results/stage1/images/```. ```results/stage1/index.html``` visualizes the results.

#### Second stage

Run the matlab script ```shape_context_warp.m``` to extract the TPS transformation control points.

Then ```test_stage2.sh``` will do the refinement and generate the final results, which locates in ```results/stage2/images/```. ```results/stage2/index.html``` visualizes the results.


### Train

#### Prepare data
Go inside ```prepare_data```. 

First run ```extract_tps.m```. This will take sometime, you can try run it in parallel or directly download the pre-computed TPS control points via Google Drive and put them in ```data/tps/```.

Then run ```./preprocess_viton.sh```, and the generated TF records will be in ```prepare_data/tfrecord```.


#### First stage
Run ```train_stage1.sh```

#### Second stage
Run ```train_stage2.sh```


<!---
### Todo list
- [x] Code of testing the first stage.
- [x] Data preparation code.
- [x] Code of training the first stage.
- [x] Shape context matching and warping.
- [x] Code of testing the second stage.
- [x] Code of training the second stage.
-->

### Citation

If this code or dataset helps your research, please cite our paper:


    @inproceedings{han2017viton,
      title = {VITON: An Image-based Virtual Try-on Network},
      author = {Han, Xintong and Wu, Zuxuan and Wu, Zhe and Yu, Ruichi and Davis, Larry S},
      booktitle = {CVPR},
      year  = {2018},
    }
