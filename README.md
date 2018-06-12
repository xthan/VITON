## VITON: An Image-based Virtual Try-on Network
Code and dataset for the CVPR 2018 paper "VITON: An Image-based Virtual Try-on Network"

### Person representation extraction
The person representation used in this paper are extracted by a 2D pose estimator and a human parser:
* [Realtime Multi-Person Pose Estimation](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)
* [Self-supervised Structure-sensitive Learning](https://github.com/Engineering-Course/LIP_SSL)

### Dataset
The dataset is no longer publicly available due to copyright issues. For thoese who have already downloaded the dataset, please note that using or distributing it is illegal!

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
