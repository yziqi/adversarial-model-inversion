# Adversarial Model Inversion Attack

This repo provides an example of the adversarial model inversion attack in the 
paper ["Neural Network Inversion in Adversarial Setting via Background Knowledge Alignment"](https://dl.acm.org/citation.cfm?id=3354261)

## Data

The target classifier (identity classification) is trained on the [FaceScrub](http://vintage.winklerbros.net/facescrub.html) 
dataset, and the adversary will use the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset as background 
knowledge to train the inversion model. 

#### Download

[FaceScrub](http://vintage.winklerbros.net/facescrub.html), [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

#### Extract Face

[FaceScrub](http://vintage.winklerbros.net/facescrub.html): Extract the face of each image according to the official 
bounding box information.

[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html): To extract the face of each image, crop the official 
align-cropped version (size 178 × 218) by width and height of 108 from upper left coordinate (35, 70). Please contact 
the authors of CelebA for details about the face identities, and then "clean" the CelebA by removing celebrities that 
are included in FaceScrub.

Transform both datasets to greyscale images with each pixel value in [0, 1]. Resize both datasets to 64 × 64.

## Setup

The code is written in Python3. You can install the required packages by running:

```
$ pip3 install -r requirements.txt
```

## Run

Train the target classifier:

```
$ python3 train_classifier.py
```

Train the inversion model:

```
$ python3 train_inversion.py
```

You can set the truncation size by the `--truncation` parameter.

## Citation

```
@inproceedings{Yang:2019:NNI:3319535.3354261,
 author = {Yang, Ziqi and Zhang, Jiyi and Chang, Ee-Chien and Liang, Zhenkai},
 title = {Neural Network Inversion in Adversarial Setting via Background Knowledge Alignment},
 booktitle = {Proceedings of the 2019 ACM SIGSAC Conference on Computer and Communications Security},
 series = {CCS '19},
 year = {2019},
 isbn = {978-1-4503-6747-9},
 location = {London, United Kingdom},
 pages = {225--240},
 numpages = {16},
 url = {http://doi.acm.org/10.1145/3319535.3354261},
 doi = {10.1145/3319535.3354261},
 acmid = {3354261},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {deep learning, model inversion, neural networks, privacy, security},
}
```
