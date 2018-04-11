# transIQA
Image quality assessment by transfer learning

09/04/18: setup a simple network with dataset


#### Dependencies:
Dlib 19.10 (for human face detection)

#### Version:
version_0.15(10/04/18):
1. Dataset: full use:
<br>dataset structure: ./pristine/, ./GB/GB1-5, ./GN/GN1-5,
 ./JP2K/JP2K1-5, ./JPEG/JPEG1-5
2. Source scripts:
<br> dataset.py: dataset, dataloader
<br> main.py: train
<br> tools.py: show image
<br> model.py: net model structure

Next Version:
1. Read image first (for efficiency when training)
<br> **opencv** is needed for _.jp2k_ image
2. Preparation: detect faces
<br> (1+(2: _dataset.py_ and _tools.py_
<br> When initializing _FaceScoreDataset_: detect face of pristine images and crop image.
<br> For the same time, crop the image of the other generated images
3. Training and testing MOS: train on small patches of one image
<br> for one batch image, averagely generate all small patches of the image and forward-backward