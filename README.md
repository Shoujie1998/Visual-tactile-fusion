# Visual-tactile Fusion for Transparent Object Grasping in Complex Backgrounds
We propose TGCNN model, which is a generative architecture that takes in a 3-channel input RGB image and generates pixel-wise grasps in the form of two images. The 3-channel RGB image is passed through convolutional layers, residual layers and convolution transpose layers to generate two images.


## Dataset

It is difficult to collect and annotate dataset manually, especially for transparent objects. On the one hand, changes in lighting and background can cause significant changes in the appearance of transparent objects, which are sometimes difficult for even people to discern. on the other hand, it is difficult to obtain a large number of scenes in reality that include background and lighting changes.

To address this problem, a synthetic transparent objects multi-background grasping dataset containing 12,000 images is made, which is named SimTrans12K. SimTrans12K contains 6 types of objects and 2,000 different scenes, among which different scenes are obtained by intercepting movie frames.  In order to verify the detection effect in real scenes, we added 110 transparent objects with different backgrounds and 50 transparent objects with different brightnesses, each image contains 5-6 labels, totaling about 900 labels.



In the data set we published, in order to facilitate subsequent custom processing, we classified and placed the data of each object in the following format:


	├─Synthetic dataset
	│  ├─cup1
	│  │  ├─anno_mask
	│  │  ├─center
	│  │  └─rgb
	│  ├─cup2
	│  │  ├─anno_mask
	│  │  ├─center
	│  │  └─rgb
	│  ├─cup3
	│  │  ├─anno_mask
	│  │  ├─center
	│  │  └─rgb
	│  ├─cup4
	│  │  ├─anno_mask
	│  │  ├─center
	│  │  └─rgb
	│  ├─cup5
	│  │  ├─anno_mask
	│  │  ├─center
	│  │  └─rgb
	│  └─cup6
	│      ├─anno_mask
	│      ├─center
	│      └─rgb
	│
	├─Real dataset
	│  ├─Different backgrounds
	│  ├─Different brightnesses





In order to use our proposed network for training, the data needs to be reorganized into the following format:

	.
	├── test
	│   ├── anno_mask
	│   ├── center
	│   └── rgb
	└── train
	    ├── anno_mask
	    ├── center
	    └── rgb

By dividing different data sets, different tasks can be designed.

## Requirements

- imageio
- numpy
- opencv-python
- matplotlib
- scikit-image
- torch
- torchvision
- torchsummary
- tensorboardX
- Pillow


## Model Training

You can use the train_network.py file for network training.

Example for training:

```bash
python train_network.py --dataset-path <Path To Dataset> 
```

After the training is complete, you can see the results of the training under the Logs folder.
