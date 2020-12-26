## Implementation

I use a fully convolutional network (FCN) style architecture with two trunks/inputs for the RGB and depth respectively. The encoder has 4 layers (32 filters of size 5, stride of 2 for the first 3 layers and stride 1 for the final layer). The features for the RGB and depth are concatenated channelwise and we interleave 2 conv layers with bilinear upsampling to construct the heatmaps.

The model is trained using the BCE (Binary cross entropy) loss, classifying the pixels with probabilities for pick.

Modalities: Both RGB and depth are used.

## Things to try

Given more time I would try the following things:

* Data preprocessing: Resize, crop the data
* Data augmentation: add some noise, rotations, brightness etc to augment datset
* Different architectures: try U-net architectures
* Hyperparameter Tuning: learning rate, conv layer params (num filters, etc)

## Running Instructions

### Train:

* modify params in train.py
* Run: `python train.py`

### Test:

* modify params in test.py
	* select which training run_name to load the model from
	* select which checkpoint epoch_number to load the model for
* Run: `python test.py`
* output is generated in the 'test' folder for corresponding run folder in:
	* `output/[run_name]/test`

## References

Zeng et al. "Robotic pick-and-place of novel objects in clutter with multi-affordance grasping and cross-domain image matching." 2018 IEEE international conference on robotics and automation (ICRA). IEEE, 2018.