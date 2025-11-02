# README
This is the Plant Disease Project, aimed at early detection of crop diseases in farms to prevent crop loss and increase yield. 

## Synthetic Dataset

### SyntheticDatasetMaking.ipynb:

This file creates artificial plant images for yolo training, by augmenting and placing leaf images from the PlantVillage dataset, over random backgrounds along with bounding boxes. 

Images from PlantVillage are systematically selected across the 38 classes, to ensure there would be minimal class imbalances.
The plant are then extracted by simple edge detection (to remove background), and is added to a cache.

A random background is selected. Random amount of leaves are selected from the leaf cache and augmentation is applied. The augmented leaves are placed at random locations of the background along with bounding boxes, this allows the artificial image to be automatically labeled. 

These artificial images are then compiled into a new dataset with specified file directory

### Synthetic plant dataset (folder)

A sample synthetic dataset that was created from the SyntheticDatasetMaking.ipynb file, containing 10,000 labeled images in YOLO format.

