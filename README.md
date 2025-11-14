# README
This is the Plant Disease Project, aimed at early detection of crop diseases in farms to prevent crop loss and increase yield. 

## Synthetic Dataset

### [SyntheticDatasetMaking.ipynb](https://github.com/tmastercoding/Plant-Disease-Project/tree/main/synthetic_plant_dataset):

This file creates artificial plant images for yolo training, by augmenting and placing leaf images from the PlantVillage dataset, over random backgrounds along with bounding boxes. 

Images from PlantVillage are systematically selected across the 38 classes, to ensure there would be minimal class imbalances.
The plant are then extracted by simple edge detection (to remove background), and is added to a cache.

A random background is selected. Random amount of leaves are selected from the leaf cache and augmentation is applied. The augmented leaves are placed at random locations of the background along with bounding boxes, this allows the artificial image to be automatically labeled. 

These artificial images are then compiled into a new dataset with specified file directory

### [Synthetic plant dataset (folder)](https://github.com/tmastercoding/Plant-Disease-Project/tree/main/synthetic_plant_dataset)

A sample synthetic dataset that was created from the SyntheticDatasetMaking.ipynb file, containing 10,000 labeled images in YOLO format.


### [Backgrounds (folder)](https://github.com/tmastercoding/Plant-Disease-Project/tree/main/backgrounds)

Contains 20 sample backgrounds used for the synthetic dataset (could be replaced with other ones)

## Website
### [Website Server](https://github.com/tmastercoding/Plant-Disease-Project/tree/main/websiteServer)

Contains two python files and a pretrained model

app-withNano: The website is hosted on your computer, processing is done on the jetson nano

app-withoutNano: The website is hosted on your computer, processing is done on the computer

### [websiteNano](https://github.com/tmastercoding/Plant-Disease-Project/tree/main/websiteNano)

Contains three python files and a pretrained model

app-EdgeWithCPU: A website is hosted with processing on the edge with the CPU on your jetson nano

app-EdgeWithGPU: A website is hosted with processing on the edge with the GPU on your jetson nano

app-EdgeWithServer: An API is hosted on the Nano that links with the app-withNano on your computer.

## [Runs](https://github.com/tmastercoding/Plant-Disease-Project/tree/main/runs)
Runs is a folder containing images of graphs and data of our YOLOv8 training. It also contains the yolo weights file called best.pt.

## How to Host Website:
1. Run app-EdgeWithServer on the Jetson Nano
2. Change NANO_API_URL variable to your 'nano IP address'/infer on app-withNano
3. Run app-withNano on your computer
4. Host for remote connection (outside of your network) with the command ngrok http 8000 on terminal.

## Credits:
- Github Repo - Tay Oh (tmastercoding)
- Prototyping with Keras - Tay Oh (tmastercoding), Braden Lau (Koogoshau)
- Attempts with PlantDoc - Tay Oh (tmastercoding), Braden Lau (Koogoshau)
- Synthetic Dataset - Tay Oh (tmastercoding), Braden Lau (Koogoshau)
- YOLO Training and Testing - Tay Oh (tmastercoding)
- Website without Nano - Harry Lam (adyinae1128)
- About.html - Gabriel Tsang
- Website with Nano - Tay Oh(tmastercoding)
- Nano Side API code - Tay Oh(tmastercoding)
- Nano Side YOLO inference - Tay Oh(tmastercoding)
- Server Hosting & Troubleshooting - Tay Oh(tmastercoding)
