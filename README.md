
## iSAID: A Large-scale Dataset for Instance Segmentation in Aerial Images, CVPR workshops, 2019.

**Codes for Data Preparation and Evaluation**

1.  **Environment and dependencies installation**
    1. Create the conda environment
            ```conda env create -f environment.yml```
    2. Activate the current working environment
             ```source activate py_isaid```
    3. Setup pycocotols for the evalaution server
               - `cd cocoapi/PythonAPI`
               - `make`
               - `python setup.py install`
    4. Setup cityscapesScripts for the evalaution server
             - `cd preprocess/cityscapesScripts`
             - `python setup.py install`
    5. Setup detectron for the evalaution server
             - `cd preprocess/Detectron`
             - `make`
    6. Note: opencv version == 3.4.2
             
2.  **Data Preparation for Training, Validation and Testing**
    1. Please download iSAID dataset that contains image segmentation masks. Also, download original images from DOTA dataset. 
    Make sure that the final dataset must have this structure:
    ```
        iSAID
        ├── test
        │   └── images
        │       ├── P0006.png
        │       └── ...
        │       └── P0009.png
        ├── train
        │   └── images
        │       ├── P0002_instance_color_RGB.png
        │       ├── P0002_instance_id_RGB.png
        │       ├── P0002.png
        │       ├── ...
        │       ├── P0010_instance_color_RGB.png
        │       ├── P0010_instance_id_RGB.png
        │       └── P0010.png
        └── val
            └── images
                ├── P0003_instance_color_RGB.png
                ├── P0003_instance_id_RGB.png
                ├── P0003.png
                ├── ...
                ├── P0004_instance_color_RGB.png
                ├── P0004_instance_id_RGB.png
                └── P0004.png
    ```
    Note that the segmentation masks for the test images are withheld for the evaluation server.
    
    3. Change the current working directory to preprocess folder.
        ```cd preprocess```
    4. Create symlink for iSAID dataset as
        ```ln -s /path-of-iSAID-dataset ./dataset/```
    
    5. Split training and validation images into patches
        ```python split.py --set train,val```
    
    6. Split test images into patches
        ```python split.py --set test```
    
    7. Create coco-format json annotation files for train and val split images
        ```python preprocess.py --set train,val```


        
        Make sure that the final dataset after preprocesing must have this structure:

    ```
    iSAID_patches
    ├── test
    │   └── images
    │       ├── P0006_0_0_800_800.png
    │       └── ...
    │       └── P0009_0_0_800_800.png
    ├── train
    │   └── instance_only_filtered_train.json
    │   └── images
    │       ├── P0002_0_0_800_800_instance_color_RGB.png
    │       ├── P0002_0_0_800_800_instance_id_RGB.png
    │       ├── P0002_0_800_800.png
    │       ├── ...
    │       ├── P0010_0_0_800_800_instance_color_RGB.png
    │       ├── P0010_0_0_800_800_instance_id_RGB.png
    │       └── P0010_0_800_800.png
    └── val
        └── instance_only_filtered_val.json
        └── images
            ├── P0003_0_0_800_800_instance_color_RGB.png
            ├── P0003_0_0_800_800_instance_id_RGB.png
            ├── P0003_0_0_800_800.png
            ├── ...
            ├── P0004_0_0_800_800_instance_color_RGB.png
            ├── P0004_0_0_800_800_instance_id_RGB.png
            └── P0004_0_0_800_800.png
    ```
        
3. **Method**
    1. Run your instance segmentation method on patches and generate json file of predictions

4. **Evaluation**
    1. Change the current working directory to evaluate folder.
        ```cd ../evaluate```
    3. Given json of predictions and json of val set ground truth (obtained after preprocess.py), Compute Average Precision
        ```python evaluate.py ```
