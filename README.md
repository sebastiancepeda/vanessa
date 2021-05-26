# vanessa

VANESSA: VAihiNgEn Semantic SegmentAtion

This was tested in a system with GPU NVIDIA GeForce GTX 1070 and Ubuntu 16.04.7.

## Instructions:
- You need to have installed the drivers of your GPU if you are using one.
- You need to create a folder called 'vaihingen/', and create there two folders:
  - 'images/': And copy there all the images (top_mosaic_09cm_area1.tif, ..., top_mosaic_09cm_area38.tif) 
  - 'labels/': And copy there all the labels (top_mosaic_09cm_area1.tif, ..., top_mosaic_09cm_area38.tif)
- Also, you need to copy the included 'sets.csv' file, to the 'vaihingen/' folder.
- Then, you need to edit the file "segmentation/training_pixel_tile_seg_model.py", configuring the variable 'path', 
  in the line 115 to the path of the folder 'vaihingen/', for example: 'path = "/home/sebastian/vaihingen"'.
- The script "install.sh" creates a python virtual environment and installs in 
  it all the dependencies in the requirements.txt file.
- The script "run_tests.sh" activates the virtual environment and runs the 
  tests of the project.
- The script "train_pixel_tile_level.sh"  activates the virtual environment 
  and runs the python script "segmentation/training_pixel_tile_seg_model.py", 
  which runs two models to compare between them:
  - A pixel level segmentation model that uses a U-Net like structure.
  - A pixel level segmentation model that uses a U-Net like structure, but 
    with a second output for pretraining with patch level data, to be 
    able to leverage image data with a more coarse labeled information.
