# vanessa

VANESSA: VAihiNgEn Semantic SegmentAtion

## Instructions:
- You need to have installed the drivers of your GPU if you are using one.
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
