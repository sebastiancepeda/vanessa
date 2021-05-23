export PYTHONPATH=$(pwd) && \
echo PYTHONPATH: $PYTHONPATH && \
source venv/bin/activate && \
python segmentation/training_pixel_seg_model.py
