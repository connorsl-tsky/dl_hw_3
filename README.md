# dl_hw_3
medical image segmentation

kaggle: https://www.kaggle.com/competitions/hw-3-uc-2026/overview

# How To Run Model

`python ./main.py` -> for training and testing  
`python ./test_model.py` -> purely for testing

There are several control constants in main.py to dictate behavior:
- `IMAGE_SIZE` - Determines what size the image is reshaped to before testing. I would leave it at (512, 512) for the best accuracy, but maybe change it to (256, 256) if you have memory issues.
- `SAVE_NAME` - Is the name of the model to file to either save/load from. See entries for `SAVE` and `LOAD`
- `TRAIN` - This will train the model on the testing and validation datasets. 
- `SAVE` - This will save the model - after training - to the specified `SAVE_NAME` file. This value is ignored when `TRAIN` is set to `False`. 
- `BLOCK2` - Set this to `True` if you want to train a U-Net 2 Block model, and `False` if you want to train a U-Net 3 BLock model. This value is ignored if `Train` is set to `False` or if `LOAD` is set to `True`.
- `LOAD` - This is particularly used with checkpoints, so as to continue training. It is to load a model from a checkpoint instead of building a new one. The file it will look for is `"check_"+SAVE_NAME`. This value is ignored if `TRAIN` is set to `False`

These constants are located above the `if __name__ != "__main__"` line.

Note that when training, the model will create checkpoints denoted as `"check_+SAVE_NAME"` after each epoch. It does not create a separate file after each epoch, but overwrites the same file over and over. This was used to pause and continue training if necessary.

running `test_model.py` will load the `MODEL_FILE` .keras file located at the top of the file, and output predictions from the testing set to a `./predictions` directory. If there is no `./predictions` directory, it will make one. Change the `TEST_IMAGE_PATH` variable to determine where to load the image files from.