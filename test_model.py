import keras
from main import load_images
import numpy as np
import os
from PIL import Image



TEST_IMAGE_PATH = "./Data/test/image"
MODEL_FILE = "unet2.keras" # this is my best one

def test_model(model_file):
    # load model and images
    model = keras.models.load_model(model_file)
    test = load_images(image_dir=TEST_IMAGE_PATH, image_size=(512, 512), isMask=False)
    test = test / 255.0 # standardize

    # predict
    yhat = model.predict(test)
    yhat = (np.array(yhat) > 0.5).astype(np.uint8) * 255

    test_names = sorted(os.listdir(TEST_IMAGE_PATH), key=lambda n:int(os.path.splitext(n)[0]))

    # create directory if it doesn't exist
    if not os.path.exists("./predictions"):
        os.makedirs("./predictions")

    # save images into 'predictions' directory
    for name, p in zip(test_names, yhat):
        pred = p.reshape((512, 512)).astype(np.uint8) # from (256, 256, 1) to (256, 256)
        mask = Image.fromarray(pred).resize((512, 512), Image.NEAREST)
        mask.save(f"./predictions/{name}")


if __name__ == "__main__":
    test_model(MODEL_FILE)