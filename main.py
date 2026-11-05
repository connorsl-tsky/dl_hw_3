import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import keras
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split

keras.mixed_precision.set_global_policy("mixed_float16")

def encoder_block(inputs, num_filters):
    """
    heavy inspiration from https://www.geeksforgeeks.org/machine-learning/u-net-architecture-explained/

    modified with claude
    """

    x = keras.layers.Conv2D(num_filters, 3, padding='same', kernel_initializer="he_normal")(inputs)
    x = keras.layers.Activation('relu')(x)
    
    x = keras.layers.Conv2D(num_filters, 3, padding='same', kernel_initializer="he_normal")(x)
    x = keras.layers.Activation('relu')(x)

    pooled = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)
    
    return x, pooled

def decoder_block(inputs, skip_features, num_filters):
    """
    heavy inspiration from https://www.geeksforgeeks.org/machine-learning/u-net-architecture-explained/
    """

    # up-conv
    x = keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(inputs)

    #concat 
    x = keras.layers.Concatenate()([x, skip_features])

    x = keras.layers.Conv2D(num_filters, 3, padding='same', kernel_initializer="he_normal")(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(num_filters, 3, padding='same', kernel_initializer="he_normal")(x)
    x = keras.layers.Activation('relu')(x)

    return x

def unet_2block(input_shape=(512, 512, 3), num_classes=1):
    """
    heavy inspiration from https://www.geeksforgeeks.org/machine-learning/u-net-architecture-explained/
    """

    inputs = keras.layers.Input(shape=input_shape)
    
    # Contracting Path
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)

    # unet paper says to do drop out after contracting path
    # for data augmentation
    drop1 = keras.layers.Dropout(rate=0.5)(p2)
    
    # Bottleneck
    b1 = keras.layers.Conv2D(256, 3, padding='same', kernel_initializer="he_normal")(drop1)
    b1 = keras.layers.Activation('relu')(b1)
    b1 = keras.layers.Conv2D(256, 3, padding='same', kernel_initializer="he_normal")(b1)
    b1 = keras.layers.Activation('relu')(b1)
    
    # Expansive Path
    d1 = decoder_block(b1, s2, 128)
    d2 = decoder_block(d1, s1, 64)
    
    outputs = keras.layers.Conv2D(num_classes, 1, padding='same', activation='sigmoid', dtype='float32')(d2)
    
    model = keras.models.Model(inputs=inputs, outputs=outputs, name='U-Net2')
    return model


def unet_3block(input_shape=(512, 512, 3), num_classes=1):
    """
    heavy inspiration from https://www.geeksforgeeks.org/machine-learning/u-net-architecture-explained/
    """

    inputs = keras.layers.Input(shape=input_shape)
    
    # Contracting Path
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)

    # unet paper says to do drop out after contracting path
    # for data augmentation
    drop1 = keras.layers.Dropout(rate=0.5)(p3)
    
    # Bottleneck
    b1 = keras.layers.Conv2D(256, 3, padding='same', kernel_initializer="he_normal")(drop1)
    b1 = keras.layers.Activation('relu')(b1)
    b1 = keras.layers.Conv2D(256, 3, padding='same', kernel_initializer="he_normal")(b1)
    b1 = keras.layers.Activation('relu')(b1)
    
    # Expansive Path
    d1 = decoder_block(b1, s3, 256)
    d2 = decoder_block(d1, s2, 128)
    d3 = decoder_block(d2, s1, 64)
    
    outputs = keras.layers.Conv2D(num_classes, 1, padding='same', activation='sigmoid',  dtype='float32')(d3)
    
    model = keras.models.Model(inputs=inputs, outputs=outputs, name='U-Net3')
    return model

def load_images(image_dir: str, image_size: tuple[int], isMask: bool) -> np.ndarray:
    """
    turn images into np array of images
    input directory containing images and size that you want to reshape the images into
    output array of images e.g. (20, 512, 512, 3)
    some code taken from https://www.geeksforgeeks.org/machine-learning/u-net-architecture-explained/
    but mostly my own creation
    """
    if isMask:
        images = np.array([np.ndarray((image_size[0],image_size[1]))]) # empty array, first entry is all 0s
    else:
        images = np.array([np.ndarray((image_size[0],image_size[1],3))]) # empty array, first entry is all 0s
    for image_file in os.scandir(image_dir):
        if not image_file.is_file():
            raise ValueError(f"{image_file} is not a file")
        if isMask:
            image = Image.open(image_file).resize(image_size, Image.NEAREST)
        else:
            image = Image.open(image_file).convert('RGB')
            image = image.resize(image_size)
        image = np.asarray(image)
        images = np.append(images, [image], axis=0)
    return images[1:].astype(np.uint8)

def binary_convert_threshold(arr, threshold):
    """
    used for masks, which are shape (batch, h, w, 1)
    """
    for batch in range(len(arr)):
        for h in range(len(arr[batch])):
            for w in range(len(arr[batch][h])):
                if arr[batch][h][w][0] > threshold:
                    arr[batch][h][w][0] = 1
                else:
                    arr[batch][h][w][0] = 0
    return arr

@keras.saving.register_keras_serializable()
def dice_bce_loss(y_true, y_pred):
    """
    custom loss function to incorporate dice in loss
    since bce can sometimes favor only picking negative since very few of the pixels are true
    """

    bce = keras.ops.mean(keras.losses.binary_crossentropy(y_true, y_pred))
    inter = keras.ops.sum(y_true * y_pred)
    dice = (2 * inter + 1) / (keras.ops.sum(y_true) + keras.ops.sum(y_pred) + 1)
    return bce + (1 - dice)


TRAIN = True
SAVE = True
SAVE_NAME = "unet2block512.keras"
BLOCK2 = True # 2block if true, 3block if false
IMAGE_SIZE = (512, 512)

if __name__ == "__main__":
    # PREPROCESSING

    # load images and resize
    img = load_images(image_dir="./Data/train/image", image_size=IMAGE_SIZE, isMask=False)
    
    # load masks and resize
    mask = load_images(image_dir="./Data/train/mask", image_size=IMAGE_SIZE, isMask=True)
    mask = mask.reshape(mask.shape[0], IMAGE_SIZE[0], IMAGE_SIZE[1], 1)

    # standardize

    img = (img / 255.0).astype("float32")
    mask = (mask > 127).astype("float32")

    # validation
    train_img, val_img, train_mask, val_mask = train_test_split(img, mask, test_size=0.2, random_state=42)

    # data augmentation
    # to both images and masks
    transform = keras.Sequential()
    transform.add(keras.layers.RandomRotation(factor=0.1))
    transform.add(keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1))
    transform.add(keras.layers.RandomElasticTransform())

    # this only gets done to the images, not the masks
    color = keras.Sequential()
    color.add(keras.layers.RandomBrightness(factor=0.2, value_range=(0,1)))
    color.add(keras.layers.RandomContrast(factor=0.2))


    # stolen form claude, see prompt 1 in prompts.md
    train_ds = (tf.data.Dataset.from_tensor_slices((train_img, train_mask))
                .shuffle(100)
                .batch(1))
    train_ds = train_ds.map(lambda x, y: (transform({"images": x, "segmentation_masks": y}, training=True)), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(lambda x: (x["images"], x["segmentation_masks"])) # separate back into x and y
    train_ds = train_ds.map(lambda x, y: (color(x, training=True), y))
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
 
    val_ds = tf.data.Dataset.from_tensor_slices((val_img, val_mask)).batch(1)



    # TRAINING

    if TRAIN:
        if BLOCK2:
            model = unet_2block(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
        else:
            model = unet_3block(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001), # need to add .99 momentum somehow
            loss=dice_bce_loss,
            metrics=[
                keras.metrics.BinaryAccuracy(),
                keras.metrics.TruePositives(thresholds=0.5), # need these for dice/iou
                keras.metrics.TrueNegatives(thresholds=0.5),
                keras.metrics.FalseNegatives(thresholds=0.5),
                keras.metrics.FalsePositives(thresholds=0.5)
            ]
        )
        model.summary()
        model.fit(train_ds, validation_data=val_ds, epochs=200, 
                  callbacks=[keras.callbacks.EarlyStopping(
                      monitor="val_loss",
                      restore_best_weights=True,
                      patience=15,
                  )])

        if SAVE:
            model.save(SAVE_NAME)
    else:
        model = keras.models.load_model(SAVE_NAME)



    # SUBMISSION

    # load test
    test = load_images(image_dir="./Data/test/image", image_size=IMAGE_SIZE, isMask=False)
    test = test / 255.0 # standardize

    # eval
    loss, acc, tp, tn, fn, fp = model.evaluate(val_ds)
    print(f"IOU: {tp / (tp + fp + fn)}, DICE: {2*tp / (2*tp + fp + fn)}, LOSS: {loss}, BIN ACC: {acc}, TP: {tp}, TN: {tn}, FN: {fn}, FP: {fp}")
    
    """
    Runs:

    want higher iou and i think higher DICE

    model.keras = LOSS: 0.7138576507568359, BIN ACC: 0.9064369201660156, IOU: 0.0, TP: 0.0, TN: 237617.0, FN: 24527.0, FP: 0.0
    model2.keras = LOSS: 0.1573638916015625, BIN ACC: 0.9406547546386719, IOU: 0.0, TP: 11079.0, TN: 235508.0, FN: 13448.0, FP: 2109.0 - i think i stopped standardizing the mask and kept it binary
    3model.keras = IOU: 0.43483525354076463, DICE: 0.6061117504155479, LOSS: 0.16268201172351837, BIN ACC: 0.9412422180175781, TP: 11851.0, TN: 234890.0, FN: 12676.0, FP: 2727.0 - first iteration of unet3. still kinda sucks

    first major tune - skip connections, dice loss, threshold, He weights, not loading train from test, onednn, float32
    unet2block.keras - IOU: 0.5979284699443967, DICE: 0.7483795190972508, LOSS: 0.4314207434654236, BIN ACC: 0.9562788009643555, TP: 68177.0, TN: 934554.0, FN: 21904.0, FP: 23941.0, 62 epochs
    unet3block.keras - 226pm or 235pm IOU: 0.5765706351961125, DICE: 0.7314237907575791, LOSS: 0.46918046474456787, BIN ACC: 0.9534645080566406, TP: 66444.0, TN: 933336.0, FN: 23637.0, FP: 25159.0 91 epochs 407pm (90m)
    okay 3 block isn't much better

    i think i need to do 512x512
    but i might not be able to do it tonight. only have 40m or so
    lets see if its possible

    wait my laptop has a gpu???
    """

    # predict
    yhat = model.predict(test)
    yhat = (np.array(yhat) > 0.5).astype(np.uint8) * 255
    for i in range(len(yhat)):
        pred = yhat[i].reshape(IMAGE_SIZE).astype(np.uint8) # from (256, 256, 1) to (256, 256)
        mask = Image.fromarray(pred).resize((512, 512), Image.NEAREST)
        mask.save(f"./predictions/{i}.png")

    # then run make_submission.py ./predictions
