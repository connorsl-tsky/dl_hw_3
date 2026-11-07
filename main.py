import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import keras
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split


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
    b1 = keras.layers.Conv2D(1024, 3, padding='same', kernel_initializer="he_normal")(drop1)
    b1 = keras.layers.Activation('relu')(b1)
    b1 = keras.layers.Conv2D(1024, 3, padding='same', kernel_initializer="he_normal")(b1)
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
    b1 = keras.layers.Conv2D(1024, 3, padding='same', kernel_initializer="he_normal")(drop1)
    b1 = keras.layers.Activation('relu')(b1)
    b1 = keras.layers.Conv2D(1024, 3, padding='same', kernel_initializer="he_normal")(b1)
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
    # thank you claude
    files = sorted(
        (f for f in os.scandir(image_dir)),
        key=lambda f: int(os.path.splitext(f.name)[0])
    )
    for image_file in files:
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

IMAGE_SIZE = (512, 512)
SAVE_NAME = "unet2.keras"
TRAIN = False
SAVE = True # save model at end, ignored when not training
BLOCK2 = True # 2block if true, 3block if false, ignored when not training, and when load is true
LOAD = False # load model from checkpoints, ignored when not training

if __name__ == "__main__":
    print(f"IMAGE_SIZE: {IMAGE_SIZE}, SAVE_NAME: {SAVE_NAME}, TRAIN: {TRAIN}, SAVE: {SAVE}, BLOCK2: {BLOCK2}, LOAD: {LOAD}")

    # PREPROCESSING

    # load images and resize
    img = load_images(image_dir="./Data/train/image", image_size=IMAGE_SIZE, isMask=False)
    
    # load masks and resize
    mask = load_images(image_dir="./Data/train/mask", image_size=IMAGE_SIZE, isMask=True)
    mask = mask.reshape(mask.shape[0], IMAGE_SIZE[0], IMAGE_SIZE[1], 1)

    # standardize
    img = (img / 255.0).astype("float32")
    # img = img.astype("float32")
    mask = (mask > 127).astype("float32")

    # validation
    train_img, val_img, train_mask, val_mask = train_test_split(img, mask, test_size=0.2, random_state=42)

    # data augmentation
    # to both images and masks
    transform = keras.Sequential()
    transform.add(keras.layers.RandomRotation(factor=0.1))
    transform.add(keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1))
    # transform.add(keras.layers.RandomElasticTransform())

    # this only gets done to the images, not the masks
    color = keras.Sequential()
    color.add(keras.layers.RandomBrightness(factor=0.2, value_range=(0,1)))
    color.add(keras.layers.RandomContrast(factor=0.2))


    # stolen form claude, see prompt 1 in prompts.md
    # create training and validation data pipelines
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
        keras.mixed_precision.set_global_policy("mixed_float16")
        if LOAD:
            # load checkpoint
            loaded = "check_"+SAVE_NAME
            print(f"LOADING {loaded}")
            model = keras.models.load_model(loaded)
        else:
            if BLOCK2:
                # build new 2block model
                model = unet_2block(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
            else:
                # build new 3block model
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

        # train
        model.fit(train_ds, validation_data=val_ds, epochs=200, 
                  callbacks=[
                    keras.callbacks.EarlyStopping(monitor="val_loss",restore_best_weights=True, patience=15),
                    keras.callbacks.ModelCheckpoint("check_"+SAVE_NAME,save_best_only=True, monitor="val_loss")
                  ])

        if SAVE:
            # save
            model.save(SAVE_NAME)
    else:
        # load model without training
        model = keras.models.load_model(SAVE_NAME)



    # SUBMISSION

    # eval
    loss, acc, tp, tn, fn, fp = model.evaluate(val_ds)
    print(f"IOU: {tp / (tp + fp + fn)}, DICE: {2*tp / (2*tp + fp + fn)}, LOSS: {loss}, BIN ACC: {acc}, TP: {tp}, TN: {tn}, FN: {fn}, FP: {fp}")
    print(f"IMAGE_SIZE: {IMAGE_SIZE}, SAVE_NAME: {SAVE_NAME}, TRAIN: {TRAIN}, SAVE: {SAVE}, BLOCK2: {BLOCK2}, LOAD: {LOAD}")

    # load test
    test = load_images(image_dir="./Data/test/image", image_size=IMAGE_SIZE, isMask=False)
    test = test / 255.0 # standardize
    
    # predict
    yhat = model.predict(test)
    yhat = (np.array(yhat) > 0.5).astype(np.uint8) * 255

    # sort test_names
    test_names = sorted(os.listdir("./Data/test/image"), key=lambda n:int(os.path.splitext(n)[0]))

    # create directory if it doesn't exist
    if not os.path.exists("./predictions"):
        os.makedirs("./predictions")

    # output to ./predictions
    for name, p in zip(test_names, yhat):
        pred = p.reshape(IMAGE_SIZE).astype(np.uint8) # from (256, 256, 1) to (256, 256)
        mask = Image.fromarray(pred).resize((512, 512), Image.NEAREST)
        mask.save(f"./predictions/{name}")

    # then run `make_submission.py ./predictions`
    
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
    okay, working on getting it configured

    comment out elastic transform
    commenting out elastictransform makes it take really fast -> that augmentation layer is the bottleneck
    my memory hovers around 78% then, and I don't even think i use the GPU, all CPU
    but i also have tabs open

    i'm gonna try 2 batches with elastic transform and close everything out
    it was pretty slow, ran it for 1 epoch took like 2m. 
    i did close all my tabs besides wsl and task manager
    memory would rise to about 75%
    GPU was not used during training, idk
    CPU would climb to 100% for augmentation, but be barely used during training
    so idek what's doing the work
    i think i gotta get rid of elastic transform and see if that does it
    but no errors

    no elastic transform, 2 block, 512 x 512, batch 1
    IOU: 0.6818822035274008, DICE: 0.8108560779075891, LOSS: 0.3103015720844269, BIN ACC: 0.9682290554046631, TP: 285635.0, TN: 3775412.0, FN: 64429.0, FP: 68828.0
    unet2block512.keras - IOU: 0.6890869380463042, DICE: 0.8159283249722387, LOSS: 0.3062680959701538, BIN ACC: 0.9675533771514893, TP: 301623.0, TN: 3756590.0, FN: 72074.0, FP: 64017.0 - not sure if this change is real, but my files were unsorted

    no elastic, 3 block, 512, 512, batch 2
    unet3block512.keras - IOU: 0.6904805990100799, DICE: 0.8169044938042057, LOSS: 0.3041748106479645, BIN ACC: 0.9673187732696533, TP: 305789.0, TN: 3751440.0, FN: 67908.0, FP: 69167.0
    slightly better. 

    lets do unet2 again, since the better file sizing stuff, and we can submit
    and turn batch back to 1 because it makes me feel it's faster
    unet2block_2.keras = IOU: 0.6803584021988752, DICE: 0.8097777251669348, LOSS: 0.316974937915802, BIN ACC: 0.9662847518920898, TP: 300996.0, TN: 3751896.0, FN: 72701.0, FP: 68711.0
    not as good as before, shoul di try to retrain it? idk
    
    ERROR NOT 512, ACCIDENTALLY CHANGED 3 BLOCK INSTEAD
    lets try 512 bottleneck?
    unet 2 block
    unet2block_3_bottle.keras - IOU: 0.685155712563485, DICE: 0.8131660563535882, LOSS: 0.3105551600456238, BIN ACC: 0.9669370651245117, TP: 301783.0, TN: 3753845.0, FN: 71914.0, FP: 66762.0, 105 epochs
    a little better than the previous
    let's try a redo from the checkpoint
    unet2block_3_bottle_2.keras - IOU: 0.6898456407702908, DICE: 0.8164599465497157, LOSS: 0.3027138113975525, BIN ACC: 0.9680709838867188, TP: 297865.0, TN: 3762519.0, FN: 75832.0, FP: 58088.0
    ah it does imporve
    okay, so error, the pervious one is not 512

    unet2, 512 image, 512 bottleneck, for real this time
    nah fuck it, 1024 bottleneck 
    unet2_bottle i believe, i don't think it finished
    unet2_bottle.keras = IOU: 0.6865768692717528, DICE: 0.8141661157350152, LOSS: 0.3088523745536804, BIN ACC: 0.967362642288208, TP: 299870.0, TN: 3757543.0, FN: 73827.0, FP: 63064.0
    it's better on first run than unet2block512.keras
    i'll try it with 3 block, and whichever gets highest i'll train it from the checkpoint one or two more times
    let me submit this one, or no the unet2block_3_bottle_2.keras one and see what the score is
    SHIT UH .809?????
    damn okay we gotta try unet3block

    unet3block, 512 image, 1024 bottleneck, no elastic
    unet3_1024.keras - IOU: 0.6649644104908401, DICE: 0.7987731224775011, LOSS: 0.3344310224056244, BIN ACC: 0.9650876522064209, TP: 290634.0, TN: 3757237.0, FN: 83063.0, FP: 63370.0
    hm, quite a bit worse

    so i'll run unet2_bottle a couple times
    see if i can break .7 iou or something
    unet2 - IOU: 0.6868649889620829, DICE: 0.814368658376989, LOSS: 0.31036046147346497, BIN ACC: 0.9661476612091064, TP: 311450.0, TN: 3740867.0, FN: 62247.0, FP: 79740.0
    i'll try it again see if it gets better
    unet 2 - IOU: 0.6906293528325081, DICE: 0.8170085911207166, LOSS: 0.3039122223854065, BIN ACC: 0.9675393104553223, TP: 303937.0, TN: 3754217.0, FN: 69760.0, FP: 66390.0
    AGAIN
    unet 2 - IOU: 0.6894323325405816, DICE: 0.816170401455272, LOSS: 0.305648535490036, BIN ACC: 0.9671845436096191, TP: 305544.0, TN: 3751122.0, FN: 68153.0, FP: 69485.0
    ARG AGAIN
    unet 2 - IOU: 0.689389930189643, DICE: 0.8161406882687591, LOSS: 0.30439087748527527, BIN ACC: 0.9678151607513428, TP: 299613.0, TN: 3759698.0, FN: 74084.0, FP: 60909.0
    FUCK
    AGAIN
    unet 2 - IOU: 0.6906963548050298, DICE: 0.817055472843532, LOSS: 0.3033336102962494, BIN ACC: 0.9675099849700928, TP: 304307.0, TN: 3753724.0, FN: 69390.0, FP: 66883.0
    keeping that
    best one yet, .81 score or something on kaggle, or .801. 


    final unet 2
    unet 2 - IOU: 0.6906963548050298, DICE: 0.817055472843532, LOSS: 0.3033336102962494, BIN ACC: 0.9675099849700928, TP: 304307.0, TN: 3753724.0, FN: 69390.0, FP: 66883.0
    IOU - 0.691
    DICE - 0.817


    final unet 3
    unet3_1024.keras - IOU: 0.6649644104908401, DICE: 0.7987731224775011, LOSS: 0.3344310224056244, BIN ACC: 0.9650876522064209, TP: 290634.0, TN: 3757237.0, FN: 83063.0, FP: 63370.0
    IOU - 0.665
    DICE - 0.799


    3 unnorm - IOU: 0.0, DICE: 0.0, LOSS: nan, BIN ACC: 0.9095194339752197, TP: 0.0, TN: 3814801.0, FN: 373697.0, FP: 5806.0
    2 unnorm - IOU: 0.0, DICE: 0.0, LOSS: nan, BIN ACC: 0.9104194641113281, TP: 0.0, TN: 3818576.0, FN: 373697.0, FP: 2031.0
    """

    
