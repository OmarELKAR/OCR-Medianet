import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import sys
import os
import tensorflow as tf

img_list = os.listdir('./Datasets/PageSegData/PageImg')
img_list = [filename.split(".") for filename in img_list]

#visualize img and segmented image
def visualize(img, seg_img):
    plt.figure(figsize=(20,20))
    plt.subplot(1,2,1)
    plt.imshow(np.squeeze(img[:,:,:,0], axis=0), cmap='gray')
    plt.title('Image')
    plt.subplot(1,2,2)
    plt.imshow(np.squeeze(seg_img[:,:,:,0], axis=0), cmap='gray')
    plt.title('Segmented Image')
    plt.show()

def get_segmented_img(img, n_classes):
    seg_labels = np.zeros((512,512,1))
    img = cv2.resize(img, (512,512))
    img = img[:,:,0]
    class_list = [0,24]

    seg_labels[:,:,0] = (img!=0).astype(int)

    return seg_labels

def get_img(img):
    img = cv2.resize(img, (512,512))
    return img

def batch_generator(filelist, n_classes, batch_size):
    while True:
        X=[]
        Y=[]
        for i in range(batch_size):
            fn = random.choice(filelist)
            img = cv2.imread(f'./Datasets/PageSegData/PageImg/{fn[0]}.JPG',0)
            ret,img = cv2.threshold(img,150,255,cv2.THRESH_BINARY_INV)
            img = cv2.resize(img,(512,512))
            img = np.expand_dims(img, axis=-1)
            img=img/255

            seg = cv2.imread(f'./Datasets/PageSegData/PageSeg/{fn[0]}_mask.png', 1)
            seg = get_segmented_img(seg, n_classes)

            X.append(img)
            Y.append(seg)
        yield np.array(X), np.array(Y)

def conv_neural_net(pretrained_weights = None, input_size = (512,512,1)):
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth must be set for each GPU separately
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    
    inputs = tf.keras.layers.Input(shape=input_size)

    # Convolutional Block 1
    conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", kernel_initializer=tf.keras.initializers.he_normal())(inputs)
    conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", kernel_initializer=tf.keras.initializers.he_normal())(conv1)
    pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv1)

    # Convolutional Block 2
    conv2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.he_normal)(pool1)
    conv2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.he_normal)(conv2)
    conv2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.he_normal())(pool1)
    conv2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.he_normal())(conv2)
    pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv2)

    # Convolutional Block 3
    conv3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.he_normal())(pool2)
    conv3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.he_normal())(conv3)
    pool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv3)

    # Convolutional Block 4
    conv4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.he_normal())(pool3)
    conv4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.he_normal())(conv4)
    drop4 = tf.keras.layers.Dropout(0.5)(conv4)
    pool4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(drop4)

    # Convolutional Block 5
    conv5 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.he_normal())(pool4)
    conv5 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.he_normal())(conv5)
    drop5 = tf.keras.layers.Dropout(0.5)(conv5)

    # Upsampling Block 1
    up6 = tf.keras.layers.Conv2D(512, (2, 2), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.he_normal())(tf.keras.layers.UpSampling2D(size=(2, 2))(drop5))
    merge6 = tf.keras.layers.Concatenate(axis=3)([drop4, up6])
    conv6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.he_normal())(merge6)
    conv6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.he_normal())(conv6)

    # Upsampling Block 2
    up7 = tf.keras.layers.Conv2D(256, (2, 2), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.he_normal())(tf.keras.layers.UpSampling2D(size=(2, 2))(conv6))
    merge7 = tf.keras.layers.Concatenate(axis=3)([conv3, up7])
    conv7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.he_normal())(merge7)
    conv7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.he_normal())(conv7)

    # Upsampling Block 3
    up8 = tf.keras.layers.Conv2D(128, (2, 2), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.he_normal())(tf.keras.layers.UpSampling2D(size=(2, 2))(conv7))
    merge8 = tf.keras.layers.Concatenate(axis=3)([conv2, up8])
    conv8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.he_normal())(merge8)
    conv8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.he_normal())(conv8)

    # Upsampling Block 4
    up9 = tf.keras.layers.Conv2D(64, (2, 2), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.he_normal())(tf.keras.layers.UpSampling2D(size=(2, 2))(conv8))
    merge9 = tf.keras.layers.Concatenate(axis=3)([conv1, up9])
    conv9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.he_normal())(merge9)
    conv9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.he_normal())(conv9)
    conv9 = tf.keras.layers.Conv2D(2, (3, 3), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.he_normal())(conv9)
    
    # Output layer
    conv10 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = tf.keras.models.Model(inputs=inputs, outputs=conv10)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

model = conv_neural_net(pretrained_weights=sys.argv[1])
#model.summary()

random.shuffle(img_list)
trainF = img_list[0:int(0.75*len(img_list))]
testF = img_list[int(0.75*len(img_list)):]

mc = tf.keras.callbacks.ModelCheckpoint("weights{epoch:08d}.h5", save_weights_only=True, save_freq=1)

model.fit(batch_generator(trainF, 2, 2), epochs=4, steps_per_epoch=100, validation_data=batch_generator(testF,2,2),
                    validation_steps=400, callbacks=[mc], shuffle=1)
"""

img = cv2.imread(f'./Tests/test.jpg', 0)
ret, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
img = cv2.resize(img, (512,512))
img = np.expand_dims(img, axis=-1)
img = np.expand_dims(img, axis=0)
pred = model.predict(img)
pred = np.squeeze(np.squeeze(pred, axis=0), axis=-1)
plt.imshow(pred, cmap="gray")

plt.imsave("test_img_mask.JPG", pred)
"""