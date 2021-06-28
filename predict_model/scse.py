import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import cv2 as cv
import glob, math, sys
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.layers import  *
import  time


gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError:
        pass


def sSE_block(inputs):
    x = Conv2D(1, 1, strides=1, padding='same')(inputs)
    #     B,H,W,C----->B,H,W,
    x = tf.keras.layers.Activation('sigmoid')(x)
    x = tf.multiply(x, inputs)
    return x


def cSE(inputs, rate=16):
    shape = inputs.shape
    #     print(shape)
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = Reshape((1, 1, shape[-1]))(x)
    #     print(x.shape)
    x = Conv2D(shape[-1] // 16, 1, strides=1, padding='same')(x)
    x = Conv2D(shape[-1], 1, strides=1, padding='same')(x)
    x = tf.keras.layers.Activation('sigmoid')(x)
    #     B,1,1,C
    x = tf.multiply(x, inputs)
    return x


def scSE_block(inputs):
    s = sSE_block(inputs)
    c = cSE(inputs, rate=16)
    add = tf.add(s, c)
    return add


def UNet(num_classes=2, input_shape=(512, 512, 3)):
    input = Input(shape=input_shape)

    conv1 = Conv2D(64, 3, padding='same', activation='relu')(input)
    conv1 = Conv2D(64, 3, padding='same', activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=2)(conv1)

    conv2 = Conv2D(128, 3, padding='same', activation='relu')(pool1)
    conv2 = Conv2D(128, 3, padding='same', activation='relu')(conv2)
    pool2 = MaxPooling2D(pool_size=2)(conv2)

    conv3 = Conv2D(256, 3, padding='same', activation='relu')(pool2)
    conv3 = Conv2D(256, 3, padding='same', activation='relu')(conv3)
    pool3 = MaxPooling2D(pool_size=2)(conv3)

    conv4 = Conv2D(512, 3, padding='same', activation='relu')(pool3)
    conv4 = Conv2D(512, 3, padding='same', activation='relu')(conv4)
    pool4 = MaxPooling2D(pool_size=2)(conv4)

    conv5 = Conv2D(1024, 3, padding='same', activation='relu')(pool4)
    conv5 = Conv2D(1024, 3, padding='same', activation='relu')(conv5)

    up1 = Conv2DTranspose(512, 3, strides=2, padding='same', activation='relu')(conv5)
    concat1 = tf.concat([up1, conv4], axis=-1)
    conv6 = Conv2D(512, 3, padding='same', activation='relu')(concat1)
    conv6 = Conv2D(512, 3, padding='same', activation='relu')(conv6)
    conv6 = scSE_block(conv6)

    up2 = Conv2DTranspose(256, 3, strides=2, padding='same', activation='relu')(conv6)
    concat2 = tf.concat([up2, conv3], axis=-1)
    conv7 = Conv2D(256, 3, padding='same', activation='relu')(concat2)
    conv7 = Conv2D(256, 3, padding='same', activation='relu')(conv7)
    conv7 = scSE_block(conv7)

    up3 = Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(conv7)
    concat3 = tf.concat([up3, conv2], axis=-1)
    conv8 = Conv2D(128, 3, padding='same', activation='relu')(concat3)
    conv8 = Conv2D(128, 3, padding='same', activation='relu')(conv8)
    conv8 = scSE_block(conv8)

    up4 = Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(conv8)
    concat4 = tf.concat([up4, conv1], axis=-1)
    conv9 = Conv2D(64, 3, padding='same', activation='relu')(concat4)
    conv9 = Conv2D(64, 3, padding='same', activation='relu')(conv9)
    conv9 = scSE_block(conv9)

    outputs = Conv2D(num_classes, 1, padding='same', activation='softmax')(conv9)
    model = tf.keras.Model(inputs=input, outputs=outputs)
    return model


'''
model = UNet(2)
# model.summary()
try:
    model.load_weights(r'D:\ProjectSummary\build_detection\predict_model\scse.h5')
    print('load weights scse')
except OSError as e:
    print('加载模型时出现错误， 错误原因为：{}'.format(e))

'''
# 需要指定对应的权重的路径以及名称
'''

over_lap = 0.3


def detection(img_arr, save_path=None):
    img = cv.cvtColor(img_arr, cv.COLOR_BGR2RGB)
    img = img / 127.5 - 1
    if len(img.shape) == 2:
        h, w = img.shape
    else:
        h, w, c = img.shape
    h_num = math.ceil((h - 152) / 360)
    w_num = math.ceil((w - 152) / 360)
    new_h = h_num * 360 + 152
    new_w = w_num * 360 + 152
    tmp_img = np.zeros((max(new_h, 512), max(new_w, 512), 3))
    pred_result = np.zeros((max(new_h, 512), max(new_w, 512)), np.int8)
    tmp_img[:h, :w, :] = img
    for i in range(0, new_h-152, 360):
        for j in range(0, new_h-152, 360):
            test_part = tmp_img[i:i+512, j:j+512,:]
            test_part = np.expand_dims(test_part, axis=0)
            pred_part = model.predict(test_part)
            pred_part = tf.argmax(pred_part, axis=-1)
            pred_part = pred_part[..., tf.newaxis]
            pred_part = tf.squeeze(pred_part)
            pred_result[i:i+512, j:j+512] += pred_part
    pred_result = np.where(pred_result >= 1, 255, 0)
    cv.imwrite('D:/ProjectSummary/build_detection/all_result/scse.png', pred_result[:h,:w],
               [int(cv.IMWRITE_PNG_COMPRESSION), 0])


if __name__ == '__main__':
    # img = cv.imread('D:/test_image/3147.0-375.0DOM.tif')
    img = cv.imread(sys.argv[1])
    detection(img)
    print('scse预测完成')



'''
