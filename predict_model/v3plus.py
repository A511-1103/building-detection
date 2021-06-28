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


def channel_gate(inputs, rate=16):
    channels = inputs.shape[-1]
    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(inputs)

    fc1 = tf.keras.layers.Dense(channels // rate)(avg_pool)
    fc1 = tf.keras.layers.BatchNormalization()(fc1)
    fc1 = tf.keras.layers.Activation('relu')(fc1)

    fc2 = tf.keras.layers.Dense(channels // rate)(fc1)
    fc2 = tf.keras.layers.BatchNormalization()(fc2)
    fc2 = tf.keras.layers.Activation('relu')(fc2)

    fc3 = tf.keras.layers.Dense(channels)(fc2)

    return fc3


def spatial_gate(inputs, rate=16, d=4):
    channels = inputs.shape[-1]

    conv = tf.keras.layers.Conv2D(channels // rate, 1)(inputs)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.Activation('relu')(conv)

    conv = tf.keras.layers.Conv2D(channels // rate, 3, dilation_rate=d, padding='same')(conv)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.Activation('relu')(conv)

    conv = tf.keras.layers.Conv2D(channels // rate, 3, dilation_rate=d, padding='same')(conv)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.Activation('relu')(conv)

    conv = tf.keras.layers.Conv2D(1, 1)(conv)

    return conv


def BAM_attention(inputs):
    c_out = channel_gate(inputs)
    #     B,C
    s_out = spatial_gate(inputs)

    c_out = tf.keras.layers.RepeatVector(inputs.shape[1] * inputs.shape[2])(c_out)
    #     B,C,H*W
    c_out = tf.reshape(c_out, [-1, inputs.shape[1], inputs.shape[2], inputs.shape[-1]])
    #     B,H,W,C
    out = tf.add(c_out, s_out)
    #     Broadcasting
    out = tf.keras.layers.Activation('sigmoid')(out)
    out = tf.add(tf.multiply(out, inputs), inputs)

    return out


def SKNet_block(inputs, reduce=16):
    conv = tf.keras.layers.Conv2D(256, 3, strides=1, padding='same')(inputs)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.ReLU()(conv)

    d1 = tf.keras.layers.Conv2D(256, 1, strides=1, padding='same')(conv)
    d1 = tf.keras.layers.BatchNormalization()(d1)
    d1 = tf.keras.layers.ReLU()(d1)

    d6 = tf.keras.layers.Conv2D(256, 3, strides=1, padding='same', dilation_rate=6)(conv)
    d6 = tf.keras.layers.BatchNormalization()(d6)
    d6 = tf.keras.layers.ReLU()(d6)

    d12 = tf.keras.layers.Conv2D(256, 3, strides=1, padding='same', dilation_rate=12)(conv)
    d12 = tf.keras.layers.BatchNormalization()(d12)
    d12 = tf.keras.layers.ReLU()(d12)

    d18 = tf.keras.layers.Conv2D(256, 3, strides=1, padding='same', dilation_rate=18)(conv)
    d18 = tf.keras.layers.BatchNormalization()(d18)
    d18 = tf.keras.layers.ReLU()(d18)

    gap = tf.keras.layers.GlobalAvgPool2D()(conv)
    gap = tf.keras.layers.Reshape((1, 1, gap.shape[-1]))(gap)
    gap = tf.keras.layers.Conv2D(256, 1, strides=1, padding='same')(gap)
    gap = tf.keras.layers.BatchNormalization()(gap)
    gap = tf.keras.layers.ReLU()(gap)
    gap = tf.keras.layers.UpSampling2D(size=conv.shape[1])(gap)

    total_features = tf.keras.layers.add([d1, d6, d12, d18, gap])

    total_features = tf.keras.layers.GlobalAvgPool2D()(total_features)

    channels = total_features.shape[-1]

    total_features = tf.keras.layers.Reshape((1, 1, channels))(total_features)

    total_features = tf.keras.layers.Conv2D(int(channels / reduce), 1, strides=1, padding='same')(total_features)
    total_features = tf.keras.layers.BatchNormalization()(total_features)
    total_features = tf.keras.layers.ReLU()(total_features)

    weighs = []

    for i in range(5):
        cur_weight = tf.keras.layers.Conv2D(channels, 1, strides=1, padding='same')(total_features)
        weighs.append(cur_weight)

    concat = tf.keras.layers.concatenate(weighs, axis=-2)
    concat = tf.keras.layers.Softmax(axis=-2)(concat)

    w = []
    for i in range(5):
        cur_w = tf.keras.layers.Cropping2D(cropping=((0, 0), (i, 4 - i)))(concat)
        w.append(cur_w)

    A1 = tf.keras.layers.multiply([d1, w[0]])
    A2 = tf.keras.layers.multiply([d6, w[1]])
    A3 = tf.keras.layers.multiply([d12, w[2]])
    A4 = tf.keras.layers.multiply([d18, w[3]])
    A5 = tf.keras.layers.multiply([gap, w[4]])

    multi_scale_fusion = tf.keras.layers.add([A1, A2, A3, A4, A5])
    multi_scale_fusion = tf.keras.layers.BatchNormalization()(multi_scale_fusion)
    multi_scale_fusion = tf.keras.layers.ReLU()(multi_scale_fusion)

    return multi_scale_fusion


def sSE_block(inputs):
    x = tf.keras.layers.Conv2D(1, 1, strides=1, padding='same')(inputs)
    #     B,H,W,C----->B,H,W,
    x = tf.keras.layers.Activation('sigmoid')(x)
    x = tf.multiply(x, inputs)
    return x


def cSE(inputs, rate=16):
    shape = inputs.shape
    #     print(shape)
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Reshape((1, 1, shape[-1]))(x)
    #     print(x.shape)
    x = tf.keras.layers.Conv2D(shape[-1] // 16, 1, strides=1, padding='same')(x)
    x = tf.keras.layers.Conv2D(shape[-1], 1, strides=1, padding='same')(x)
    x = tf.keras.layers.Activation('sigmoid')(x)
    #     B,1,1,C
    x = tf.multiply(x, inputs)
    return x


def scSE_block(inputs):
    s = sSE_block(inputs)
    c = cSE(inputs, rate=16)
    add = tf.add(s, c)
    return add


def Xception_DeepLabV3_Plus(shape=(512, 512, 3), num_classes=2):
    Input = tf.keras.layers.Input(shape=shape)

    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same', )(Input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    #     print(x.shape)
    #     x=BAM_attention(x)
    c = x

    residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', )(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(128, (3, 3), padding='same', )(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = add([x, residual])

    # x4

    #     print(x.shape)
    #     x=BAM_attention(x)
    c1 = x

    residual = Conv2D(256, (1, 1), strides=(2, 2), padding='same')(x)
    residual = BatchNormalization()(residual)
    x = Activation('relu')(x)
    x = SeparableConv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(256, (3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)

    x = add([x, residual])

    # x8

    #     x=BAM_attention(x)
    c2 = x
    #     print(x.shape)

    # 如果未使用空洞卷积的话，此处进行降采样处理，否则则使用空洞卷积的dilate代替
    residual = Conv2D(728, (1, 1), strides=2, padding='same')(x)
    residual = BatchNormalization()(residual)
    x = Activation('relu')(x)
    x = SeparableConv2D(728, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu', )(x)
    x = SeparableConv2D(728, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(728, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)

    x = add([x, residual])
    c3 = x
    # x16

    for i in range(16):
        residual = x

        x = Activation('relu')(x)
        x = SeparableConv2D(728, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(728, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(728, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = add([x, residual])
    c4 = x

    # strides=1,not 2
    # so total down 4 times
    #     x=BAM_attention(x)

    residual = Conv2D(1024, (1, 1), strides=1, padding='same')(x)
    residual = BatchNormalization()(residual)
    x = Activation('relu')(x)
    x = SeparableConv2D(728, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(1024, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(1024, (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)

    x = add([x, residual])

    x = SeparableConv2D(1536, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(1536, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(2048, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    c5 = x

    #  input 唯一的一次卷积进行降采样处理
    #  流入 3个block  3次降采样
    #  中间 16个block
    #  输出 2个block  第一个使用残差+降采样，第二个未使用残差
    def conv_bn_relu(inputs, filters, kernel_size, strides, activate=True, padding='same', dilate=1):
        conv = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding=padding, dilation_rate=dilate)(inputs)
        conv = tf.keras.layers.BatchNormalization()(conv)
        if activate:
            conv = tf.keras.layers.Activation('relu')(conv)
        return conv

    def ASPP(inputs):
        conv = conv_bn_relu(inputs=inputs, filters=256, kernel_size=1, strides=1)

        pool1 = conv_bn_relu(inputs=inputs, filters=256, kernel_size=3, strides=1, dilate=6)
        pool2 = conv_bn_relu(inputs=inputs, filters=256, kernel_size=3, strides=1, dilate=12)
        pool3 = conv_bn_relu(inputs=inputs, filters=256, kernel_size=3, strides=1, dilate=18)

        avg_pool = tf.keras.layers.AveragePooling2D(pool_size=32)(inputs)
        avg_pool = conv_bn_relu(inputs=avg_pool, filters=256, kernel_size=1, strides=1)
        avg_pool = tf.keras.layers.UpSampling2D(size=32)(avg_pool)

        concat = tf.concat([conv, pool1, pool2, pool3, avg_pool], axis=-1)
        return concat

    sk_conv1 = SKNet_block(c5)
    c5 = ASPP(c5)
    conv1 = conv_bn_relu(inputs=c5, filters=256, kernel_size=1, strides=1)

    conv1 = tf.keras.layers.concatenate([conv1, sk_conv1])
    conv1 = conv_bn_relu(inputs=conv1, filters=256, kernel_size=3, strides=1)
    conv1 = conv_bn_relu(inputs=conv1, filters=256, kernel_size=3, strides=1)
    conv1 = scSE_block(conv1)
    '''
    融合两种多尺度特征后再进行上采样处理

    '''
    up1 = tf.keras.layers.UpSampling2D(size=2)(conv1)

    concat1 = tf.concat([up1, c2], axis=-1)
    concat1 = conv_bn_relu(inputs=concat1, filters=256, kernel_size=3, strides=1)
    concat1 = conv_bn_relu(inputs=concat1, filters=256, kernel_size=3, strides=1)
    concat1 = scSE_block(concat1)
    #     64*64*256
    up2 = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same')(concat1)

    concat2 = tf.concat([up2, c1], axis=-1)
    concat2 = conv_bn_relu(inputs=concat2, filters=128, kernel_size=3, strides=1)
    concat2 = conv_bn_relu(inputs=concat2, filters=128, kernel_size=3, strides=1)
    concat2 = scSE_block(concat2)

    up3 = tf.keras.layers.Conv2DTranspose(64, 3, 2, padding='same')(concat2)
    concat3 = tf.concat([c, up3], axis=-1)
    concat3 = conv_bn_relu(inputs=concat3, filters=64, kernel_size=3, strides=1)
    concat3 = conv_bn_relu(inputs=concat3, filters=64, kernel_size=3, strides=1)
    concat3 = scSE_block(concat3)

    output = tf.keras.layers.UpSampling2D(size=2)(concat3)
    output = conv_bn_relu(inputs=output, filters=32, kernel_size=3, strides=1)
    output = conv_bn_relu(inputs=output, filters=32, kernel_size=3, strides=1)

    up = Conv2D(num_classes, 1, 1, activation='softmax')(output)

    model = tf.keras.Model(inputs=Input, outputs=up)
    # model.summary()

    return model

'''
model = Xception_DeepLabV3_Plus()
try:
    model.load_weights(r'D:\ProjectSummary\build_detection\predict_model\deep.h5')
    print('load weights v3plus')
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
    cv.imwrite('D:/ProjectSummary/build_detection/all_result/v3plus.png', pred_result[:h,:w],
               [int(cv.IMWRITE_PNG_COMPRESSION), 0])


if __name__ == '__main__':
    # img = cv.imread('D:/test_image/3147.0-375.0DOM.tif')
    img = cv.imread(sys.argv[1])
    detection(img)
    print('v3plus 预测完成')

'''