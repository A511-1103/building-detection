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



import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Conv2DTranspose, BatchNormalization, \
    Input, Activation, Add
from tensorflow.keras.models import Model


class ResNetFamily():
    def __init__(self, input_shape=(512,512,3)):
        self.inputs = Input(input_shape)
        self.f_size = 64

    def bn_conv_a(self, input_tensor, f_size, k_size, name, dila_rate=1):
        x = Conv2D(filters=f_size, kernel_size=k_size, padding='same',
                   name=name,dilation_rate=dila_rate,
                   kernel_initializer='he_normal')(input_tensor)
        x = BatchNormalization(name='{}_BN'.format(name))(x)
        x = Activation('relu', name='{}_AC'.format(name))(x)
        return x

    def res_block1(self, input_tensor, f_size, name, dila_rate=1):
        convx = self.bn_conv_a(input_tensor, f_size, 3, name='{}_1'.format(name), dila_rate=dila_rate)
        convx = self.bn_conv_a(convx, f_size, 3, name='{}_2'.format(name), dila_rate=dila_rate)
        out_tensor = Add(name='{}_add'.format(name))([input_tensor, convx])
        out_tensor = Activation('relu', name='{}_AC'.format(name))(out_tensor)
        return out_tensor

    def res34(self, input_tensor):
        f_size = self.f_size

        conv1 = self.bn_conv_a(input_tensor, f_size=f_size, k_size=3, name='conv1_1')
        conv1 = self.bn_conv_a(conv1, f_size=f_size, k_size=3, name='conv1_2')
        conv1 = self.bn_conv_a(conv1, f_size=f_size, k_size=3, name='conv1_3')
        # pool1 = MaxPool2D()(conv1)
        conv2 = Conv2D(f_size, 1, strides=2, padding='same', name='pool1')(conv1)
        for i in range(3):
            conv2 = self.res_block1(conv2, f_size=f_size, name='conv2_{}'.format(i))

        conv3 = Conv2D(f_size*2, 1, strides=2, padding='same', name='pool2')(conv2)
        for i in range(4):
            conv3 = self.res_block1(conv3, f_size=2 * f_size, name='conv3_{}'.format(i), dila_rate=1)

        conv4 = Conv2D(f_size * 4, 1, strides=2, padding='same', name='pool3')(conv3)
        for i in range(6):
            conv4 = self.res_block1(conv4, f_size=4 * f_size, name='conv4_{}'.format(i), dila_rate=1)

        conv5 = Conv2D(f_size * 8, 1, strides=2, padding='same', name='pool4')(conv4)
        for i in range(3):
            conv5 = self.res_block1(conv5, f_size=8 * f_size, name='conv5_{}'.format(i), dila_rate=1)
        return [conv1, conv2, conv3, conv4, conv5]

    def feature_fusion(self, net):
        conv1, conv2, conv3, conv4, conv5 = net

        conv2, conv3 = self.low_to_high_feature(conv1, conv2, conv3)
        conv3, conv4 = self.low_to_high_feature(conv2, conv3, conv4)
        conv1 = self.attention_demo(conv1)
        conv2 = self.attention_demo(conv2)
        conv3 = self.attention_demo(conv3)
        conv4 = self.attention_demo(conv4)
        conv5 = self.attention_demo(conv5)

        up4 = self.upsame_feature(conv4, conv5, name='4', dila_rate=1)
        up3 = self.upsame_feature(conv3, up4, name='3', dila_rate=1)
        up2 = self.upsame_feature(conv2, up3, name='2', dila_rate=1)
        up1 = self.upsame_feature(conv1, up2, name='1', dila_rate=1)
        out_tensor = Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')(up1)
        out_tensor = Conv2D(2, 3, padding='same', activation='softmax', kernel_initializer='he_normal')(out_tensor)
        return out_tensor

    def attention_demo(self, inputs):
        gap=tf.keras.layers.GlobalAveragePooling2D()(inputs)
        in_filters=gap.shape[-1]

        fc1=tf.keras.layers.Dense(in_filters//2)(gap)
        fc1=tf.keras.layers.BatchNormalization()(fc1)
        fc1=tf.keras.layers.ReLU()(fc1)

        fc2=tf.keras.layers.Dense(in_filters)(fc1)
        fc2=tf.keras.layers.BatchNormalization()(fc2)
        fc2=tf.keras.layers.Activation('sigmoid')(fc2)

        out=tf.reshape(fc2,[-1,1,1,in_filters])
    #     in_filters是通道数目,shape用list表示
        out=tf.multiply(inputs,out)
        return  out


    def channel_gate(self, inputs, rate=16):
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

    def spatial_gate(self, inputs, rate=16, d=4):
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

    def upsame_feature(self,low_f, high_f, name, dila_rate=1):
        high_f_up = Conv2DTranspose(low_f.shape[-1], 2, strides=(2, 2),
                                    activation = 'relu', padding = 'same')(high_f)
        out_f = tf.concat([low_f, high_f_up], axis=-1)
        out_f = Conv2D(low_f.shape[-1], 1, activation='relu', kernel_initializer='he_normal')(out_f)
        out_f = self.res_block1(out_f, out_f.shape[-1], name='upsame_{}'.format(name), dila_rate=dila_rate)
        return out_f

    def low_to_high_feature(self, low_f, mid_f, high_f):
        low_f_1 = MaxPool2D()(low_f)
        low_f_2 = MaxPool2D(strides=4)(low_f)
        mid_f_1 = MaxPool2D()(mid_f)
        high_f = tf.concat([high_f, mid_f_1, low_f_2], axis=-1)
        high_out_f = Conv2D(high_f.shape[-1], 1, activation='relu', kernel_initializer='he_normal')(high_f)
        mid_f = tf.concat([mid_f, low_f_1], axis=-1)
        mid_out_f = Conv2D(mid_f.shape[-1], 1, activation='relu', kernel_initializer='he_normal')(mid_f)
        return mid_out_f, high_out_f


    def run_model(self, name):
        if name == 'res34':
            net = self.res34(self.inputs)   # Trainable params: 22,910,272
            net = self.feature_fusion(net)
        else:
            raise ValueError('This network does not exist.')

        model = Model(self.inputs, net)
        return model


'''
resnet = ResNetFamily()
model = resnet.run_model('res34')
# model.summary()
try:
    model.load_weights(r'D:\ProjectSummary\build_detection\predict_model\resnet34.h5')
    print('load weights res34')
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
    cv.imwrite('D:/ProjectSummary/build_detection/all_result/res34.png', pred_result[:h,:w],
               [int(cv.IMWRITE_PNG_COMPRESSION), 0])


if __name__ == '__main__':
    # img = cv.imread('D:/test_image/3147.0-375.0DOM.tif')
    img = cv.imread(sys.argv[1])
    detection(img)
    print('res34预测完成')



'''