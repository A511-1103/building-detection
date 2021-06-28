import os
os.environ['TF_CPP_MIN_LOG)LEVEL'] = '2'
from multiprocessing import Process
import tornado.ioloop
import base64
from flask import request
from flask import Flask, jsonify, json
import time, math
from predict_model.res34 import ResNetFamily
from predict_model.hrnet import HRNet
from predict_model.v3plus import Xception_DeepLabV3_Plus
from predict_model.scse import UNet
from predict_model.bam import Xception_DeepLabV3_Plus_bam
import cv2 as cv
import numpy as np
import tensorflow as tf
from model_fuse import model_confuse
from edge_3 import _detection


'''
服务端 Server
对外提供建筑物检测的API接口
'''
app = Flask(__name__)


def load_model():
    try:
        resnet = ResNetFamily()
        res_model = resnet.run_model('res34')
        res_model.load_weights(r'D:\ProjectSummary\build_detection\predict_model\resnet34.h5')
        print('load weights res_model')
    except OSError as e:
        print('加载模型时出现错误， 错误原因为：{}'.format(e))

    try:
        hr_model = HRNet()
        hr_model.load_weights(r'D:\ProjectSummary\build_detection\predict_model\hrnet.h5')
        print('load weights hr_model')
    except OSError as e:
        print('加载模型时出现错误， 错误原因为：{}'.format(e))

    try:
        v3_model = Xception_DeepLabV3_Plus()
        v3_model.load_weights(r'D:\ProjectSummary\build_detection\predict_model\deep.h5')
        print('load weights v3_model')
    except OSError as e:
        print('加载模型时出现错误， 错误原因为：{}'.format(e))

    try:
        unet_model = UNet(2)
        unet_model.load_weights(r'D:\ProjectSummary\build_detection\predict_model\scse.h5')
        print('load weights unet_model')
    except OSError as e:
        print('加载模型时出现错误， 错误原因为：{}'.format(e))

    try:
        bam_model = Xception_DeepLabV3_Plus_bam()
        bam_model.load_weights(r'D:\ProjectSummary\build_detection\predict_model\bam.h5')
        print('load weights bam_model')
    except OSError as e:
        print('加载模型时出现错误， 错误原因为：{}'.format(e))

    return res_model, hr_model, v3_model, unet_model, bam_model


def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)


res_model, hr_model, v3_model, unet_model, bam_model = load_model()
print('加载模型完成')
del_file(r'D:\ProjectSummary\build_detection\receive_file')
# 定义路由
@app.route("/photo", methods=['POST'])
def get_frame():
    try:
        abs_path = r'D:\ProjectSummary\build_detection\all_result'
        user_IP = request.headers.get('clientID')
        user_path = os.path.join(abs_path, user_IP)
        print(user_path)
        if not os.path.exists(user_path):
            os.makedirs(user_path)
        else:
            del_file(user_path)

        # 接收图片
        upload_file = request.files['file']

        # 获取图片名
        file_name = upload_file.filename
        if upload_file == None:
            return jsonify(status='NG', data=None, points={}, error='传入的图片错误')
        if file_name == None:
            return jsonify(status='NG', data=None, points={}, error='传入的图片名字为空')
        # 文件保存目录（桌面）
        file_path = r'D:\ProjectSummary\build_detection\receive_file'
        if upload_file:
            # 地址拼接
            file_paths = os.path.join(file_path, file_name)
            # 保存接收的图片到桌面
            upload_file.save(file_paths)

            run_model(file_paths, user_path)
            print('等待融合中...')
            model_confuse(user_path)

            try:
                points, h = _detection(user_path + r'/result.png')
            except Exception as e:
                points = None
            print(points)

            # 打开一张其他图片作为结果返回，
            with open(user_path + r'/result.png', 'rb') as f:
                res = base64.b64encode(f.read())
            data = {}
            data['status'] = 'success'
            data['data'] = res
            data['points'] = {}
            for i in range(len(points)):
                point_str = ''
                point_x_y = points[i]
                point_x = point_x_y[0]
                point_y = point_x_y[1]
                if len(point_x) != len(point_y):
                    return jsonify(status='NG',
                                   data=None,
                                   points={},
                                   error='轮廓优化时出现错误，请检查服务端 edge_3.py文件')
                for t_x in range(len(point_x)):
                    x = point_x[t_x]
                    y = point_y[t_x]
                    point_str += '{},{} '.format(x, y)
                # tmp = {'point_x': point_x, 'point_y': point_y}
                data['points']['{}'.format(i)] = point_str
            # return jsonify(status='OK', data=res, points=[points])
            data['error'] = 'None'
            print(data)
            return json.dumps(data, ensure_ascii=False, encoding='utf-8')
    except Exception as e:
        return jsonify(status='NG', data=None, points={}, error=e)


class MyProcess(Process):
    def __init__(self):
        super(MyProcess, self).__init__()

    def run(self) -> None:
        detection(self.img_path, model=self.model, save_name=self.name)
        # os.system(r'python ./predict_model/{}.py {}'.format(self.model, self.img_path))

    def getP(self, img_path, model, name):
        self.model = model
        self.img_path = img_path
        self.name = name


def run_model(img_path, user_path):
    models = ['res34', 'hrnet', 'v3plus', 'scse', 'bam']
    use_Process = False
    if use_Process:
        res34 = MyProcess()
        res34.getP(img_path, res_model, models[0])
        res34.start()

        hrnet = MyProcess()
        hrnet.getP(img_path, hr_model, models[1])
        hrnet.start()

        v3plus = MyProcess()
        v3plus.getP(img_path, v3_model, models[2])
        v3plus.start()

        scse = MyProcess()
        scse.getP(img_path, unet_model, models[3])
        scse.start()

        bam = MyProcess()
        bam.getP(img_path, bam_model, models[4])
        bam.start()

        res34.join()
        hrnet.join()
        v3plus.join()
        scse.join()
        bam.join()
    else:
        detection(img_path, user_path, model=res_model, save_name=models[0])
        detection(img_path, user_path, model=hr_model, save_name=models[1])
        detection(img_path, user_path, model=v3_model, save_name=models[2])
        detection(img_path, user_path, model=unet_model, save_name=models[3])
        detection(img_path, user_path, model=bam_model, save_name=models[4])


def detection(img_path, user_path, model, save_name='model'):
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
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
    cv.imwrite(user_path + '/{}.png'.format(save_name), pred_result[:h,:w],
               [int(cv.IMWRITE_PNG_COMPRESSION), 0])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)

