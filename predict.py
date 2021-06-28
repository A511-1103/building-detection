import os
os.environ['TF_CPP_MIN_LOG)LEVEL'] = '2'
from multiprocessing import Process
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


def load_model():
    try:
        resnet = ResNetFamily()
        res_model = resnet.run_model('res34')
        res_model.load_weights(r'D:\ProjectSummary\build_detection\predict_model\resnet34.h5')
        print('load weights res_model 1/5')
    except OSError as e:
        print('加载模型时出现错误， 错误原因为：{}'.format(e))

    try:
        hr_model = HRNet()
        hr_model.load_weights(r'D:\ProjectSummary\build_detection\predict_model\hrnet.h5')
        print('load weights hr_model 2/5')
    except OSError as e:
        print('加载模型时出现错误， 错误原因为：{}'.format(e))

    try:
        v3_model = Xception_DeepLabV3_Plus()
        v3_model.load_weights(r'D:\ProjectSummary\build_detection\predict_model\deep.h5')
        print('load weights v3_model 3/5')
    except OSError as e:
        print('加载模型时出现错误， 错误原因为：{}'.format(e))

    try:
        unet_model = UNet(2)
        unet_model.load_weights(r'D:\ProjectSummary\build_detection\predict_model\scse.h5')
        print('load weights unet_model 4/5')
    except OSError as e:
        print('加载模型时出现错误， 错误原因为：{}'.format(e))

    try:
        bam_model = Xception_DeepLabV3_Plus_bam()
        bam_model.load_weights(r'D:\ProjectSummary\build_detection\predict_model\bam.h5')
        print('load weights bam_model 5/5')
    except OSError as e:
        print('加载模型时出现错误， 错误原因为：{}'.format(e))

    return res_model, hr_model, v3_model, unet_model, bam_model


def del_files(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_files(c_path)
        else:
            os.remove(c_path)


def del_file(path):
    if os.path.exists(path):
        os.remove(path)
        print('成功删除中间过程文件 {}'.format(path))
    else:
        print('中间过程文件 {} 不存在'.format(path))


def run_model(img_path, user_path, name=''):
    models = ['res34_', 'hrnet_', 'v3plus_', 'scse_', 'bam_']
    models = [i + name for i in models]
    detection(img_path, user_path, model=res_model, save_name=models[0])
    print('第一个模型 res_model 检测完成，当前模型预测进度 1/5，请等待...')
    detection(img_path, user_path, model=hr_model, save_name=models[1])
    print('第二个模型 hr_model 检测完成，当前模型预测进度 2/5，请等待...')
    detection(img_path, user_path, model=v3_model, save_name=models[2])
    print('第三个模型 v3_model 检测完成，当前模型预测进度 3/5，请等待...')
    detection(img_path, user_path, model=unet_model, save_name=models[3])
    print('第四个模型 unet_model 检测完成，当前模型预测进度 4/5，请等待...')
    detection(img_path, user_path, model=bam_model, save_name=models[4])
    print('第五个模型 bam_model 检测完成，当前模型预测进度 5/5，等待进行模型融合')


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


def write_points(points, path):
    f = open(path, 'w', encoding='utf-8')
    for i in range(len(points)):
        point_str = ''
        point_x_y = points[i]
        point_x = point_x_y[0]
        point_y = point_x_y[1]
        for t_x in range(len(point_x)):
            x = point_x[t_x]
            y = point_y[t_x]
            point_str += '{},{} '.format(x, y)
        f.writelines(point_str)
        f.writelines('\n')
    f.close()


if __name__ == '__main__':
    user_path = r'D:\ProjectSummary\build_detection\all_result\auto_predict'    # 结果保存路径
    predict_mode = '2'      # predict_mode = '1' 表示单张检测；predict_mode = '2'，表示对整个文件夹中的图片进行预测
    res_model, hr_model, v3_model, unet_model, bam_model = load_model()
    print('加载模型完成')
    del_files(user_path)
    if predict_mode == '1': # 单张预测模式
        img_path = r'D:\test_image\1.png'       # 请输入正确的图片绝对路径，不可包含中文路径，如：r'D:\ProjectSummary\data\1.tif'
        name = str(img_path.split('\\')[-1])[:-4]
        now_path = os.path.join(user_path, name)
        if not os.path.exists(now_path):
            os.makedirs(now_path)
        run_model(img_path, now_path, name=name)
        model_confuse(now_path, name=name)
        print('模型融合完成，等待轮廓优化...')
        points, h = _detection(now_path + r'\{}_result.png'.format(name))
        write_points(points, path=now_path + r'\{}.txt'.format(name))
        img_name = ['bam_{}.png'.format(name), 'hrnet_{}.png'.format(name),
                    'scse_{}.png'.format(name), 'v3plus_{}.png'.format(name),
                    'res34_{}.png'.format(name)]
        for i in img_name:
            del_file(os.path.join(now_path, i))

    elif predict_mode == '2':   # 多张图片预测
        img_path = r'D:\ProjectSummary\build_detection\receive_file'          # 请输入正确的包含待预测图片所在的目录，确保文件夹中只有待预测图片，不可出现其他文件夹或其他格式的文件
        for i in os.listdir(img_path):
            abs_path = os.path.join(img_path, i)
            if not os.path.exists(abs_path):
                print('{}不存在'.format(abs_path))
            else:
                name = str(i)[:-4]
                now_path = os.path.join(user_path, name)
                if not os.path.exists(now_path):
                    os.makedirs(now_path)
                run_model(abs_path, now_path, name=name)
                model_confuse(now_path, name=name)
                print('模型融合完成，等待轮廓优化...')
                points, h = _detection(now_path + r'/{}_result.png'.format(name))
                write_points(points, path=now_path+r'\{}.txt'.format(name))
                img_name = ['bam_{}.png'.format(name), 'hrnet_{}.png'.format(name),
                            'scse_{}.png'.format(name), 'v3plus_{}.png'.format(name),
                            'res34_{}.png'.format(name)]
                for i in img_name:
                    del_file(os.path.join(now_path, i))
    else:
        raise NameError('请检查代码137行，预测模式输入有误')
    print(r'任务结束，请在 D:\ProjectSummary\build_detection\all_result\auto_predict 查看结果')