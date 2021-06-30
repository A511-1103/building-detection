import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2 as cv
import numpy as np
import tensorflow as tf
import time, random, shutil
from multiprocessing import Process
from tqdm import tqdm
import skimage


'''
介绍      --->    数据增强方法
1、原始图像有4736张
2、使用随机上下翻转，设置阈值为0.3，随机产生一个0-1内的小数，大于阈值时，进行上下翻转
3、使用随机左右翻转，设置阈值为0.3，随机产生一个0-1内的小数，大于阈值时，进行左右翻转
4、随机尺度缩放+随机copy
    对标签进行分析: 若标签中建筑物的占比大于20%，不处理；小于20%记为待处理样本
    在待处理样本中: 若标签中建筑物的占比小于20%、大于7.5%记为优样本；
                 若标签中建筑物的占比小于7.5%记为劣样本；
        随机在优样本和劣样本中各取一张图像，进行随机尺度缩放，然后把处理过的优样本中的建筑物单独提取出来，copy到劣样本中。
'''

class MyProcess(Process):   # 多进程函数
    def __init__(self):
        super(MyProcess, self).__init__()

    def run(self) -> None:
        cv.imwrite('{}'.format(self.img_path), self.img)
        cv.imwrite('{}'.format(self.lab_path), self.lab)

    def getPara(self, img, lab, img_path, lab_path):
        self.img = img
        self.lab = lab
        self.img_path = img_path
        self.lab_path = lab_path


class Data_Enhance():
    def __init__(self):
        self.img_w = self.img_h = 512
        '''
        注意，此处的路径需要根据实际情况进行修改，且格式如下，不能出现中文路径
        1、self.read_img_path：需要进行数据增强的图片路径
        2、self.read_lab_path：需要进行数据增强的标签图片路径
        3、self.save_img_path：进行数据增强后，图片保存路径
        4、self.save_lab_path：进行数据增强后，标签图片保存路径
        '''
        self.read_img_path = r'D:\dataset\train\image'          #
        self.read_lab_path = r'D:\dataset\train\label'
        self.save_img_path = r'D:\dataset\new_train1\image'
        self.save_lab_path = r'D:\dataset\new_train1\label'

        if not os.path.exists(self.read_img_path) or not os.path.exists(self.read_lab_path):
            raise FileNotFoundError('路径不存在')
        self.mkdirs_path(self.save_img_path)
        self.mkdirs_path(self.save_lab_path)
        random.seed()
        self.ct = 0
        self.save_format = '.png'

    def run(self, use_process=True):
        img_names = os.listdir(self.read_img_path)
        for i in tqdm(range(len(img_names)), ncols = 80, desc='process', mininterval = 0.1):
        # for i in range(5):
            #
            img = cv.imread(os.path.join(self.read_img_path, img_names[i]))
            lab = cv.imread(os.path.join(self.read_lab_path, img_names[i]))
            img_path = os.path.join(self.save_img_path, img_names[i].split('.')[0] + self.save_format)
            lab_path = os.path.join(self.save_lab_path, img_names[i].split('.')[0] + self.save_format)
            self.start_process(img, lab, img_path, lab_path, use_process)

            if random.random() > 0.2:   # 随机上下翻转
                img_1 = tf.image.flip_up_down(img).numpy()
                lab_1 = tf.image.flip_up_down(lab).numpy()
                img_path = os.path.join(self.save_img_path, img_names[i].split('.')[0] + '_1' + self.save_format)
                lab_path = os.path.join(self.save_lab_path, img_names[i].split('.')[0] + '_1' + self.save_format)
                self.start_process(img_1, lab_1, img_path, lab_path, use_process)

            if random.random() > 0.2:   # 随机左右旋转
                img_2 = tf.image.flip_left_right(img).numpy()
                lab_2 = tf.image.flip_left_right(lab).numpy()
                img_path = os.path.join(self.save_img_path, img_names[i].split('.')[0] + '_2' + self.save_format)
                lab_path = os.path.join(self.save_lab_path, img_names[i].split('.')[0] + '_2' + self.save_format)
                self.start_process(img_2, lab_2, img_path, lab_path, use_process)

            if random.random() > 0.2:   # 随机0.6-2尺度缩放
                scale_factor = random.randint(6,20) / 10
                img_3, lab_3 = self.random_scale_resize(img, lab, scale_factor)
                img_path = os.path.join(self.save_img_path, img_names[i].split('.')[0] + '_3' + self.save_format)
                lab_path = os.path.join(self.save_lab_path, img_names[i].split('.')[0] + '_3' + self.save_format)
                self.start_process(img_3, lab_3, img_path, lab_path, use_process)

            if random.random() > 0.7:   # 随机 颜色变换
                img_4 = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                img_path = os.path.join(self.save_img_path, img_names[i].split('.')[0] + '_4' + self.save_format)
                lab_path = os.path.join(self.save_lab_path, img_names[i].split('.')[0] + '_4' + self.save_format)
                self.start_process(img_4, lab, img_path, lab_path, use_process)

        print('处理完成！')

    def random_scale_resize(self, img, lab, random_scale):  # 随机尺度缩放
        img_h, img_w, _ = img.shape
        n_h = int(img_h * random_scale)
        n_w = int(img_w * random_scale)
        x = (img_w - n_w) // 2
        y = (img_h - n_h) // 2
        image = cv.resize(img, (n_h, n_w))
        label = cv.resize(lab, (n_h, n_w))
        label = self.label_(label)
        if random_scale < 1:
            new_image = np.ones((img_h,img_w,3)) * 128
            new_label = np.zeros((img_h,img_w,3))
            new_image[y:y+n_h, x:x+n_w, :] = image
            new_label[y:y+n_h, x:x+n_w, :] = label
        else:
            x = (n_w - img_w) // 2
            y = (n_h - img_h) // 2
            x,y = x-1, y-1
            x = np.maximum(x, 0)
            y = np.maximum(y, 0)
            # print('x={},y={}'.format(x,y))
            new_image = image[y:y+512,x:x+512,:]
            new_label = label[y:y+512,x:x+512,:]
        if 0.7 > random.random() >= 0.4:
            new_image = tf.image.flip_up_down(new_image).numpy()
            new_label = tf.image.flip_up_down(new_label).numpy()
        elif random.random() >= 0.7:
            new_image = tf.image.flip_left_right(new_image).numpy()
            new_label = tf.image.flip_left_right(new_label).numpy()
        return new_image, new_label

    def label_(self, lab):  # 标签resize后会，在建筑物边缘出现杂质，以125为阈值进行清除
        n_lab = np.where(lab > 125, 255, 0)
        return n_lab

    def mkdirs_path(self, new_path):    # 检查目录是否存在，若不存在则创建
        if not os.path.exists(new_path):
            os.makedirs(new_path)
            print('成功创建：{}'.format(new_path))

    def start_process(self, img, lab, img_path, lab_path, use_process=True):
        # 根据需求，选择是否启用多进程处理
        if use_process:
            myProcess = MyProcess()
            myProcess.getPara(img, lab, img_path, lab_path)
            myProcess.start()
        else:
            cv.imwrite('{}'.format(img_path), img, [int(cv.IMWRITE_PNG_COMPRESSION), 0])
            cv.imwrite('{}'.format(lab_path), lab, [int(cv.IMWRITE_PNG_COMPRESSION), 9])


    def split_train_val(self, img_path, lab_path):
        '''
        :param img_path: 输入参数 原始影像路径
        :param lab_path: 输入参数 原始标签路径
        :return:
        '''

        '''
        train_img_save_path :   生成的训练影像样本保存路径
        train_lab_save_path :   生成的训练标签样本保存路径
        val_img_save_path   :   生成的验证影像样本保存路径
        val_lab_save_path   :   生成的验证标签样本保存路径
        split_rate = 0.1    :   分割比例，一般设置为训练样本占9成，验证样本占1成
        '''
        train_img_save_path = r'D:\ProjectSummary\data\img'
        train_lab_save_path = r'D:\ProjectSummary\data\label'
        val_img_save_path = r'D:\ProjectSummary\data\img'
        val_lab_save_path = r'D:\ProjectSummary\data\label'
        split_rate = 0.9

        img_dir = os.listdir(img_path)
        lab_dir = os.listdir(lab_path)
        img_nums = len(img_dir)
        lab_nums = len(lab_dir)
        if img_nums != lab_nums:
            raise ValueError('影像样本数量与标签样本数量不一致，请检查数据')
        else:
            self.mkdirs_path(train_img_save_path)
            self.mkdirs_path(train_lab_save_path)
            self.mkdirs_path(val_img_save_path)
            self.mkdirs_path(val_lab_save_path)

            nums_list = np.arange(img_nums)
            split_num = int(img_nums * split_rate)
            train_nums_list = nums_list[:split_num]
            val_nums_list = nums_list[split_num:]

            for i in train_nums_list:
                if img_dir[i] == lab_dir[i]:
                    copy_img(os.path.join(img_path,img_dir[i]),
                             os.path.join(lab_path,lab_dir[i]))
                else:
                    raise ValueError('文件名称不一致，请检查输入图像的文件夹中图片{}与标签{}名称'.format(img_dir[i], lab_dir[i]))

            for i in val_nums_list:
                if img_dir[i] == lab_dir[i]:
                    copy_img(os.path.join(img_path,img_dir[i]),
                             os.path.join(lab_path,lab_dir[i]))
                else:
                    raise ValueError('文件名称不一致，请检查输入图像的文件夹中图片{}与标签{}名称'.format(img_dir[i], lab_dir[i]))
            print('数据集划分完成！')


    def run_split_func(self):
        self.split_train_val(self.save_img_path, self.save_lab_path)


def mkdirs_path(new_path):    # 检查目录是否存在，若不存在则创建
    if not os.path.exists(new_path):
        os.makedirs(new_path)
        print('成功创建：{}'.format(new_path))


def copy_img(old_path, new_path):
    shutil.copy(old_path, new_path)


if __name__ == '__main__':
    data = Data_Enhance()
    run_status = [0,1]  # 默认不使用多进程处理，若想使用多进程，就将[0,1]换成[1,0]
    t1 = time.time()
    if run_status[0]:
        data.run()
        t2 = time.time()
        print('使用多进程耗时：{}'.format(t2-t1))

    if run_status[1]:
        data.run(use_process=False)
        t3 = time.time()
        print('不使用多进程耗时：{}'.format(t3-t1))

    data.run_split_func()