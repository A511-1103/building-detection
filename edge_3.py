import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import glob


"""
    算法功能：
            优化建筑物边缘。
    input:
            预测图像路径
    return:
            优化后的建筑物角点        
"""
def make_path(p_path,images_name):
    flag=os.path.exists(p_path)
    if not flag:
        os.makedirs(p_path)
    c_path=p_path+'/'+images_name
    c_flag=os.path.exists(c_path)
    if not c_flag:
        os.makedirs(c_path)


def iou(initial_bbox, erode_bbox):
    initial_bbox = np.array(initial_bbox)
    erode_bbox = np.array(erode_bbox)
    #     print(initial_bbox[:4])

    inter_left = np.maximum(initial_bbox[:2], erode_bbox[:, :2])
    inter_right = np.minimum(initial_bbox[2:4], erode_bbox[:, 2:4])
    inter_wh = inter_right - inter_left
    inter_wh = np.maximum(inter_wh, 0)
    inter_area = inter_wh[:, 0] * inter_wh[:, 1]
    #     print(inter_area)
    ini_area = (initial_bbox[2] - initial_bbox[0]) * (initial_bbox[3] - initial_bbox[1])
    prior_area = (erode_bbox[:, 2] - erode_bbox[:, 0]) * (erode_bbox[:, 3] - erode_bbox[:, 1])
    union_area = ini_area + prior_area - inter_area
    iou = inter_area / union_area
    #     print(iou)
    req = iou > 0.5
    #   no iou>0.6
    if np.any(req):
        return np.argmax(iou)
    else:
        return None


def process_td(initial_edge, erode_edge):
    initial_bbox = []
    for i in range(len(initial_edge)):
        x, y, w, h = cv.boundingRect(initial_edge[i])
        xmax = x+w
        ymax = y+h
        initial_bbox.append([x, y, xmax, ymax, i])
    erode_bbox=[]
    for j in range(len(erode_edge)):
        x, y, w, h = cv.boundingRect(erode_edge[j])
        xmax = x+w
        ymax = y+h
        erode_bbox.append([x, y, xmax, ymax, j])
    ini_map = []
    add_map = []
    for i in range(len(initial_edge)):
        res=iou(initial_bbox[i], erode_bbox)
        if res is None:
            ini_map.append(i)
        else:
            add_map.append(res)
#     消失的与新增的
#     print(ini_map,add_map)
    disapper = []
#     print('无法对应的轮廓数有:{}个'.format(len(ini_map)))
    for i in range(len(ini_map)):
        disapper.append(initial_bbox[ini_map[i]])
#         消失的,里面的内容是:xmin,ymin,xmax,ymax,cnt_idx
    add = []
#     print('新增的区域有：{}'.format(len(erode_bbox)-len(add_map)))
    for i in range(len(erode_edge)):
        if i in add_map:
            continue
        add.append(erode_bbox[i])
#         新增的
    return disapper, add


def process_rl(initial_edge, erode_edge):
    initial_bbox = []
    for j in range(len(initial_edge)):
        if initial_edge[j] is None:
            #             消失的轮廓，其对应的坐标值以0初始化
            initial_bbox.append([0, 0, 0, 0, j])
            continue
        x, y, w, h = cv.boundingRect(initial_edge[j])
        initial_bbox.append([x, y, x + w, y + h, j])
    erode_bbox = []
    for j in range(len(erode_edge)):
        x, y, w, h = cv.boundingRect(erode_edge[j])
        erode_bbox.append([x, y, x + w, y + h, j])
    in_erode = []
    not_in_erode = []
    for i in range(len(initial_bbox)):
        res = iou(initial_bbox[i], erode_bbox)
        if res is None:
            not_in_erode.append(initial_bbox[i][4])
            continue
        in_erode.append(res)
    disapper_bbox = []
    #     1.无法对应的情况：一分为二，腐蚀加填充，本身已经是忽略的框
    for i in range(len(not_in_erode)):
        disapper_bbox.append(initial_bbox[not_in_erode[i]])
    #   新增的轮廓

    new_bbox = []
    for i in range(len(erode_edge)):
        if i in in_erode:
            continue
        new_bbox.append(erode_bbox[i])
    #         xmin,ymin,xmax,ymax,index
    return disapper_bbox, new_bbox


def erode_images_process(erode_img, contours):
    bad_erode = []
    for i in range(len(contours)):
        area = cv.contourArea(contours[i])
        if area < 50:

            cv.drawContours(erode_img, contours, i, 0, cv.FILLED)
            if area <= 10:
                continue
            else:
                # print('腐蚀之后有小区快产生')
                bad_erode.append(contours[i])
    #             处理腐蚀后多余的小轮廓
    #             对处理后的图片再次进行轮廓检测
    erode_opt = erode_img[:, :, 0].copy()
    res = cv.findContours(erode_opt, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
    if len(res) == 2:
        contour, _ = res
    else:
        _, contour,_ = res
    return erode_img, contour,  bad_erode


def plot_bad_erode(images, cnt):
    for i in range(len(cnt)):
        x, y, w, h = cv.boundingRect(cnt[i])
        cv.circle(images, (int(x+w/2), int(y+h/2)), 20, (0, 0, 255))


def plot_bad_erode1(images,cnt):
    for i in range(len(cnt)):
        x, y, w, h = cv.boundingRect(cnt[i])
        cv.circle(images, (int(x+w/2), int(y+h/2)), 20, (0, 255, 0))


def detction_overlap_building(input_img, input_edge, kernel_size, iteration):
    img = input_img.copy()
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    data = img.copy()
    #     print(data.shape)
    res_ini = cv.findContours(data, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
    if len(res_ini) == 2:
        res1, _ = res_ini
    else:
        _, res1, _ = res_ini

    target_num = len(res1)

    img1 = input_img.copy()
    kernel = np.ones((1, kernel_size), np.uint8)
    # top-down
    erosion1 = cv.erode(img1, kernel, iterations=iteration)
    #     print(erosion1.shape)
    gray_e1 = cv.cvtColor(erosion1, cv.COLOR_BGR2GRAY)
    #     print(gray_e1.shape)
    res2 = cv.findContours(gray_e1, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
    if len(res2) == 2:
        contours1, _ = res2
    else:
        _, contours1, _ = res2
    opt_ero, contours1, bad_erode1 = erode_images_process(erosion1, contours1)
    if bad_erode1 != []:
        plot_bad_erode(erosion1, bad_erode1)
    single_td = len(contours1)

    img2 = input_img.copy()
    kernel = np.ones((kernel_size, 1), np.uint8)
    erosion2 = cv.erode(img2, kernel, iterations=iteration)
    #     对其进行了腐蚀之后，对于检测效果较差之处
    gray_e2 = cv.cvtColor(erosion2, cv.COLOR_BGR2GRAY)
    res3 = cv.findContours(gray_e2, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
    if len(res3) == 2:
        contours2, _ = res3
    else:
        _, contours2, _ = res3
    opt_ero1, contours2, bad_erode2 = erode_images_process(erosion2, contours2)
    if bad_erode2 != []:
        plot_bad_erode1(erosion2, bad_erode2)
    single_rl = len(contours2)

    if (single_td == target_num) and (single_rl == target_num):
        # print("没有边角重叠在一起的建筑物")
        dis = None
        add = None
        dis1 = None
        add1 = None
    else:
        if single_td != target_num:
            dis, add = process_td(res1, contours1)
        else:
            dis = None
            add = None

        if single_rl != target_num:
            dis1, add1 = process_rl(res1, contours2)
        else:
            dis1 = None
            add1 = None

        if dis != None:
            for i in range(len(dis)):
                res1[dis[i][4]] = None

        if dis1 != None:
            for i in range(len(dis1)):
                res1[dis1[i][4]] = None

        if add != None and add1 != None:
            add_2 = []
            #             print("*********")
            #             print(len(add1))
            if len(add) >= 1 and len(add1) >= 1:
                for i in range(len(add)):
                    iou1 = iou(add[i], add1)
                    res1.append(contours1[add[i][4]])
                    if iou1 is None:
                        continue
                    add_2.append(iou1)
                    #   返回的是add1中的第几个有重叠
                for i in range(len(add1)):
                    if i in add_2:
                        continue
                    res1.append(contours2[add1[i][4]])
            elif len(add) >= 1:
                for i in range(len(add)):
                    res1.append(contours1[add[i][4]])
            else:
                for i in range(len(add1)):
                    res1.append(contours2[add1[i][4]])


        elif add != None:
            for i in range(len(add)):
                res1.append(contours1[add[i][4]])
        else:
            for i in range(len(add1)):
                res1.append(contours2[add1[i][4]])
            #       在不用的图上绘制出无法对应的与新增的轮廓的外接矩形框
    return res1, erosion1, erosion2, dis, add, dis1, add1


def small_target(edge,epsilon):
    approx = cv.approxPolyDP(edge,epsilon,True)
    points = approx.reshape((-1, 2))
    count=0
    rate=0.002
    while len(points)!=4:
        epsilon = rate * cv.arcLength(edge, True)
        rate=rate+0.002
        approx = cv.approxPolyDP(edge,epsilon,True)
        points = approx.reshape((-1, 2))
        count+=1
        if count>10:
            break
    if len(points)==4:
        pass
        # print("小目标的优化结果为4边形")
    else:
        # print("小目标的优化方法为外接最小矩形")
        rect = cv.minAreaRect(edge)
        # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
        points = cv.boxPoints(rect)
    return points


def big_building(edge,epsilon):
    epsilon = 0.005 * cv.arcLength(edge, True)
    approx = cv.approxPolyDP(edge,epsilon, True)
    points = approx.reshape((-1, 2))
    return points


def big_building1(edge,epsilon):
    epsilon = 0.004 * cv.arcLength(edge, True)
    approx = cv.approxPolyDP(edge,epsilon, True)
    points = approx.reshape((-1, 2))
    return points


def big_building2(edge,epsilon):
    epsilon = 0.002 * cv.arcLength(edge, True)
    approx = cv.approxPolyDP(edge,epsilon, True)
    points = approx.reshape((-1, 2))
    return points


def _detection( label_path):
    img = cv.imread(label_path)
    #     cimg=img[:,:,0].copy()
    cimg = img.copy()
    cimg = cv.cvtColor(cimg, cv.COLOR_BGR2GRAY)
    # RGB------>Gray
    initial_img = img.copy()
    res = cv.findContours(cimg, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
    if len(res) == 2:
        contours, idx = res
    else:
        _, contours, idx = res

    for i in range(len(contours)):
        area = cv.contourArea(contours[i])
        cv.fillPoly(initial_img, [contours[i]], (255, 255, 255))
        if area <= 100:
            # print('正在填充面积小于100的区域')
            cv.drawContours(initial_img, contours, i, 0, cv.FILLED)
            continue

    re, erode1, erode2, dis, add, dis1, add1 = detction_overlap_building(initial_img, contours, 7, 1)

    # 图片的保存
    chang_bbox = initial_img.copy()
    if dis != None:
        for i in range(len(dis)):
            cv.rectangle(chang_bbox, (dis[i][0], dis[i][1]), (dis[i][2], dis[i][3]), (0, 255, 0), 2)
    if dis1 != None:
        for i in range(len(dis1)):
            cv.rectangle(chang_bbox, (dis1[i][0], dis1[i][1]), (dis1[i][2], dis1[i][3]), (0, 255, 0), 2)

    if add1 is not None:
        for i in range(len(add1)):
            cv.rectangle(erode2, (add1[i][0], add1[i][1]), (add1[i][2], add1[i][3]), (0, 255, 0), 2)
    if add is not None:
        for i in range(len(add)):
            cv.rectangle(erode1, (add[i][0], add[i][1]), (add[i][2], add[i][3]), (0, 0, 255), 2)
    area_result = []
    contours = re
    all_coner = []
    for i in range(len(contours)):
        if contours[i] is None:
            continue

        area = cv.contourArea(contours[i])
        area_result.append(int(area))
        epsilon = 0.01 * cv.arcLength(contours[i], True)

        M = cv.moments(contours[i])
        if M["m00"] <= 10:
            # print('填充之后再次出现面小于10的区域')
            continue

        if area < 150:
            points = small_target( contours[i], epsilon=epsilon)
        elif 150 < area < 300:
            epsilon = 5 * epsilon
            approx = cv.approxPolyDP(contours[i], epsilon, True)
            points = approx.reshape((-1, 2))
        elif 3000 < area < 8000:
            points = big_building(contours[i], epsilon=epsilon)
        elif 8000 < area <= 15000:
            points = big_building1( contours[i], epsilon=epsilon)
        elif area > 15000:
            points = big_building2( contours[i], epsilon=epsilon)
        else:
            approx = cv.approxPolyDP(contours[i], epsilon, True)
            points = approx.reshape((-1, 2))
        x1 = points[:, 0]
        x1 = list(x1)
        x1.append(points[0, 0])
        y1 = points[:, 1]
        y1 = list(y1)
        y1.append(points[0, 1])
        all_coner.append([x1, y1])

    return all_coner, img.shape[0]

# if __name__ == "__main__":
#     # label_path = glob.glob(r'E:\data\model\other_data\tets\labels\*.tif')
#     # print('当前路径下待检测样本数为：{}'.format(len(label_path)))
#     all_images_x = []
#     all_images_y = []
#     all_images_shape = []
#     for i in range(5):
#         point, shape = detection()
#         print(point)
