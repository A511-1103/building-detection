import numpy as np
import glob
import matplotlib.pyplot as plt
import cv2 as cv
import time
import  os


def fill_and_delete(label):
    gray_label = label[:,:,0].copy()
    # print(np.unique(gray_label))
    res1=cv.findContours(gray_label,mode=cv.RETR_EXTERNAL,method=cv.CHAIN_APPROX_NONE)
    contours,idx1=res1
    # print(len(contours))

    for i in range(len(contours)):
        area=cv.contourArea(contours[i])
        cv.fillPoly(gray_label,[contours[i]],(255,255,255))
        '''
        填补空洞
        '''
        if area<=1000:
            # print('正在填充面积小于100的区域')
            cv.drawContours(gray_label,contours,i,0,cv.FILLED)
            continue

    res1=cv.findContours(gray_label,mode=cv.RETR_EXTERNAL,method=cv.CHAIN_APPROX_NONE)
    contours1,idx1=res1
    # print(len(contours1))
    plt.imshow(gray_label)
    cv.imwrite('gray.png',gray_label)
    return gray_label,contours1


def dilate_process(h, w, contours, kernel, iter_time):
    result = []
    for j in range(len(contours)):
        cur_img = np.zeros((h, w), dtype=np.uint8)
        #         print(len(contours))
        cv.drawContours(cur_img, contours, j, 255, cv.FILLED)
        dilate1 = cv.dilate(cur_img, kernel, iterations=iter_time)
        res = cv.findContours(dilate1, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
        if len(res) == 2:
            contours1, _ = res
        else:
            _, contours1, _ = res
        result.append(contours1[0])
    #     print('膨胀阶段')
    return result


def fill_small_target(img, contours):
    fill_flag = False
    for i in range(len(contours)):
        area = cv.contourArea(contours[i])
        cv.fillPoly(img, [contours[i]], (255, 255, 255))
        if area <= 500:
            fill_flag = True
            #             print('正在填充腐蚀之后面积小于1000的区域')
            cv.drawContours(img, contours, i, 0, cv.FILLED)
            continue
    return img, fill_flag


def erode_process(img, kernel_size, iteration):
    erode = img.copy()
    kernel = np.ones((1, kernel_size), np.uint8)
    erosion1 = cv.erode(erode, kernel, iterations=iteration)
    contours1, _ = cv.findContours(erosion1, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
    if len(contours1) == 1:
        #         print('此处没有发生水平方向的重叠')
        return None
    else:
        #         print('此物体发生了水平方向的重叠')
        erosion1, flag = fill_small_target(erosion1, contours1)
        if not flag:
            #             print('没有可以填充的小物体存在')
            h, w = img.shape
            cnt = dilate_process(h, w, contours1, kernel, iteration)
            return cnt
        else:
            contours1, _ = cv.findContours(erosion1, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
            if len(contours1) == 0:
                return False
            h, w = img.shape
            cnt = dilate_process(h, w, contours1, kernel, iteration)
            return cnt


def erode_process1(img, kernel_size, iteration):
    erode = img.copy()
    kernel = np.ones((kernel_size, 1), np.uint8)
    erosion1 = cv.erode(erode, kernel, iterations=iteration)
    contours1, _ = cv.findContours(erosion1, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
    if len(contours1) == 1:
        #         print('此处没有发生竖直方向的重叠')
        return None
    else:
        #         print('此物体发生了竖直方向的重叠')
        erosion1, flag = fill_small_target(erosion1, contours1)
        if not flag:
            #             print('没有可以填充的小物体存在')
            h, w = img.shape
            cnt = dilate_process(h, w, contours1, kernel, iteration)
            '''
            返回了多个目标的cnt
            '''
            return cnt

        else:
            contours1, _ = cv.findContours(erosion1, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
            h, w = img.shape
            if len(contours1) == 0:
                return False
            cnt = dilate_process(h, w, contours1, kernel, iteration)

            return cnt


def iou(bbox1, bbox2):
    '''
    bbox1---->[4,]
    bbox2---->[N,4]
    '''
    wh = np.minimum(bbox1[2:], bbox2[:, 2:]) - np.maximum(bbox1[:2], bbox2[:, :2])
    wh = np.maximum(wh, 0)

    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1])
    union_area = wh[:, 0] * wh[:, 1]

    res_iou = union_area / (bbox1_area + bbox2_area - union_area)

    return res_iou


def compute_iou(cnt1, cnt2):
    all_bbox1 = []
    for i in range(len(cnt1)):
        x1, y1, w1, h1 = cv.boundingRect(cnt1[i])
        x2 = x1 + w1
        y2 = y1 + h1
        all_bbox1.append([x1, y1, x2, y2])
    all_bbox1 = np.array(all_bbox1)

    all_bbox2 = []
    for i in range(len(cnt2)):
        x1, y1, w1, h1 = cv.boundingRect(cnt2[i])
        x2 = x1 + w1
        y2 = y1 + h1
        all_bbox2.append([x1, y1, x2, y2])
    all_bbox2 = np.array(all_bbox2)

    valid_cnt1_index = []
    valid_cnt2_index = []

    all_mask = [False] * len(all_bbox2)
    for i in range(len(all_bbox1)):
        res_iou = iou(all_bbox1[i], all_bbox2)
        iou_mask = res_iou > 0.7
        all_mask += iou_mask
        if not np.any(iou_mask):
            valid_cnt1_index.append(i)

    for i in range(len(all_mask)):
        if all_mask[i]:
            continue
        valid_cnt2_index.append(i)

    return valid_cnt1_index, valid_cnt2_index


def eroede_dilate_process(gray_label, contours1):
    h, w = gray_label.shape
    all_cnt = []
    for i in range(len(contours1)):
        plot_img = np.zeros((h, w), dtype=np.uint8)
        cv.drawContours(plot_img, contours1, i, 255, cv.FILLED)
        #         plt.imshow(plot_img)
        cur_cnt = erode_process(plot_img, 5, 5)
        cur_cnt1 = erode_process1(plot_img, 5, 5)

        if cur_cnt == False or cur_cnt1 == False:
            continue
        elif cur_cnt is None and cur_cnt1 is None:
            all_cnt.append(contours1[i])
            continue
        #           没有新目标的产生,默认使用原始的cnt
        elif cur_cnt is not None and cur_cnt1 is not None:
            #         dely_compute_iou

            for k in cur_cnt:
                all_cnt.append(k)
            for k in cur_cnt1:
                all_cnt.append(k)

            '''
            valid_1,valid_2 = compute_iou(cur_cnt,cur_cnt1)
            if len(valid_1)>=1:
                for m in valid_1:
                    all_cnt.append(cur_cnt[m])

            if len(valid_2)>=1:
                for n in valid_2:
                    all_cnt.append(cur_cnt1[n])
            '''
            continue
            '''此处函数待验证是否有BUG以及是否实现所需功能'''
        elif cur_cnt is not None:
            for j in range(len(cur_cnt)):
                all_cnt.append(cur_cnt[j])
            continue
        else:
            for j in range(len(cur_cnt1)):
                all_cnt.append(cur_cnt1[j])
    #         plt.show()
    #         plt.cla()
    return all_cnt


def small_target(input_img,edge,epsilon):
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
        print("小目标的优化结果为4边形")
    else:
        print("小目标的优化方法为外接最小矩形")
        rect = cv.minAreaRect(edge)
        # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
        points = cv.boxPoints(rect)
    return points


def big_building(img,edge,epsilon):
    epsilon = 0.005 * cv.arcLength(edge, True)
    approx = cv.approxPolyDP(edge,epsilon, True)
#     points = approx.reshape((-1, 2))
    return approx


def big_building1(img,edge,epsilon):
    epsilon = 0.004 * cv.arcLength(edge, True)
    approx = cv.approxPolyDP(edge,epsilon, True)
#     points = approx.reshape((-1, 2))
    return approx


def big_building2(img,edge,epsilon):
    epsilon = 0.002 * cv.arcLength(edge, True)
    approx = cv.approxPolyDP(edge,epsilon, True)
#     points = approx.reshape((-1, 2))
    return approx


def only_plt(input_img,all_cnt):
    for i in range(len(all_cnt)):
        cv.drawContours(input_img,all_cnt,i,255,cv.FILLED)
    return input_img


def model_confuse(path, name=''):
    '''
    save_path = 'all_result' + '/' + dir_name
    print(save_path)
    os.makedirs(save_path, exist_ok=True)
    '''

    all_path = glob.glob(path + '/' + '*.png')
    print(all_path)

    if len(all_path) != 5:
        print('no five images')
        return

    label = cv.imread(all_path[0])
    gray_label, cnt = fill_and_delete(label)
    all_cnt = eroede_dilate_process(gray_label, cnt)
    label = np.zeros((label.shape))
    l1 = only_plt(label.copy(), all_cnt)

    label = cv.imread(all_path[1])
    gray_label, cnt = fill_and_delete(label)
    all_cnt = eroede_dilate_process(gray_label, cnt)
    label = np.zeros((label.shape))
    l2 = only_plt(label.copy(), all_cnt)

    label = cv.imread(all_path[2])
    gray_label, cnt = fill_and_delete(label)
    all_cnt = eroede_dilate_process(gray_label, cnt)
    label = np.zeros((label.shape))
    l3 = only_plt(label.copy(), all_cnt)

    label = cv.imread(all_path[3])
    gray_label, cnt = fill_and_delete(label)
    all_cnt = eroede_dilate_process(gray_label, cnt)
    label = np.zeros((label.shape))
    l4 = only_plt(label.copy(), all_cnt)

    label = cv.imread(all_path[4])
    gray_label, cnt = fill_and_delete(label)
    all_cnt = eroede_dilate_process(gray_label, cnt)
    label = np.zeros((label.shape))
    l5 = only_plt(label.copy(), all_cnt)

    final_label = l1 // 255 + l2 // 255 + l3 // 255 + l4 // 255 + l5 // 255

    # cv.imwrite(save_path + '/' + 'sliding_bam1.png', l1)
    # cv.imwrite(save_path + '/' + 'sliding_deep1.png', l2)
    # cv.imwrite(save_path + '/' + 'sliding_scse1.png', l3)
    # cv.imwrite(save_path + '/' + 'sliding_res341.png', l4)
    # cv.imwrite(save_path + '/' + 'sliding_hrnet1.png', l5)

    label = np.where(final_label >= 3, 255, 0)
    label = np.array(label, np.uint8)
    # print(label.dtype)

    def only_plt1(input_img, all_cnt):
        all_color = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255],
                     [255, 255, 255]]
        for i in range(len(all_cnt)):
            colors = all_color[i % 7]
            cv.drawContours(image=input_img,
                            contours=[all_cnt[i]],
                            contourIdx=-1,
                            color=colors,
                            thickness=3)
        return input_img

    gray_label, cnt = fill_and_delete(label)
    all_cnt = eroede_dilate_process(gray_label, cnt)
    # plot_res = only_plt1(images, all_cnt)

    label = np.zeros(gray_label.shape,np.uint8)
    for i in range(len(all_cnt)):
        cv.drawContours(label, all_cnt, i, 255, cv.FILLED)


    # h, w, c = plot_res.shape

    cv.imwrite(path + r'\{}_result.png'.format(name),label)
    # cv.imwrite(save_path + '/' + 'sliding_plot_images.png', l1)

    '''
    top = 0
    down = 512
    left = 0
    right = 512
    while down <= h:
        #     left = 0
        #     right = 512
        while right <= w:
            cv.line(plot_res, (left, 0), (left, h), (0, 255, 0), 2, 8)
            cv.line(plot_res, (right, 0), (right, h), (255, 0, 0), 2, 8)
            left = right - int(512 * 0.5)
            right = left + 512

        cv.line(plot_res, (0, top), (w, top), (255, 255, 255), 2, 8)

        cv.line(plot_res, (0, down), (w, down), (0, 0, 0), 2, 8)

        top = down - int(512 * 0.5)
        down = top + 512
        print(top, down)

    cv.imwrite(save_path + '/' + 'sliding_plot_images.png', plot_res)
    '''


if __name__ == '__main__':

    name = os.listdir('D:/res_image')
    # 名称

    for i in range(len(name)):
        path = 'D:/res_image' +'/'+ name[i]
        #     绝对路劲
    #     child path
        model_confuse(path,name[i])
        print('图片{}预测结束'.format(name[i]))