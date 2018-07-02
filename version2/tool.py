import cv2
import numpy as np
import json
import base64
import requests

# 参数设定
# 车牌面积范围
MIN_AREA = 7000
MAX_AREA = 200000
# 车牌宽高比值
MIN_RATE = 2.6
MAX_RATE = 3.6
# 蓝色范围
lower_blue = np.array([100, 110, 110])
upper_blue = np.array([130, 255, 255])
# 黄色范围 （暂时没有用到）
lower_yellow = np.array([15, 55, 55])
upper_yellow = np.array([50, 255, 255])

# 函数返回灰度化的图像，经过处理，车牌与背景分割出来
def preProcess(src):
    # 高斯平滑，中值滤波
    gaussian = cv2.GaussianBlur(src, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    median = cv2.medianBlur(gaussian, 5)
    # 将rgb模型转化为hsv模型，方便颜色定位
    # 根据阈值找到对应颜色
    hsv = cv2.cvtColor(median, cv2.COLOR_BGR2HSV)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    blue = cv2.bitwise_and(hsv, hsv, mask=mask_blue)
    # 灰度化
    gray = cv2.cvtColor(blue, cv2.COLOR_BGR2GRAY)
    # 形态学操作：膨胀 与 腐蚀
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    dilation = cv2.dilate(gray, element, iterations=1)
    erosion = cv2.erode(dilation, element, iterations=1)
    return erosion

# 过滤掉一些不符合要求地轮廓，返回轮廓的最小包围矩形
def getRect(contours):
    result = []
    for contour in contours:
        # 计算轮廓的面积
        area = cv2.contourArea(contour)
        if area < MIN_AREA or area > MAX_AREA:
            continue
        # 求出轮廓的最小包围矩形
        rect = cv2.minAreaRect(contour)
        # 长宽
        width, height = rect[1]
        if height == 0 or width==0:
            continue
        rate = max(width, height) / min(width, height)
        # 过滤掉长宽比例不符合要求的轮廓
        if rate < MIN_RATE or rate > MAX_RATE:
            continue
        # 符合要求则添加进result
        result.append(rect)
    return result

# 通过minAreaRect得到旋转角度
def getRotation_M(rect):
    # 角度:[-90,0)
    angle = rect[2]
    # 倾斜校正
    if abs(angle) > 45 and abs(angle) < 90:
        angle = 90 - abs(angle)
    scale = 1
    # 得到旋转矩阵
    return cv2.getRotationMatrix2D(rect[0], angle, scale)

# 将原图进行旋转
def rotateSrc(src, rotation_M):
    desImg = src.copy()
    rows, cols, depth = src.shape
    desImg = cv2.warpAffine(src, rotation_M, (rows+200, cols+200))
    return  desImg


# 根据矩形在原图裁剪图片
# src 原图  rect 最小包围矩形 padding 缩小矩形范围
def cutSrcByRect(src, rect, padding=0):
    box = cv2.boxPoints(rect)
    min_x = int(min(box[:, 0])) + padding
    max_x = int(max(box[:, 0])) - padding
    min_y = int(min(box[:, 1])) + padding
    max_y = int(max(box[:, 1])) - padding
    target = src[min_y:max_y, min_x:max_x]
    return target


#### 通过百度ai接口进行图片识别 输入参数access_token有效期为一个月，一个月之后需要重新获取
def getCarNumberByBaiDuAI(access_token, picturePath):
    url = "https://aip.baidubce.com/rest/2.0/ocr/v1/license_plate?access_token="
    #url = "https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic?access_token="
    url += access_token
    headers = {}
    headers["Content-Type"] = "application/x-www-form-urlencoded"
    data = {}
    data["multi_detect"] = False # 图像中是否有多个车牌需要识别
    with open(picturePath, "rb") as f:
        # b64encode是编码，b64decode是解码
        base64_data = base64.b64encode(f.read())
    data["image"] = base64_data
    response = requests.post(url, data=data, headers=headers)
    html = response.content
    html_doc = str(html, 'utf-8')
    result = json.loads(html_doc)
    return result


if __name__ == '__main__':
    baiduAIAccessToken = '24.8f04d71f6aab2e42083a20c78e3b7b4e.2592000.1532249479.282335-11432255'
    path = './testPicture/'
    tempPath = './tempPicture/'
    outputPath = './outputPicture/'
    for i in range(1,8):
        name = str(i) + '.jpg'
        src = cv2.imread(path+name)
        afterPreProcess = preProcess(src)
        pic, contours, hierarchy = cv2.findContours(afterPreProcess, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rect = getRect(contours)
        if len(rect) != 1:
            print('最小包围矩形有多个，系统无法识别')
        else:
            Rotation_M = getRotation_M(rect[0])
            rotatePic = rotateSrc(src, Rotation_M)
            cv2.imwrite(tempPath+name, rotatePic)
            src = cv2.imread(tempPath+name)
            afterPreProcess = preProcess(src)
            pic, contours, hierarchy = cv2.findContours(afterPreProcess, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rect = getRect(contours)
            if len(rect)!=1:
                print('最小包围矩形有多个，系统无法识别')
            else:
                result = cutSrcByRect(src, rect[0], -5)
                cv2.imwrite(outputPath+name, result)
                print(name,"成功截图")
                print(outputPath+name)
                print(getCarNumberByBaiDuAI(baiduAIAccessToken, outputPath+name))

