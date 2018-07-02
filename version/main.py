import cv2
import numpy as np
import base64
import requests
import json

#### 图像预处理函数
def preProcess(src):
    # # 调试用代码块
    # # 功能：显示处理过后的图像
    # cv2.namedWindow("src", cv2.WINDOW_NORMAL)
    # cv2.imshow("src", src)
    # # 调试用代码块

    # 灰度化
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # # 调试用代码块
    # # 功能：显示处理过后的图像
    # cv2.namedWindow("gray", cv2.WINDOW_NORMAL)
    # cv2.imshow("gray", gray)
    # # 调试用代码块

    # 高斯平滑，中值滤波处理
    gaussian = cv2.GaussianBlur(gray, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    median = cv2.medianBlur(gaussian, 5)
    # # 调试用代码块
    # # 功能：显示处理过后的图像
    # cv2.namedWindow("median", cv2.WINDOW_NORMAL)
    # cv2.imshow("median", median)
    # # 调试用代码块

    # Sobel边缘检测
    sobel = cv2.Sobel(median, cv2.CV_8U, 1, 0, ksize=3)
    # # 调试用代码块
    # # 功能：显示处理过后的图像
    # cv2.namedWindow("sobel", cv2.WINDOW_NORMAL)
    # cv2.imshow("sobel", sobel)
    # # 调试用代码块

    # 二值化
    ret, binary = cv2.threshold(sobel, 170, 255, cv2.THRESH_BINARY)

    # 以下膨胀和腐蚀的参数需要自己测试
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 6))
    # 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element2, iterations=10)
    # # 调试用代码块
    # # 功能：显示处理过后的图像
    # cv2.namedWindow("peng", cv2.WINDOW_NORMAL)
    # cv2.imshow("peng", dilation)
    # # 调试用代码块
    # 腐蚀一次，去掉细节
    erosion = cv2.erode(dilation, element1, iterations=12)
    # # 调试用代码块
    # # 功能：显示处理过后的图像
    # cv2.namedWindow("fu", cv2.WINDOW_NORMAL)
    # cv2.imshow("fu", erosion)
    # # 调试用代码块
    # 再次膨胀，让轮廓明显一些
    dilation2 = cv2.dilate(erosion, element2, iterations=14)
    # # 调试用代码块
    # # 功能：显示处理过后的图像
    # cv2.namedWindow("dilation2", cv2.WINDOW_NORMAL)
    # cv2.imshow('dilation2', dilation2)
    # # 调试用代码块
    return  dilation2


#### 轮廓筛选函数 需要输入预处理并且二值化的图像
def filterContours(src, afterPreProcess):
    targets = [];
    pic, contours, hierarchy = cv2.findContours(afterPreProcess, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历每一个轮廓 过滤掉面积过大或者过小的轮廓此处取值[130000, 260000]
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area>=130000 and area<=260000:
            # 将轮廓近似为矩形
            epsilon = 0.15 * cv2.arcLength(cnt, True)
            # approx 存储了矩形的对角点
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            for j in range(0, len(approx), 2):
                point_m = (approx[j][0][0], approx[j][0][1])
                point_n = (approx[j + 1][0][0], approx[j + 1][0][1])
                # # 调试用代码 begin
                # # 输出矩形的点坐标
                # print(point_m, point_n)
                # # 调试用代码 end
                # # 调试用代码 begin
                # # 可以在原图画出approx所代表的矩形
                # cv2.rectangle(src, point_m, point_n, (0, 0, 255), 3)
                # # 调试用代码 end
                min_x = min(point_n[0], point_m[0])
                max_x = max(point_n[0], point_m[0])
                min_y = min(point_n[1], point_m[1])
                max_y = max(point_n[1], point_m[1])
                # 这一步是为了y方向上切多一些
                if min_y - 50 < 0:
                    min_y = 0
                else:
                    min_y = min_y - 50
                temp = src[min_y:max_y+20, min_x:max_x]
                targets.append(temp)
    return targets

# 根据车牌的颜色直方图过滤
def checkByColor(img):
    # # 调试用代码块 begin
    # # 功能：显示图像灰度分布直方图
    # plt.hist(img.ravel(),256,[0,256]);
    # plt.show()
    # # 调试用代码块 end

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    maxColor = np.argmax(np.bincount(gray.ravel()))
    if maxColor>= 64 and maxColor<=129:
        return True
    else:
        return False

#### 通过百度ai接口进行图片识别 输入参数access_token有效期为一个月，一个月之后需要重新获取
def getCarNumberByBaiDuAI(access_token, picturePath):
    url = "https://aip.baidubce.com/rest/2.0/ocr/v1/license_plate?access_token="
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
    prefixPath = './InputPicture/'
    pictureNum = 9
    suffixPath = '.jpg'
    outputDir = './OutputPicture/'

    for picIndex in range(1, pictureNum):
        src = cv2.imread(prefixPath + str(picIndex) + suffixPath)
        afterPreProcess = preProcess(src)
        targets = filterContours(src, afterPreProcess)
        if len(targets)==1:
            picName = outputDir+str(picIndex)+"_0"+".jpg"
            cv2.imwrite(picName, targets[0])
            result = getCarNumberByBaiDuAI(baiduAIAccessToken, picName)
            if 'error_code' in result:
                print("--------  " + picName)
                print("!!!!错误：百度ai识别失败，错误信息为", result)
            else:
                print("--------  " + picName)
                print("颜色：" + result['words_result']['color'])
                print("号码：" + result['words_result']['number'])
        else:
            for i in range(len(targets)):
                picName = outputDir+str(picIndex)+"_"+str(i)+".jpg"
                if checkByColor(targets[i]):
                    cv2.imwrite(picName, targets[i])
                    result = getCarNumberByBaiDuAI(baiduAIAccessToken, picName)
                    if 'error_code' in result:
                        print("--------  " + picName)
                        print("!!!!错误：百度ai识别失败，错误信息为", result)
                    else:
                        print("--------  " + picName)
                        print("颜色：" + result['words_result']['color'])
                        print("号码：" + result['words_result']['number'])

