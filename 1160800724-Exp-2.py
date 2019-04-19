import numpy as np
import cv2
def read_pic(filepath):
    img = cv2.imread(filepath)
    imggray = cv2.imread(filepath,0)
    return img,imggray

def regu(num):
    if(num>255):
        return 255
    elif num < 0:
        return 0
    else :
        return num

'''
修改图像的亮度，brightness取值0～2 <1表示变暗 >1表示变亮
'''
def change_brightness(filepath,brightness):
    img,imggray = read_pic(filepath)
    averB = 0
    averG = 0
    averR = 0
    for i in range(abs(img.shape[0])):
        for j in range(abs(img.shape[1])):
            averB = averB + img[i,j,0]
            averG = averG + img[i,j,1]
            averR = averR + img[i,j,2]
    averB = averB/img.size
    averG = averG/img.size
    averR = averR/img.size
    for i in range(abs(img.shape[0])):
        for j in range(abs(img.shape[1])):
            img[i,j,0] = int(regu(img[i,j,0]+(brightness-1)*averB))
            img[i, j, 1] = int(regu(img[i, j, 1] + (brightness - 1) * averG))
            img[i, j, 2] = int(regu(img[i, j, 2] + (brightness - 1) * averR))
    return img

'''
修改图像的对比度,coefficent>0, <1降低对比度,>1提升对比度 建议0-2
'''
def change_contrast(filepath,coefficent):
    img,imggray = read_pic(filepath)
    cal = 0
    for i in range(abs(imggray.shape[0])):
        for j in range(abs(imggray.shape[1])):
           cal = cal + imggray[i,j]
    cal = cal / imggray.size
    for i in range(abs(imggray.shape[0])):
        for j in range(abs(imggray.shape[1])):
            newgray = cal + coefficent * (imggray[i,j] - cal)
            oldgray = imggray[i,j]
            if(oldgray<0.00001):
                img[i,j,0] = 0
                img[i,j,1] = 0
                img[i,j,2] = 0
            else:
                img[i, j, 0] = int(regu(img[i, j, 0] * newgray / oldgray))
                img[i, j, 1] = int(regu(img[i, j, 1] * newgray / oldgray))
                img[i, j, 2] = int(regu(img[i, j, 2] * newgray / oldgray))
    return img

'''
修改图像饱和度，coefficient取值>0，建议取值为0～2
'''
def change_saturation(filepath,coefficient):
    img,imggray = read_pic(filepath)
    rgb = np.zeros((img.shape),np.float64)
    for i in range(abs(img.shape[0])):
        for j in range(abs(img.shape[1])):
            rgb[i,j,0] = img[i,j,0]/255
            rgb[i, j, 1] = img[i, j, 1] / 255
            rgb[i, j, 2] = img[i, j, 2] / 255
            themax = max([rgb[i,j,0],rgb[i,j,1],rgb[i,j,2]])
            themin = min([rgb[i,j,0],rgb[i,j,1],rgb[i,j,2]])
            L = (themax+themin)/2
            if themax==themin:
                S = 0
                H = 0
            else:
                if L<0.5:
                    S = (themax - themin)/(themax + themin)
                else :
                    S = (themax - themin)/(2 - themax - themin)
                if rgb[i,j,2] == themax:
                    H = (rgb[i,j,1]-rgb[i,j,0])/(themax-themin)
                elif rgb[i,j,1]==themax:
                    H = 2+(rgb[i,j,0]-rgb[i,j,2])/(themax-themin)
                else :
                    H = 4+(rgb[i,j,2]-rgb[i,j,1])/(themax-themin)
                if H < 0:
                    H = H*60+360
                else :
                    H = H*60
            S = S*coefficient
            if S == 0:
                img[i,j,0] = 0;img[i,j,1]=0;img[i,j,2]=0
            else :
                if L<0.5:
                    temp2 = L*(1+S)
                else:
                    temp2 = L+S-L*S
                temp1 = 2*L-temp2
                H = H/360
                temp3 = H+1/3
                if temp3 < 0:
                    temp3 = temp3+1
                if temp3 > 1:
                    temp3 = temp3-1
                if 6*temp3<1:
                    img[i,j,2] = int(regu((temp1 + (temp2-temp1)*6*temp3)*255))
                elif 2*temp3 < 1:
                    img[i,j,2] = int(regu(temp2*255))
                elif temp3*3 <2 :
                    img[i,j,2] = int(regu((temp1 + (temp2-temp1)*((2/3)-temp3)*6)*255))
                else :
                    img[i,j,2] = int(regu(temp1*255))
                temp3 = H
                if temp3 < 0:
                    temp3 = temp3 + 1
                if temp3 > 1:
                    temp3 = temp3 - 1
                if 6 * temp3 < 1:
                    img[i, j, 1] = int(regu((temp1 + (temp2 - temp1) * 6 * temp3) * 255))
                elif 2 * temp3 < 1:
                    img[i, j, 1] = int(regu(temp2 * 255))
                elif temp3 * 3 < 2:
                    img[i, j, 1] = int(regu((temp1 + (temp2 - temp1) * ((2 / 3) - temp3) * 6) * 255))
                else:
                    img[i, j, 1] = int(regu(temp1 * 255))
                temp3 = H-1/3
                if temp3 < 0:
                    temp3 = temp3 + 1
                if temp3 > 1:
                    temp3 = temp3 - 1
                if 6 * temp3 < 1:
                    img[i, j, 0] = int(regu((temp1 + (temp2 - temp1) * 6 * temp3) * 255))
                elif 2 * temp3 < 1:
                    img[i, j, 0] = int(regu(temp2 * 255))
                elif temp3 * 3 < 2:
                    img[i, j, 0] = int(regu((temp1 + (temp2 - temp1) * ((2 / 3) - temp3) * 6) * 255))
                else:
                    img[i, j, 0] = int(regu(temp1 * 255))
    return img

'''
转到HSV，然后进行。-360<sum<360
'''
def change_chroma(filepath,sum):
    img, imggray = read_pic(filepath)
    rgb = np.zeros((img.shape), np.float64)
    for i in range(abs(img.shape[0])):
        for j in range(abs(img.shape[1])):
            rgb[i, j, 0] = img[i, j, 0] / 255
            rgb[i, j, 1] = img[i, j, 1] / 255
            rgb[i, j, 2] = img[i, j, 2] / 255
            themax = max([rgb[i, j, 0], rgb[i, j, 1], rgb[i, j, 2]])
            themin = min([rgb[i, j, 0], rgb[i, j, 1], rgb[i, j, 2]])
            L = (themax + themin) / 2
            if themax == themin:
                S = 0
                H = 0
            else:
                if L < 0.5:
                    S = (themax - themin) / (themax + themin)
                else:
                    S = (themax - themin) / (2 - themax - themin)
                if rgb[i, j, 2] == themax:
                    H = (rgb[i, j, 1] - rgb[i, j, 0]) / (themax - themin)
                elif rgb[i, j, 1] == themax:
                    H = 2 + (rgb[i, j, 0] - rgb[i, j, 2]) / (themax - themin)
                else:
                    H = 4 + (rgb[i, j, 2] - rgb[i, j, 1]) / (themax - themin)
                if H < 0:
                    H = H * 60 + 360
                else:
                    H = H * 60
            if S == 0:
                img[i, j, 0] = 0;
                img[i, j, 1] = 0;
                img[i, j, 2] = 0
            else:
                if L < 0.5:
                    temp2 = L * (1 + S)
                else:
                    temp2 = L + S - L * S
                temp1 = 2 * L - temp2
                H = H / 360 + sum/360
                temp3 = H + 1 / 3
                if temp3 < 0:
                    temp3 = temp3 + 1
                if temp3 > 1:
                    temp3 = temp3 - 1
                if 6 * temp3 < 1:
                    img[i, j, 2] = int(regu((temp1 + (temp2 - temp1) * 6 * temp3) * 255))
                elif 2 * temp3 < 1:
                    img[i, j, 2] = int(regu(temp2 * 255))
                elif temp3 * 3 < 2:
                    img[i, j, 2] = int(regu((temp1 + (temp2 - temp1) * ((2 / 3) - temp3) * 6) * 255))
                else:
                    img[i, j, 2] = int(regu(temp1 * 255))
                temp3 = H
                if temp3 < 0:
                    temp3 = temp3 + 1
                if temp3 > 1:
                    temp3 = temp3 - 1
                if 6 * temp3 < 1:
                    img[i, j, 1] = int(regu((temp1 + (temp2 - temp1) * 6 * temp3) * 255))
                elif 2 * temp3 < 1:
                    img[i, j, 1] = int(regu(temp2 * 255))
                elif temp3 * 3 < 2:
                    img[i, j, 1] = int(regu((temp1 + (temp2 - temp1) * ((2 / 3) - temp3) * 6) * 255))
                else:
                    img[i, j, 1] = int(regu(temp1 * 255))
                temp3 = H - 1 / 3
                if temp3 < 0:
                    temp3 = temp3 + 1
                if temp3 > 1:
                    temp3 = temp3 - 1
                if 6 * temp3 < 1:
                    img[i, j, 0] = int(regu((temp1 + (temp2 - temp1) * 6 * temp3) * 255))
                elif 2 * temp3 < 1:
                    img[i, j, 0] = int(regu(temp2 * 255))
                elif temp3 * 3 < 2:
                    img[i, j, 0] = int(regu((temp1 + (temp2 - temp1) * ((2 / 3) - temp3) * 6) * 255))
                else:
                    img[i, j, 0] = int(regu(temp1 * 255))
    return img

'''
得到图片灰度图的直方图，此处直方图是一个大小为256的numpy数组
'''
def get_diagram(filepath):
    img,imggray = read_pic(filepath)
    returns = np.zeros(256)
    for i in range(abs(imggray.shape[0])):
        for j in range(abs(imggray.shape[1])):
            returns[imggray[i,j]] = returns[imggray[i,j]]+1
    print(returns)

'''
普通中值滤波，
'''
def middle_filter(filepath,parameter):
    img, imggray= read_pic(filepath)
    for i in range(1,abs(img.shape[0])-1):
        for j in range(1,abs(img.shape[1])-1):
            thelist0 = [img[i-1,j-1,0],img[i,j-1,0],img[i+1,j-1,0],img[i-1,j,0],img[i+1,j,0],
                       img[i -1, j +1,0],img[i,j+1,0],img[i+1,j+1,0]]
            aver0 = sum(thelist0)/8
            if abs(img[i,j,0]-aver0)>parameter:
                middle0 = sorted(thelist0)[4]
                img[i,j,0] = int(regu(middle0))
            thelist1 = [img[i - 1, j - 1, 1], img[i, j - 1, 1], img[i + 1, j - 1, 1], img[i - 1, j, 1],
                        img[i + 1, j, 1],img[i - 1, j + 1, 1], img[i, j + 1, 1], img[i + 1, j + 1, 1]]
            aver1 = sum(thelist1) / 8
            if abs(img[i, j, 1] - aver1) > parameter:
                middle1 = sorted(thelist1)[4]
                img[i, j, 1] = int(regu(middle1))
            thelist2 = [img[i - 1, j - 1, 2], img[i, j - 1, 2], img[i + 1, j - 1, 2], img[i - 1, j, 2],
                        img[i + 1, j, 2], img[i - 1, j + 1, 2], img[i, j + 1, 2], img[i + 1, j + 1, 2]]
            aver2 = sum(thelist2) / 8
            if abs(img[i, j, 2] - aver2) > parameter:
                middle2 = sorted(thelist2)[4]
                img[i, j, 2] = int(regu(middle2))
    return img

'''
parameter为差值多少进行滤波
'''
def equal_filter(filepath,parameter):
    img, imggray = read_pic(filepath)
    for i in range(1, abs(img.shape[0])-1):
        for j in range(1, abs(img.shape[1])-1):
            thelist0 = [img[i - 1, j - 1, 0], img[i, j - 1, 0], img[i + 1, j - 1, 0], img[i - 1, j, 0],
                        img[i + 1, j, 0],
                        img[i - 1, j + 1, 0], img[i, j + 1, 0], img[i + 1, j + 1, 0]]
            aver0 = sum(thelist0) / 8
            if abs(img[i, j, 0] - aver0) > parameter:
                img[i, j, 0] = int(regu(aver0))
            thelist1 = [img[i - 1, j - 1, 1], img[i, j - 1, 1], img[i + 1, j - 1, 1], img[i - 1, j, 1],
                        img[i + 1, j, 1], img[i - 1, j + 1, 1], img[i, j + 1, 1], img[i + 1, j + 1, 1]]
            aver1 = sum(thelist1) / 8
            if abs(img[i, j, 1] - aver1) > parameter:
                img[i, j, 1] = int(regu(aver1))
            thelist2 = [img[i - 1, j - 1, 2], img[i, j - 1, 2], img[i + 1, j - 1, 2], img[i - 1, j, 2],
                        img[i + 1, j, 2], img[i - 1, j + 1, 2], img[i, j + 1, 2], img[i + 1, j + 1, 2]]
            aver2 = sum(thelist2) / 8
            if abs(img[i, j, 2] - aver2) > parameter:
                img[i, j, 2] = int(regu(aver2))
    return img

'''
应用roberts算子进行边缘检测
'''
def RobertsOperator(roi):
    operator_first = np.array([[-1,0],[0,1]])
    operator_second = np.array([[0,-1],[1,0]])
    return np.abs(np.sum(roi[1:,1:]*operator_first))+np.abs(np.sum(roi[1:,1:]*operator_second))

def RobertAlpgrithm(image):
    image = cv2.copyMakeBorder(image,1,1,1,1,cv2.BORDER_DEFAULT)
    for i in range(1,image.shape[0]):
        for j in range(1,image.shape[1]):
            image[i,j] = RobertsOperator(image[i-1:i+2,j-1:j+2])
    return image[1:image.shape[0],1:image.shape[1]]

def Roberts(filepath):
    saber,gray_saber = read_pic(filepath)
    gray_saber = cv2.resize(gray_saber,(200,200))
    Robert_saber = RobertAlpgrithm(gray_saber)
    return Robert_saber

'''
应用sobel算子进行边缘检测 operation_type为1或2
2是vertical
'''
def SobelOperator(roi, operator_type):
    if operator_type == 1:
        sobel_operator = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    elif operator_type == 2:
        sobel_operator = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    else:
        raise ("type Error")
    result = np.abs(np.sum(roi * sobel_operator))
    return result

def SobelAlogrithm(image, operator_type):
    new_image = np.zeros(image.shape)
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            new_image[i - 1, j - 1] = SobelOperator(image[i - 1:i + 2, j - 1:j + 2], operator_type)
    new_image = new_image * (255 / np.max(image))
    return new_image.astype(np.uint8)

def Sobel(filepath,operation):
    saber,gray_saber = read_pic(filepath)
    gray_saber = cv2.resize(gray_saber,(200,200))
    return SobelAlogrithm(gray_saber,operation)

def fast_middle_filter(filepath,parameter):
    img,imggray = read_pic(filepath)
    for q in range(3):
        for i in range(1, abs(img.shape[0]) - 2):
            diagram = np.zeros(256)
            a = []
            j = 1
            for k in range(-1, 2):
                for l in range(-1, 2):
                    a.append(img[i + l, j + k, q])
                    diagram[img[i + l, j + k, q]] += 1
            a.pop(4)
            diagram[img[i,j,q]] = diagram[img[i,j,q]]-1
            b = []
            for ii in range(8):
                b.append(a[ii])
            middle = b[4]
            aver = sum(a) / 8
            if abs(img[i, j, q] - aver) > parameter:
                img[i, j, q] = middle
            nm = 0
            for iii in a:
                if iii < middle:
                    nm = nm+1
            while (j < abs(img.shape[1]) - 2):
                for k1 in range(-1, 2):
                    diagram[img[i + k1, j - 1, q]] = diagram[img[i + k1, j - 1, q]] - 1
                    a.pop(0)
                    if img[i + k1, j - 1, q] < middle:
                        nm = nm - 1
                a.insert(img[i,j,q],1)
                diagram[img[i,j,q]] = diagram[img[i,j,q]]+1
                if img[i,j,q]<middle:
                    nm = nm+1
                for k1 in range(-1, 2):
                    diagram[img[i + k1, j + 2, q]] = diagram[img[i + k1, j + 2, q]] + 1
                    a.append(img[i + k1, j + 2, q])
                    if (img[i + k1, j + 2, q] < middle):
                        nm = nm + 1
                a.pop(4)
                diagram[img[i,j+1,q]] = diagram[img[i,j+1,q]]-1
                if img[i,j+1,q]<middle:
                    nm = nm-1
                j = j + 1
                while nm > 4:
                    if middle == 0:
                        print('sss')
                        break
                    else:
                        nm = nm - diagram[middle-1]
                        if nm <= 4:
                            middle = middle-1
                            break
                        else:
                            middle = middle - 1
                while nm < 4:
                    if middle == 255:
                        break
                    else:
                        nm = nm + diagram[middle]
                        if nm > 4:
                            break
                        else:
                            middle = middle + 1
                aver = sum(a) / 8
                if abs(img[i, j, q] - aver) > parameter:
                    img[i, j, q] = middle
                nm = 0
                for iii in a:
                    if iii < middle:
                        nm = nm + 1
    return img

if __name__ == "__main__":
    while True:
        print("*********************************************************")
        print('请输入选择的操作：1-调整图像亮度    2-调整图像对比度')
        print("3-调整图像饱和度    4-调整图像色度     5-得到图像直方图")
        print("6-应用中值滤波增强图像     7-应用均值滤波增强图像")
        print('8-应用Roberts算子进行边缘检测   9-应用Sobel算子进行边缘检测')
        print('10-中值滤波的快速算法      0-退出程序')
        print('注：若操作需要参数，则实现效果和参数相关')
        print('********************************************************')
        choice = input('请输入选择的操作：')
        if choice=='1':
            filepath = input('请输入图像路径：')
            brightness = float(input('请输入亮度参数，取值0-2，brightness<1表示变暗，brightness>1表示变亮'
                               '图像默认输出名称为brightness.bmp'))
            returns = change_brightness(filepath, brightness)
            cv2.imwrite('brightness.bmp',returns)
        elif choice == '2':
            filepath = input('请输入图像路径：')
            coefficient = float(input('请输入对比度系数，取值范围为0-2，<1降低对比度，>1提升对比度'
                                '图像默认输出名称为contrast.bmp'))
            returns = change_contrast(filepath, coefficient)
            cv2.imwrite('contrast.bmp',returns)
        elif choice == '3':
            filepath = input('请输入图像路径：')
            coefficient = float(input('请输入饱和度系数，取值范围0-2，<1降低饱和度，>1提升饱和度'
                                '图像默认输出名称为saturation.bmp'))
            returns = change_saturation(filepath, coefficient)
            cv2.imwrite('saturation.bmp',returns)
        elif choice == '4':
            filepath = input('请输入图像路径：')
            sum1 = float(input('请输入色度系数，取值范围为-360-360'
                        '图像默认输出名称为chroma.bmp'))
            returns = change_chroma(filepath, sum1)
            cv2.imwrite('chroma.bmp',returns)
        elif choice == '5':
            filepath = input('请输入图像路径：')
            get_diagram(filepath)
        elif choice == '6':
            filepath = input('请输入图像路径：')
            parameter = float(input("请输入噪点和邻近点平均值的差值限制（建议>30）,"
                              "图像默认输出名称为middle_filter.bmp"))
            returns = middle_filter(filepath, parameter)
            cv2.imwrite('middle_filter.bmp',returns)
        elif choice == '7':
            filepath = input('请输入图像路径：')
            parameter = float(input("请输入噪点和邻近点平均值的差值限制（建议>30）,"
                              "图像默认输出名称为equal_filter.bmp"))
            returns = equal_filter(filepath, parameter)
            cv2.imwrite('equal_filter.bmp', returns)
        elif choice == '8':
            filepath = input('请输入图像路径：')
            print('图像的默认输出名称为roberts.bmp')
            returns = Roberts(filepath)
            cv2.imwrite('roberts.bmp',returns)
        elif choice == '9':
            filepath = input('请输入图像路径：')
            operation = int(input('输入选择的操作：1表示水平，2表示竖直'))
            print('图像的默认输出名称为sobel.bmp')
            returns = Sobel(filepath,operation)
            cv2.imwrite('sobel.bmp',returns)
        elif choice == '10':
            filepath = input('请输入图像路径：')
            parameter = float(input("请输入噪点和邻近点平均值的差值限制（建议>30）,"
                              "图像默认输出名称为fast_middle_filter.bmp"))
            returns = fast_middle_filter(filepath,parameter)
            cv2.imwrite('fast_middle_filter.bmp',returns)
        elif choice == '0':
            exit(0)
        else:
            print("选择错误！")