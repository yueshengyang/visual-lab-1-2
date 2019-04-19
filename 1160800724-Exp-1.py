import struct
import numpy as np
import math
import sys
'''
用于获得文件的头部信息
'''
def openbmp(path):
    f = open(path, "rb")
    f.read(10)
    bias = f.read(4)
    bias = struct.unpack("i", bias)[0]
    f.read(4)
    width = f.read(4)
    width = struct.unpack("i", width)[0]
    height = f.read(4)
    height = struct.unpack("i", height)[0]
    f.read(2)
    bpp = f.read(2)
    bpp = struct.unpack("h", bpp)[0]
    n = bias - 30
    f.read(n)
    size = bias
    return width, abs(height), size

def readfile(filepath,size,width,height):
    data = []
    with open(filepath,'rb') as f:
        data.append(f.read(size))
        duodu = 0
        if width *3 %4 != 0:
            duodu = 4-width*3%4
        for i in range(height):
            for j in range(width):
                d = f.read(1)
                d = struct.unpack("B", d)[0]
                data.append(d)
                d = f.read(1)
                d = struct.unpack("B", d)[0]
                data.append(d)
                d = f.read(1)
                d = struct.unpack("B", d)[0]
                data.append(d)
            for i in range(duodu):
                d = f.read(1)
                d = struct.unpack("B", d)[0]
                data.append(d)
    return data

def RGB_to_YIQ(data,width,height,size,):
    thedata = []
    for i in range(len(data)):
       thedata.append(data[i])
    if (width*3)%4 ==0:
        duodu = 0
    else:
        duodu =  4-(width*3)%4
    for i in range(height):
        for j in range(width):
            pianyi = 1+i*(width*3+duodu)+j*3
            R = thedata[pianyi+2]
            G = thedata[pianyi+1]
            B = thedata[pianyi]
            Y = int(0.299 * R + 0.587 * G + 0.114 * B)
            I = int(0.596 * R - 0.274 * G - 0.322 * B)
            Q = int(0.211 * R - 0.523 * G + 0.312 * B)
            thedata[pianyi] = int(Y)
            thedata[pianyi+1] = int(I)+128
            thedata[pianyi+2] = int(Q)+128
    return thedata

def RGB_to_HSI(data,width,height,size):
    thedata = []
    for i in range(len(data)):
        thedata.append(data[i])
    if (width * 3) % 4 == 0:
        duodu = 0
    else:
        duodu = 4 - (width * 3) % 4
    for i in range(height):
        for j in range(width):
            pianyi = 1 + i * (width * 3 + duodu) + j * 3
            R = thedata[pianyi + 2]
            G = thedata[pianyi + 1]
            B = thedata[pianyi]
            if R + G + B == 0:
                h = math.pi / 2
                s = 1
                p = 0
            else:
                r = R / (R + G + B)
                g = G / (R + G + B)
                b = B / (R + G + B)
                if b <= g:
                    if ((r - g) ** 2) + (r - b) * (g - b)<0.00001:
                        h = 0
                    else :
                        if 0.5 * (r - g + r - b) / math.sqrt(((r - g) ** 2)
                                                             + (r - b) * (g - b))>1:
                            h = 0
                        else :
                            h = math.acos(0.5 * (r - g + r - b) / math.sqrt(
                                ((r - g) ** 2) + (r - b) * (g - b)))
                else:
                    if ((r - g) ** 2) + (r - b) * (g - b) < 0.00001:
                        h = 0
                    else :
                        h = 2 * math.pi - math.acos(0.5 * (r - g + r - b) /
                                                    math.sqrt(((r - g) ** 2) + (r - b) * (g - b)))
                min = r
                if min > g:
                    min = g
                if min > b:
                    min = b
                s = 1 - 3 * min
                p = (R + G + B) / (3 * 255)
            H = h * 180 / math.pi
            S = s * 100
            I = p * 255
            thedata[pianyi] = int(H)
            thedata[pianyi+1] = int(S)
            thedata[pianyi+2] = int(I)
    return thedata

def RGB_to_YCbCr(data,width,height,size):
    thedata = []
    for i in range(len(data)):
        thedata.append(data[i])
    if (width*3)%4 ==0:
        duodu = 0
    else:
        duodu =  4-(width*3)%4
    for i in range(height):
        for j in range(width):
            pianyi = 1+i*(width*3+duodu)+j*3
            R = thedata[pianyi+2]
            G = thedata[pianyi+1]
            B = thedata[pianyi]
            Y = int(0.299 * R + 0.587 * G + 0.114 * B)
            Cb = int(-0.169 * R - 0.331 * G + 0.500 * B)
            Cr = int(0.500 * R - 0.419 * G -0.081 * B)
            thedata[pianyi] = int(Y)
            thedata[pianyi+1] = int(Cb)+128
            thedata[pianyi+2] = int(Cr)+128
    return thedata

def RGB_to_XYZ(data,width,height,size):
    thedata = []
    for i in range(len(data)):
        thedata.append(data[i])
    if (width * 3) % 4 == 0:
        duodu = 0
    else:
        duodu = 4 - (width * 3) % 4
    for i in range(height):
        for j in range(width):
            pianyi = 1 + i * (width * 3 + duodu) + j * 3
            R = thedata[pianyi + 2]
            G = thedata[pianyi + 1]
            B = thedata[pianyi]
            X = 0.412453 * R + 0.357580 * G + 0.180423 * B
            Y = 0.212671 * R +0.715160 * G +0.072169 * B
            Z = 0.019334 * R +0.119193 * G + 0.950227 * B
            thedata[pianyi] = int(X)
            thedata[pianyi + 1] = int(Y)
            thedata[pianyi + 2] = int(Z)
    return thedata

def newname(path, mode):
    str = path.split("/")[-1]
    str = str.split(".")[0]
    str = str + "-1160800724-" + mode + ".bmp"
    return str

def regu(x):
    if x<0:
        return 0
    elif x > 255:
        return 255
    else:
        return int(x)

if __name__ == '__main__':
    width, height, size = openbmp(sys.argv[1])
    data = readfile(sys.argv[1],size,width,height)
    thedata = RGB_to_YIQ(data,width,height,size)
    with open(newname(sys.argv[1],'YIQ'),'wb') as f:
        f.write(thedata[0])
        for i in range(1,len(thedata)):
            d = struct.pack("B", regu(thedata[i]))
            f.write(d)
    thedata1 = RGB_to_HSI(data,width,height,size)
    with open(newname(sys.argv[1],'HSI'),'wb') as f1:
        f1.write(thedata1[0])
        for i in range(1,len(thedata1)):
            d = struct.pack('B',regu(thedata1[i]))
            f1.write(d)
    thedata2 = RGB_to_YCbCr(data,width,height,size)
    with open(newname(sys.argv[1],'YCbCr'),'wb') as f3:
        f3.write(thedata2[0])
        for i in range(1, len(thedata2)):
            d = struct.pack('B', regu(thedata2[i]))
            f3.write(d)
    thedata3 = RGB_to_XYZ(data,width,height,size)
    with open(newname(sys.argv[1],'XYZ'),'wb') as f4:
        f4.write(thedata3[0])
        for i in range(1, len(thedata3)):
            d = struct.pack('B', regu(thedata3[i]))
            f4.write(d)