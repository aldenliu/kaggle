#coding: utf-8
import cv2
import pdb
def getSift():
    ''''' 
    get and extract sift feature
    '''
    img_path1 = './00022e1a.jpg'
    #读取图像
    img = cv2.imread(img_path1)
    #转换为灰度图
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #创建sift的类
    sift = cv2.SIFT()
    #在图像中找到关键点 也可以一步计算#kp, des = sift.detectAndCompute
    kp = sift.detect(gray,None)
    print type(kp),type(kp[0])
    #Keypoint数据类型分析 http://www.cnblogs.com/cj695/p/4041399.html
    print kp[0].pt
    #计算每个点的sift
    des = sift.compute(gray,kp)
    print type(kp),type(des)
    #des[0]为关键点的list，des[1]为特征向量的矩阵
    print type(des[0]), type(des[1])
    print des[0],des[1]
    #可以看出共有885个sift特征，每个特征为128维
    print des[1].shape
    #在灰度图中画出这些点
    img=cv2.drawKeypoints(gray,kp)
    ecv2.imwrite('sift_keypoints.jpg',img)
    plt.imshow(img),plt.show()

def show_pic(img, name):
    cv2.namedWindow(name)
    cv2.imshow(name, img)
    cv2.waitKey (0)  


def get_sift(img):
    print 'begin to dectect sift'
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kps = sift.detect(gray, None)
    cv2.drawKeypoints(img, kps, img, (255, 0, 0))
    show_pic(img, 'test')

def get_akaze(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    akaze = cv2.AKAZE_create()
    (kps, des) = akaze.detectAndCompute(gray, None)
    #filter_kps = filter(lambda x: x.octave > 0, kps)
    return (kps, des)

def test():
    img_path1 = './00022e1a.jpg'
    img_path2 = './b81d5270.jpg'
    img_path3 = './dba58da5.jpg'
    img_path4 = './fff04277.jpg'
    #读取图像
    img2 = cv2.imread(img_path2)
    img3 = cv2.imread(img_path3)
    img4 = cv2.imread(img_path4)
    #get_sift(img)
    img2 = get_akaze(img2)
    img3 = get_akaze(img3)
    img4 = get_akaze(img4)
    cv2.imwrite('./akaze2.jpg', img2)
    cv2.imwrite('./akaze3.jpg', img3)
    cv2.imwrite('./akaze4.jpg', img4)

def main():
    test()

if __name__ == '__main__':
    main()
