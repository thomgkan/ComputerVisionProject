import numpy as np
import cv2

def panorama(method):

    if method =='SIFT':
        meth = cv2.xfeatures2d.SIFT_create()
    if method == 'SURF':
        meth = cv2.xfeatures2d.SURF_create()

    for i in range(3):
        if i == 0:
            #im1 = cv2.imread('im_1.jpg')
            #im2 = cv2.imread('im_0.jpg')
            im1 = cv2.imread('hotel-01.png')
            im2 = cv2.imread('hotel-00.png')

            scale_percent = 50
            width = int(im1.shape[1] * scale_percent / 100)
            height = int(im1.shape[0] * scale_percent / 100)
            dim = (width, height)
            im1 = cv2.resize(im1, dim, interpolation=cv2.INTER_AREA)
            im2 = cv2.resize(im2, dim, interpolation=cv2.INTER_AREA)
        if i == 1:
            #im1 = cv2.imread('im_2.jpg')
            im1 = cv2.imread('hotel-02.png')
            im1 = cv2.resize(im1, dim, interpolation=cv2.INTER_AREA)
        if i == 2:
            #im1 = cv2.imread('im_3.jpg')
            im1 = cv2.imread('hotel-03.png')
            im1 = cv2.resize(im1, dim, interpolation=cv2.INTER_AREA)

        kp1 = meth.detect(im1)
        des1 = meth.compute(im1, kp1)
        kp2 = meth.detect(im2)
        des2 = meth.compute(im2, kp2)

        d1 = des1[1]
        d2 = des2[1]

        n1 = d1.shape[0]
        n2 = d2.shape[0]

        matches = []
        for i in range(n1):

            fv = d1[i, :]
            diff = d2 - fv
            diff = np.abs(diff)
            distances = np.sum(diff, axis=1)
            i1 = np.argmin(distances)
            mindist = distances[i1]

            fv = d2[i1, :]
            diff2 = d1 - fv
            diff2 = np.abs(diff2)
            distances2 = np.sum(diff2, axis=1)
            i2 = np.argmin(distances2)

            if i == i2:
                matches.append(cv2.DMatch(i, i1, mindist))

        dimg = cv2.drawMatches(im1, des1[0], im2, des2[0], matches, None)

        # cv2.namedWindow('1')
        # cv2.imshow('1', dimg)
        # cv2.waitKey(0)

        img_pt1 = []
        img_pt2 = []
        for x in matches:
            img_pt1.append(kp1[x.queryIdx].pt)
            img_pt2.append(kp2[x.trainIdx].pt)
        img_pt1 = np.array(img_pt1)
        img_pt2 = np.array(img_pt2)


        M, mask = cv2.findHomography(img_pt2, img_pt1, cv2.RANSAC)
        img4 = cv2.warpPerspective(im2, M, (im2.shape[1] + 5000, im2.shape[0] + 5000))

        img4[0: im1.shape[0], 0: im1.shape[1]] = im1

        # cv2.namedWindow('2')
        # cv2.imshow('2', img4)
        # cv2.waitKey(0)

        gray = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)
        crop = img4[y:y + h, x:x + w]

        blck1 = 0

        for i in range(crop.shape[0]):
            if crop[i, 0, 0] == 0 and crop[i, 0, 1] == 0 and crop[i, 0, 2] == 0:
                blck1 = blck1 + 1

        cropnew = crop[0:y + h - blck1, 0:x + w ]

        im2 = cropnew
        # cv2.namedWindow('3')
        # cv2.imshow('3', im2)
        # cv2.waitKey(0)


    return im2

#pano=panorama('SIFT')
cv2.imwrite('panoramasift.png',panorama('SIFT'))
#pano=panorama('SURF')
cv2.imwrite('panoramasurf.png',panorama('SURF'))


