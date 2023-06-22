import cv2
import numpy as np
from matplotlib import pyplot as plt


class ocularEllipseFitter:
    def __init__(self):
        self.mode = None
        self.image = None

    def load_image(self, path, gray=True):
        if gray:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(path)
        self.image = img
        return img

    def set_mode(self, mode):
        modes = ['cup', 'disc']
        if mode in modes:
            self.mode = mode
            return mode
        else:
            print('incorrect mode, please choose between "cup" or "disc"!')
            return -1

    def get_threshold(self, image):
        if self.mode == 'disc':
            blur = cv2.GaussianBlur(image, (5, 5), 0)
            _, th3 = cv2.threshold(
                blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            return th3
        elif self.mode == 'cup':
            _, th1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            return th1
        else:
            print('incorrect mode!')
            return -1

    def get_edgemap(self, thresh):
        edge = cv2.Canny(image=thresh, threshold1=100, threshold2=200)
        return edge

    def fit_circle(self, edgemap, input=None):
        if input is not None:
            img = input.copy()
            input = True

        rows = edgemap.shape[0]
        circles = cv2.HoughCircles(
            edgemap, cv2.HOUGH_GRADIENT, 1, rows / 8, param1=100, param2=30,
            minRadius=0, maxRadius=0)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                if input:
                    cv2.circle(img, center, 1, (0, 100, 100), 3)
                else:
                    cv2.circle(edgemap, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                if input:
                    cv2.circle(img, center, radius, (255, 0, 255), 3)
                else:
                    cv2.circle(edgemap, center, radius, (255, 0, 255), 3)
                break
            if input:
                return img
            else:
                return edgemap
        else:
            print('No circle detected!')
            return -1

    def fit_ellipse(self, thresh, input=None):
        if input is not None:
            img = input.copy()
            input = True

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 0:
            for i in range(len(contours)):
                if len(contours[i]) >= 5:
                    ellipse = cv2.fitEllipse(contours[i])
                    if input:
                        cv2.ellipse(img, ellipse, (0, 0, 255), 3)
                    else:
                        cv2.ellipse(thresh, ellipse, (0, 0, 255), 3)
            if input:
                return img
            else:
                return thresh
        else:
            print('No contour found!')
            return -1

    def plot_image(self, image, title=None, gray=True):
        if gray:
            plt.imshow(image, 'gray')
        else:
            plt.imshow(image)
        if title:
            plt.title(title)
        plt.xticks([]), plt.yticks([])
        plt.show()
        return 0

    def fit(self):
        rgb_img = cv2.imread('./imgs/fundus.png')
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        self.plot_image(rgb_img, 'Original Image', gray=False)
        thresh = self.get_threshold(self.image)
        self.plot_image(thresh, 'thresholded image')
        edge = self.get_edgemap(thresh)
        self.plot_image(edge, 'edgemap generated')
        circle = self.fit_circle(edge, input=rgb_img)
        ellipse = self.fit_ellipse(thresh, input=rgb_img)
        self.plot_image(circle, 'fitted circle', gray=False)
        self.plot_image(ellipse, 'fitted ellipse', gray=False)
        return 0


if __name__ == '__main__':
    ocular_ellipse = ocularEllipseFitter()
    ocular_ellipse.load_image('./imgs/fundus.png')
    ocular_ellipse.set_mode('disc')
    ocular_ellipse.fit()
    ocular_ellipse.set_mode('cup')
    ocular_ellipse.fit()
