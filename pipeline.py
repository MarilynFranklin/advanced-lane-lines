import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
from window import Window
from line import Line

class Pipeline():
    def __init__(self, image, mtx, dist, left_line, right_line, save_image=False):
        self.image = image
        self.mtx = mtx
        self.dist = dist
        self.save_image = save_image
        self.right_line = right_line
        self.left_line = left_line

        self.save('original.jpg', self.image)

    def is_binary(self, img):
        return len(img.shape) == 2

    def save(self, fname, img):
        if self.is_binary(img):
            img = cv2.cvtColor(img*255, cv2.COLOR_GRAY2RGB)

        if self.save_image:
            output_folder = "./output_images/"
            filename = "{output_folder!s}/{fname!s}".format(**locals())
            cv2.imwrite(filename, img)

    def undistort(self):
        undistorted = cv2.undistort(self.image, self.mtx, self.dist, None, self.mtx)
        self.save('undistorted.jpg', undistorted)
        return undistorted

    def corners_unwarp(self, img):
        src = np.float32([[(200, 720), (570, 470), (720, 470), (1130, 720)]])
        dst = np.float32([[(350, 720), (350, 0), (980, 0), (980, 720)]])
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, src)
        self.img_size = (self.image.shape[1], self.image.shape[0])

        self.warped = cv2.warpPerspective(img, self.M, self.img_size, flags=cv2.INTER_LINEAR)
        self.save('warped.jpg', self.warped)

        # Return the resulting image and matrix
        return self.warped, self.M, self.M_inv

    def draw(self, ploty, left_fitx, right_fitx):
        warp_zero = np.zeros_like(self.warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.M_inv,
                (self.image.shape[1], self.image.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(self.undistorted, 1, newwarp, 0.3, 0)

        return result

    def abs_sobel_thresh(self, img, orient='x', thresh=(0,255)):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
        else:
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # Create a copy and apply the threshold
        binary_output = np.zeros_like(scaled_sobel)
        # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too'
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return binary_output

    def mag_thresh(self, img, sobel_kernel=3, mag_thresh=(0, 255)):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, sobel_kernel)
        # Calculate the magnitude
        sobel_magn = np.sqrt(sobelx * sobelx + sobely * sobely)
        # Scale to 8-bit (0 - 255) and convert to type = np.uint8
        sobel_scaled = np.uint8(255*sobel_magn/np.max(sobel_magn))
        # Create a binary mask where mag thresholds are met
        binary_output = np.zeros_like(sobel_scaled)
        binary_output[(sobel_scaled > mag_thresh[0]) & (sobel_scaled < mag_thresh[1]) ] = 1
        return binary_output

    def dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi/2)):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the x and y gradients
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)
        # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
        dir1 = np.arctan2(abs_sobely, abs_sobelx)
        # Create a binary mask where direction thresholds are met
        binary_output = np.uint8(np.zeros_like(dir1))
        binary_output[(dir1 >= thresh[0]) & (dir1 <= thresh[1])] = 1
        return binary_output

    def gradient_thresh(self, img, sobel_kernel=3):
        thresh_sobel = (50, 150)
        thresh_mag = (50, 255)
        thresh_dir = (0.75, 1.15)

        gradx = self.abs_sobel_thresh(img, orient='x', thresh=thresh_sobel)
        grady = self.abs_sobel_thresh(img, orient='y', thresh=thresh_sobel)
        mag_binary = self.mag_thresh(img, sobel_kernel=sobel_kernel, mag_thresh=thresh_mag)
        dir_binary = self.dir_threshold(img, sobel_kernel=sobel_kernel, thresh=thresh_dir)

        combined_binary = np.zeros_like(gradx)
        combined_binary[(grady == 1) | (gradx == 1) | ((dir_binary == 1) & (mag_binary == 1))] = 1

        self.save('gradient-thresholds.jpg', combined_binary)

        return combined_binary

    def s_channel_thresh(self, img):
        thresh_s = (170, 255)

        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        img_color = hls[:,:,2]

        s_binary = np.zeros_like(img_color)
        s_binary[(img_color > thresh_s[0]) & (img_color <= thresh_s[1])] = 1

        self.save('color-threshold-s.jpg', s_binary)

        return s_binary

    def r_channel_thresh(self, img):
        thresh_r = (200, 255)
        r_img = img[:,:,0]

        r_binary = np.zeros_like(r_img)
        r_binary[(r_img > thresh_r[0]) & (r_img <= thresh_r[1])] = 1

        self.save('color-threshold-r.jpg', r_binary)
        return r_binary

    def color_thresh(self, img):
        s_binary = self.s_channel_thresh(img)
        r_binary = self.r_channel_thresh(img)

        color_binary = np.zeros_like(r_binary)
        color_binary[(s_binary == 1) | (r_binary == 1)] = 1

        self.save('color-thresholds.jpg', color_binary)

        return color_binary

    def process_image(self, img):
        sobel_kernel = 15

        gradient_binary = self.gradient_thresh(img, sobel_kernel=sobel_kernel)
        color_binary = self.color_thresh(img)

        processed_image = np.zeros_like(gradient_binary)
        processed_image[(gradient_binary == 1) | (color_binary == 1)] = 1

        self.save('preprocessed-image.jpg', processed_image)

        return processed_image

    def curvature_text(self, side, curvature):
        return 'Radius of ' + side + ' curvature: '+str(round(curvature, 2))+'m'

    def center_offset_text(self):
        side_pos = 'left'
        if self.center_offset == 0:
            side_pos = 'right'

        return 'Vehicle is '+str(round(self.center_offset, 3))+'m '+side_pos+' of center'

    def add_text(self, result):
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)
        cv2.putText(result, self.curvature_text('left', self.left_curverad), (50, 50), font, 1, color, 2)
        cv2.putText(result, self.curvature_text('right', self.right_curverad), (50, 100), font, 1, color, 2)
        cv2.putText(result, self.center_offset_text(), (50, 150), font, 1, color, 2)

        return result

    def run(self):
        self.save('original.jpg', self.image)
        self.undistorted = self.undistort()
        self.processed_image = self.process_image(self.undistorted)
        warped, perspective_M, M_inv = self.corners_unwarp(self.processed_image)

        window = Window(warped, self.left_line, self.right_line)
        window.fit_polynomial()
        result = self.draw(window.ploty, window.left_line.bestx, window.right_line.bestx)
        self.left_curverad, self.right_curverad = window.curvature()
        self.center_offset = window.center_offset()

        self.save('fitted.jpg', result)

        result = self.add_text(result)

        self.save('fitted-detail.jpg', result)

        return result

class ProcessImages():
    def __init__(self):
        self.mtx = None
        self.dist = None
        self.right_line = Line()
        self.left_line = Line()

        self.calibrate()

    def calibrate(self):
        dist_pickle = pickle.load(open("calibration_pickle.p", "rb"))
        self.mtx = dist_pickle["mtx"]
        self.dist = dist_pickle["dist"]

    def run_pipeline(self, image, save_image=False):
        pipeline = Pipeline(image, self.mtx, self.dist, self.left_line,
                self.right_line, save_image)
        lane_line_image = pipeline.run()

        return lane_line_image
