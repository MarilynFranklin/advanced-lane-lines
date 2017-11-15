import numpy as np
import cv2
import matplotlib.pyplot as plt

class Window():
    def __init__(self, image, left_line, right_line):
        self.image = image
        # Set the width of the windows +/- margin
        self.margin = 100
        # Set minimum number of pixels found to recenter window
        self.minpix = 50
        # Define conversions in x and y from pixels space to meters
        self.ym_per_pix = 10/720 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/760 # meters per pixel in x dimension

        self.left_line = left_line
        self.right_line = right_line

        nonzero = self.image.nonzero()
        self.nonzeroy = np.array(nonzero[0])
        self.nonzerox = np.array(nonzero[1])

        # Create an output image to draw on and  visualize the result
        self.out_img = np.dstack((self.image, self.image, self.image))*255

    def fit_polynomial(self):
        if np.any(self.left_line.current_fit) and np.any(self.right_line.current_fit):
            left_lane_inds, right_lane_inds = self.start_from_previous_frame(
                    self.left_line.current_fit,
                    self.right_line.current_fit
                )
        else:
            left_lane_inds, right_lane_inds = self.sliding_window()

        # Extract left and right line pixel positions
        self.leftx = self.nonzerox[left_lane_inds]
        self.lefty = self.nonzeroy[left_lane_inds]
        self.rightx = self.nonzerox[right_lane_inds]
        self.righty = self.nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(self.lefty, self.leftx, 2)
        self.right_fit = np.polyfit(self.righty, self.rightx, 2)


        # Generate x and y values for plotting
        self.ploty = np.linspace(0, self.image.shape[0]-1, self.image.shape[0] )
        self.left_fitx = self.left_fit[0]*self.ploty**2 + self.left_fit[1]*self.ploty + self.left_fit[2]
        self.right_fitx = self.right_fit[0]*self.ploty**2 + self.right_fit[1]*self.ploty + self.right_fit[2]

        self.out_img[self.nonzeroy[left_lane_inds], self.nonzerox[left_lane_inds]] = [255, 0, 0]
        self.out_img[self.nonzeroy[right_lane_inds], self.nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.figure()
        plt.imshow(self.out_img)
        plt.plot(self.left_fitx, self.ploty, color='yellow')
        plt.plot(self.right_fitx, self.ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.savefig("./output_images/plot.jpg")

        # Average over last fit
        self.left_line.find_best_fit(self.left_fitx, self.left_fit)
        self.right_line.find_best_fit(self.right_fitx, self.right_fit)

    def start_from_previous_frame(self, left_fit, right_fit):
        self.left_fit = left_fit
        self.right_fit = right_fit
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "self.image")
        # It's now much easier to find line pixels!
        left_lane_inds = ((self.nonzerox > (self.left_fit[0]*(self.nonzeroy**2) + self.left_fit[1]*self.nonzeroy +
            self.left_fit[2] - self.margin)) & (self.nonzerox < (self.left_fit[0]*(self.nonzeroy**2) +
            self.left_fit[1]*self.nonzeroy + self.left_fit[2] + self.margin)))

        right_lane_inds = ((self.nonzerox > (self.right_fit[0]*(self.nonzeroy**2) + self.right_fit[1]*self.nonzeroy +
            self.right_fit[2] - self.margin)) & (self.nonzerox < (self.right_fit[0]*(self.nonzeroy**2) +
            self.right_fit[1]*self.nonzeroy + self.right_fit[2] + self.margin)))

        return left_lane_inds, right_lane_inds

    def sliding_window(self):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(self.image[self.image.shape[0]//2:,:], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(self.image.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = self.image.shape[0] - (window+1)*window_height
            win_y_high = self.image.shape[0] - window*window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin
            # Draw the windows on the visualization image
            cv2.rectangle(self.out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high), (0,255,0), 2)
            cv2.rectangle(self.out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high), (0,255,0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) &
                (self.nonzerox >= win_xleft_low) &  (self.nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) &
                (self.nonzerox >= win_xright_low) &  (self.nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(self.nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = np.int(np.mean(self.nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        return left_lane_inds, right_lane_inds

    def center_offset(self):
        camera_center = (self.left_fitx[-1] + self.right_fitx[-1])/2
        return (camera_center-self.image.shape[1]/2)*self.xm_per_pix

    def curvature(self):
        y_eval = np.max(self.ploty)

        # Fit new polynomials to x,y in world space
        self.left_fit_cr = np.polyfit(self.lefty*self.ym_per_pix, self.leftx*self.xm_per_pix, 2)
        self.right_fit_cr = np.polyfit(self.righty*self.ym_per_pix, self.rightx*self.xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*self.left_fit_cr[0]*y_eval*self.ym_per_pix + self.left_fit_cr[1])**2)**1.5) / np.absolute(2*self.left_fit_cr[0])
        right_curverad = ((1 + (2*self.right_fit_cr[0]*y_eval*self.ym_per_pix + self.right_fit_cr[1])**2)**1.5) / np.absolute(2*self.right_fit_cr[0])

        # Now our radius of curvature is in meters
        return left_curverad, right_curverad
