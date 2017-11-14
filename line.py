import numpy as np

class Line():
    def __init__(self):
        self.smooth_factor = 15
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = []
        #y values for detected line pixels
        self.ally = []
        self.mean_distance = 0

    def find_best_fit(self, current_fit):
        self.current_fit = current_fit

        if (len(self.allx) > 0):
            self.diff_current_fit()

        if (self.mean_distance < 500):
            self.allx.append(current_fit)

        self.bestx = np.average(self.allx[-self.smooth_factor:], axis=0)

    def diff_current_fit(self):
        last_fit = self.allx[-1]
        # mean_distance = 0
        # print(last_fit)
        # for i in len(last_fit):
        #     print(self.current_fit[i][0])
        #     print(last_fit[i][0])
        #     mean_distance += (self.current_fit[i][0] - last_fit[i][0])^2 + (self.current_fit[i][1] - last_fit[i][1])^2
        #     mean_distance /= len(self.current_fit)
        #     mean_distance = sqrt(mean_distance)

        self.mean_distance = ((self.current_fit - last_fit) ** 2).mean(axis=0)
        print(self.mean_distance)
        return self.mean_distance
