import numpy as np

class Line():
    def __init__(self):
        self.smooth_factor = 15
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        self.current_fitx = [np.array([False])]
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = []
        self.mean_distance = 0

    def find_best_fit(self, current_fitx, current_fit):
        self.current_fitx = current_fitx
        self.current_fit = [np.array([False])]

        if (len(self.allx) > 0):
            self.diff_current_fitx()

        if (self.mean_distance < 500):
            self.allx.append(current_fitx)
            self.current_fit = current_fit

        self.bestx = np.average(self.allx[-self.smooth_factor:], axis=0)

    def diff_current_fitx(self):
        last_fit = self.allx[-1]
        self.mean_distance = ((self.current_fitx - last_fit) ** 2).mean(axis=0)
        print(self.mean_distance)
        return self.mean_distance
