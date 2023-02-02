import numpy as np

# Average Equal loudness Response Curve
# https://replaygain.hydrogenaud.io/equal_loudness.html
average_equal_loudness_curve = np.array(
    [
        [20, 113],
        [30, 103],
        [40, 97],
        [50, 93],
        [60, 91],
        [70, 89],
        [80, 87],
        [90, 86],
        [100, 85],
        [200, 78],
        [300, 76],
        [400, 76],
        [500, 76],
        [600, 76],
        [700, 77],
        [800, 78],
        [900, 79.5],
        [1000, 80],
        [1500, 79],
        [2000, 77],
        [2500, 74],
        [3000, 71.5],
        [3700, 70],
        [4000, 70.5],
        [5000, 74],
        [6000, 79],
        [7000, 84],
        [8000, 86],
        [9000, 86],
        [10000, 85],
        [12000, 95],
        [15000, 110],
        [20000, 125],
        [24000, 140],
    ]
)

inverse_equal_loudness_curve = average_equal_loudness_curve.copy()
inverse_equal_loudness_curve[:, 1] = 70 - inverse_equal_loudness_curve[:, 1]


# Parameters for denoising and extracting pitch contours
# http://www.justinsalamon.com/uploads/4/3/9/4/4394963/salamongomezmelodytaslp2012.pdf
threshold_factor = 0.9  # Ratio of max saliency to threshold
threshold_deviation = 0.9  # Number of standard deviations below mean to threshold
voicing_threshold_deviation = 0.2  # Threshold for voicing detection
