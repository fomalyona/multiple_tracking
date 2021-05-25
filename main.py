from __future__ import division, print_function, absolute_import
import os
from ArgParser import parse_args
from run import *
import warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    options = parse_args()
    PATH_IMG = options.PATH_IMG
    PATH_OUTPUT = options.PATH_OUTPUT
    path_file_test = options.PATH_TEST
    TRACK_TYPE = options.TRACK_TYPE
    PATH_TXT = options.PATH_TXT
    MOT_TEST = options.MOT_TEST
    Track_without_det = options.Track_without_det
    n = str(options.name_file)
    logging.info(f'выбран TRACK_TYPE {TRACK_TYPE}')
    images = read_img(PATH_IMG)
    GUI = True  # показывать результаты
    tracker = run_tracker(images, PATH_TXT, PATH_OUTPUT, PATH_IMG, path_file_test, TRACK_TYPE, n,
                          MOT_TEST=MOT_TEST, Track_without_det=Track_without_det, GUI=GUI, wt_path=options.wt_path,
                          iou_threshold=float(options.iou_threshold), n_init=int(options.n_init), max_age=int(options.max_age),
                          max_cosine_distance=float(options.max_cosine_distance), sigma_l=float(options.sigma_l),
                          sigma_h=float(options.sigma_h))
    tracker.TRACKING()
