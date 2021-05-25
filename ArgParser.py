import configargparse


def parse_args():
    p = configargparse.ArgParser(default_config_files=['test.cfg'])
    p.add_argument('-c', '--config_file', required=True, is_config_file=True, default='/home/alyona/PycharmProjects/kalman_filt/test.cfg',
                   help='config file path')

    p.add_argument('-ta', '--TRACK_TYPE', required=True, help='tracker_address')
    p.add_argument('-tc', '--name_file', required=True, help='')
    p.add_argument('-ttc', '--PATH_TEST', required=True, help='')
    p.add_argument('-sw', '--PATH_OUTPUT', required=True, help='')
    p.add_argument('-sh', '--PATH_IMG', required=True, help='')
    p.add_argument('-r', '--PATH_TXT', required=True, help='')
    p.add_argument('-t', '--MOT_TEST', required=True, help='')
    p.add_argument('-g', '--Track_without_det', required=True, help='')

    p.add_argument('-dt', '--dist_thresh', required=True, help='')
    p.add_argument('-mfs', '--max_frames_to_skip', required=True, help='')
    options = p.parse_args()
    print(options)
    print("----------")
    return options