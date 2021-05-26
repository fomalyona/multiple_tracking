import configargparse


def parse_args():
    p = configargparse.ArgParser()
    p.add_argument('-c', '--config_file', required=False, is_config_file=True, default='config_file.cfg',
                   help='config file path')
    p.add_argument('-t', '--TRACK_TYPE', required=True, help='choose: IOU, SORT, DEEP_SORT')
    p.add_argument('-n', '--name_test', required=False, help='name for MOT test', default=' ')
    p.add_argument('-pt', '--PATH_TEST', required=True, help='path for ground truth results')
    p.add_argument('-po', '--PATH_OUTPUT', required=True, help='path for output results')
    p.add_argument('-pi', '--PATH_IMG', required=True, help='path for input results')
    p.add_argument('-ptt', '--PATH_TXT', required=True, help='path for detections')
    p.add_argument('-wp', '--wt_path', required=False, default='model_data/model600.pt', help='path for siamese model')
    p.add_argument('-mt', '--MOT_TEST', required=False, help='Displaying averaged metrics', default=False)
    p.add_argument('-twd', '--Track_without_det', required=False,
                   help='Handling missed detections, only for SORT', default=False)
    p.add_argument('-it', '--iou_threshold', required=True, default=0.8, help='a value used in object detection to measure the overlap of a predicted versus actual bounding box for an object')
    p.add_argument('-ni', '--n_init', required=False, default=3, help='Detections needed for initialization')
    p.add_argument('-ma', '--max_age', required=False,  default=10, help='maximum number of steps to save the track')
    p.add_argument('-mcd', '--max_cosine_distance', default=0.5, required=False, help='for DeepSort')
    p.add_argument('-sl', '--sigma_l', required=False, default=0, help='for IOU lower detection threshold')
    p.add_argument('-sh', '--sigma_h', required=False, default=0.9, help='for IOU higher detection threshold')
    options = p.parse_args()
    print(options)
    print("----------")
    return options
