import json
import os
import glob
from scipy.interpolate import interp1d

import cv2 as cv
import numpy as np
from SORT import *
import motmetrics as mm

colors = [
    'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque',
    'black', 'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue',
    'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan',
    'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki',
    'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon',
    'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise',
    'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick',
    'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod',
    'gray', 'green', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo',
    'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue',
    'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey',
    'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey',
    'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon',
    'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen',
    'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue',
    'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab',
    'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
    'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue',
    'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon',
    'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue',
    'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle',
    'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen'
]



def read_img(path_file):
    images = []
    if os.path.splitext(path_file)[1] == '.avi':
        # Create the VideoCapture object
        cap = cv.VideoCapture(path_file)
        if not cap.isOpened():
            print("Video device or file couldn't be opened")
            exit()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("error read_img")
                break
            # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            images.append(frame)
        cap.release()
        cv.destroyAllWindows()
        return images
    elif os.path.splitext(path_file)[1] == '.png':
        print('открыли директорию')
        filenames = [img for img in glob.glob(path_file)]
        print(filenames)
        for img in filenames:
            image = cv.imread(img, cv.IMREAD_COLOR)
            images.append(image)
        return images

    else:
        list_of_files = len(os.listdir(path_file))
        for filename in range(1, list_of_files + 1):
            with open(os.path.join(path_file, '0{}.png'.format(filename))) as img:
                image = cv.imread(os.path.join(path_file, '0{}.png'.format(filename)), cv.IMREAD_COLOR)
                images.append(image)
        return images
    # else:
    #     list_of_files = len(os.listdir(path_file))
    #     for filename in range(list_of_files):
    #         with open(os.path.join(path_file, '{}.jpg'.format(filename))) as img:
    #             image = cv.imread(os.path.join(path_file, '{}.jpg'.format(filename)), cv.IMREAD_COLOR)
    #             images.append(image)
    #     return images



def read_file(path_file, images, TRACK_TYPE):
    if os.path.splitext(path_file)[1] == '.txt':
        if TRACK_TYPE == 'SORT' or TRACK_TYPE == 'DEEP_SORT':
            return path_file
        else:
            detections = dict()
            temp = []
            with open(path_file) as f_inp:
                lines = f_inp.readlines()
                for i in range(len(lines) - 1):
                    x1 = list(map(float, lines[i].split()))
                    temp.append(x1)
            for j in range(len(temp)):
                detections.setdefault('number_frame', []).extend([int(temp[j][0]) - 1 ] )
                detections.setdefault('x', []).extend([int(temp[j][2])])
                detections.setdefault('y', []).extend([int(temp[j][3])])
                detections.setdefault('w', []).extend([int(temp[j][4])])
                detections.setdefault('h', []).extend([int(temp[j][5])])
                detections.setdefault('score', []).extend([float(temp[j][6])])
                cv.rectangle(images[int(temp[j][0]) - 1], (int(temp[j][2]), int(temp[j][3])),
                             (int(temp[j][2]) + int(temp[j][4]),
                              int(temp[j][3]) + int(temp[j][5])),
                             (255, 0, 0), 2)
            dets = []
            count_frame = 0
            dets.append([])
            for i in range(1, len(detections['number_frame'])):
                if detections['number_frame'][i] == detections['number_frame'][i - 1]:
                    dets[count_frame].append({'number_frame': count_frame, 'bbox': (
                        detections['x'][i - 1], detections['y'][i - 1], detections['x'][i - 1] + detections['w'][i - 1],
                        detections['y'][i - 1] + detections['h'][i - 1]), 'score': detections['score'][i - 1], 'class': -1})
                else:

                    dets.append([])
                    dets[count_frame].append({'number_frame': count_frame, 'bbox': (
                        detections['x'][i - 1], detections['y'][i - 1], detections['x'][i - 1] + detections['w'][i - 1],
                        detections['y'][i - 1] + detections['h'][i - 1]), 'score': detections['score'][i - 1], 'class': -1})
                    count_frame += 1
            return dets
    elif os.path.splitext(path_file)[1] == '.json':
        data = []
        with open(path_file) as json_data:
            data.append(json.load(json_data))
        dets = []
        count = 0
        for j in range(len(images)):
            dets.append([])
        for tmp in range(len(data[0]['annotations'])):
            count_frame = data[0]['annotations'][count]['image_id'] - 1
            bbox = data[0]['annotations'][tmp]['bbox']
            classtitle = data[0]['annotations'][tmp]['category_id']
            dets[len(images) - count_frame - 1].append({'number_frame': len(images) - count_frame - 1,
                                                        'bbox': (bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]),
                                                        'score': float(1.0), 'class': classtitle})
            if TRACK_TYPE == 'SORT' or TRACK_TYPE == 'DEEP_SORT':
                pass
            else:
                cv.rectangle(images[len(images) - count_frame - 1], (int(bbox[0]), int(bbox[1])),
                         (int(bbox[2]) + int(bbox[0]), int(bbox[3]) + int(bbox[1])), (255, 0, 0), 2)
                pass
            count += 1
        if TRACK_TYPE == 'SORT' or TRACK_TYPE == 'DEEP_SORT':
            file_name = str(path_file) + ".txt"
            with open(file_name, "w") as file:
                for i in range(len(dets)):
                    for j in range(len(dets[i])):
                        t = [dets[i][j]['number_frame'], dets[i][j]['class'], int(dets[i][j]['bbox'][0]),
                             int(dets[i][j]['bbox'][1]),
                             int(dets[i][j]['bbox'][2]) - int(dets[i][j]['bbox'][0]), int(dets[i][j]['bbox'][3]) - int(dets[i][j]['bbox'][1]), dets[i][j]['score']]
                        str1 = ' '.join(str(e) for e in t)
                        file.write(str1 + '\n')
            return file_name
        elif TRACK_TYPE == 'IOU':
            return dets
    else:
        data = []
        list_of_files = len(os.listdir(path_file))
        for filename in range(list_of_files):
            with open(path_file + '{}.jpg.json'.format(filename)) as json_data:
                data.append(json.load(json_data))
        detections = []
        dets = []
        count = 0

        for tmp in data:
            dets.append([])
            if bool(tmp['objects']):
                bbox = tmp['objects'][0]['points']['exterior']
                classtitle = tmp['objects'][0]['classTitle']
                dets[count].append(
                    {'number_frame': count, 'bbox': (bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]), 'score': 1,
                     'class': classtitle})
                cv.rectangle(images[count], (bbox[0][0], bbox[0][1]), (bbox[1][0], bbox[1][1]), (255, 0, 0), 2)
                count += 1
            else:

                dets[count].append({'number_frame': count, 'bbox': -1, 'score': -1, 'class': -1})
                count += 1

        return dets


def get_track(tracker, images, dets, sigma_l, sigma_h, sigma_iou, t_min):
    track = {}
    track_active = tracker.track(dets, sigma_l, sigma_h, sigma_iou, t_min)
    list_active_track = list()
    for t in track_active.keys():
        list_active_track.append([t, track_active[t]])
    for j in range(len(list_active_track)):
        d = np.array(list_active_track[j][1]['bbox'])
        # print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            # list_active_track[j][1]['number_frame'], list_active_track[j][0], d[0], d[1], d[2], d[3]))
        track[int('{}'.format(int((list_active_track[j][0]))))] = \
            [d[0], d[1], d[2], d[3]]
        d = d.astype(np.int32)
        cv.rectangle(images, (d[0], d[1]), (d[2], d[3]), (0, 255, 255), 2)
        cv.putText(images, str(list_active_track[j][0]), (round((d[0] + d[2]) / 2), round((d[1] + d[3]) / 2)), 1,
                   1, (0, 0, 255), 2)
    return track

def show_result(images, PATH_OUTPUT):
    h, w, _ = images[0].shape
    out = cv.VideoWriter(PATH_OUTPUT, cv.VideoWriter_fourcc(*'MJPG'), 30.0, (w, h))
    for i in range(len(images)):
        cv.imshow('', images[i])
        cv.waitKey(50)
        out.write(images[i])
    cv.destroyAllWindows()
    out.release()




