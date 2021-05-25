import glob
import os

import motmetrics as mm
import numpy as np
import xmltodict


def get_perfect_bbox(path_file_test):
    if os.path.splitext(path_file_test)[1] == '.txt':
        detections = dict()
        temp = []
        with open(path_file_test) as f_inp:
            lines = f_inp.readlines()
            for i in range(len(lines) - 1):
                x1 = list(map(float, lines[i].split(',')))
                temp.append(x1)
            for j in range(len(temp)):
                detections.setdefault('number_frame', []).extend([int(temp[j][0])])
                detections.setdefault('id', []).extend([int(temp[j][1])])
                detections.setdefault('x', []).extend([int(temp[j][2])])
                detections.setdefault('y', []).extend([int(temp[j][3])])
                detections.setdefault('w', []).extend([int(temp[j][4])])
                detections.setdefault('h', []).extend([int(temp[j][5])])
                detections.setdefault('score', []).extend([float(temp[j][6])])
            seqs = []
            count_frame = 0
            seqs.append({})
            for i in range(1, len(detections['number_frame'])):
                if detections['number_frame'][i] == detections['number_frame'][i - 1]:
                    seqs[count_frame][int('{}'.format(detections['id'][i - 1]))] = [detections['x'][i - 1],
                                                                               detections['y'][i - 1],
                                                                               detections['x'][i - 1] + detections['w'][
                                                                                   i - 1],
                                                                               detections['y'][i - 1] + detections['h'][
                                                                                   i - 1]]
                else:
                    seqs.append({})
                    seqs[count_frame][int('{}'.format(detections['id'][i - 1]))] = [detections['x'][i - 1],
                                                                               detections['y'][i - 1],
                                                                               detections['x'][i - 1] + detections['w'][
                                                                                   i - 1],
                                                                               detections['y'][i - 1] + detections['h'][
                                                                                   i - 1]]
                    count_frame += 1
    elif os.path.splitext(path_file_test)[1] == '.xml':
        filenames = [ann for ann in glob.glob(path_file_test)]
        print(filenames)
        seqs = []
        count_frame = 0
        for ann in filenames:
            seqs.append({})
            with open(ann, errors="ignore") as fd:
                doc = xmltodict.parse(fd.read())
                data = dict(doc['annotation'])
                t = data['object']
                # print(t)
                if type(t) == list:
                    for i in range(len(t)):
                        k = dict(data['object'][i])
                        id = k['name'][-1]
                        bbox = dict(k['bndbox'])
                        seqs[count_frame][int('{}'.format(id))] = [int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']),
                                                              int(bbox['ymax'])]

                else:
                    k = dict(t)
                    id = k['name'][-1]
                    bbox = dict(k['bndbox'])
                    seqs[count_frame][int('{}'.format(id))] = [int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']),
                                                          int(bbox['ymax'])]
            count_frame += 1
    return seqs


def get_mot_accum(results, seq):
    gt_ids = []
    if seq:
        gt_boxes = []
        for id, bbox in seq.items():
            gt_ids.append(id)
            gt_boxes.append(bbox)
        gt_boxes = np.stack(gt_boxes, axis=0)
        # x1, y1, x2, y2 --> x1, y1, width, height
        gt_boxes = np.stack((gt_boxes[:, 0],
                             gt_boxes[:, 1],
                             gt_boxes[:, 2] - gt_boxes[:, 0],
                             gt_boxes[:, 3] - gt_boxes[:, 1]),
                            axis=1)

    else:
        gt_boxes = np.array([])

    track_ids = []
    track_boxes = []
    for id, boxes in results.items():
        track_ids.append(id)
        track_boxes.append(boxes)

    if track_ids:
        track_boxes = np.stack(track_boxes, axis=0)
        # x1, y1, x2, y2 --> x1, y1, width, height
        track_boxes = np.stack((track_boxes[:, 0],
                                track_boxes[:, 1],
                                track_boxes[:, 2] - track_boxes[:, 0],
                                track_boxes[:, 3] - track_boxes[:, 1]),
                               axis=1)
    else:
        track_boxes = np.array([])

    distance = mm.distances.iou_matrix(gt_boxes, track_boxes, max_iou=0.5)
    return gt_ids, track_ids, distance
    # distance = mm.distances.norm2squared_matrix(gt_boxes, track_boxes, max_d2=5.)


def evaluate_mot_accums(accums, names, generate_overall=False):
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accums,
        metrics=mm.metrics.motchallenge_metrics,
        names=names,
        generate_overall=generate_overall,)

    str_summary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names,)
    print(str_summary)