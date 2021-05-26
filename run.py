from __future__ import division, print_function, absolute_import
import random
from time import time
import logging
import cv2
import imutils.video
from IOU_tracker import IOUTracker
from MOT_test import *
from deepsort import *
from helper import *



class run_tracker:
    def __init__(self, images, PATH_TXT, PATH_OUTPUT, PATH_IMG, path_file_test, TRACK_TYPE, n,  MOT_TEST=False,
                 Track_without_det=False, GUI=True, wt_path='model_data/model600.pt', iou_threshold=0.8, n_init=3, max_age=10,
                          max_cosine_distance=0.5, sigma_l=0, sigma_h=0.9):
        self.PATH_TXT = PATH_TXT
        self.PATH_OUTPUT = PATH_OUTPUT
        self.PATH_IMG = PATH_IMG
        self.path_file_test = path_file_test
        self.TRACK_TYPE = TRACK_TYPE
        self.MOT_TEST = MOT_TEST
        self.Track_without_det = Track_without_det
        self.images = images
        self.GUI = GUI
        self.n = n
        self.iou_threshold = iou_threshold
        self.n_init = n_init
        self.max_age = max_age
        self.max_cosine_distance = max_cosine_distance
        self.sigma_l = sigma_l
        self.sigma_h = sigma_h
        self.wt_path = wt_path
        self.colors_list = []

    def TRACKING(self):
        for i in range(300):
            self.colors_list.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        print(len( self.colors_list))
        if self.TRACK_TYPE == 'SORT':
            PATH_FILE = read_file(self.PATH_TXT, self.images, self.TRACK_TYPE)
            total_frames = 0
            for seq_dets_fn in glob.glob(PATH_FILE):
                mot_tracker = Sort(max_age=self.max_age, min_hits=self.n_init, iou_threshold=self.iou_threshold, Track_without_det=self.Track_without_det, images=self.images)
                seq_dets = np.loadtxt(seq_dets_fn, delimiter=' ')
            start_time = time()
            fps_imutils = imutils.video.FPS().start()
            names = []
            result = []
            for frame in range(int(seq_dets[:, 0].max())):
                result.append({})
                frame += 1  # detection and frame numbers begin at 1
                dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
                dets[:, 2:4] += dets[:, 0:2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
                total_frames += 1
                trackers = mot_tracker.update(dets)
                for d in trackers:
                    # print('%d,%d,%.2f,%.2f,%.2f,%.2f' % (frame, d[4], d[0], d[1], d[2], d[3]))
                    d = d.astype(np.int32)
                    print(d[4])
                    cv.rectangle(self.images[frame - 1], (d[0], d[1]), (d[2], d[3]), self.colors_list[d[4]], 2)
                    cv.putText(self.images[frame - 1], str(d[4]), (int(round((d[0] + d[2]) / 2)), int(round((d[1] + d[3]) / 2))),
                               1, 2, (0, 0, 0), 2)
                    result[frame - 1]['{}'.format(d[4])] = [d[0], d[1], d[2], d[3]]
                fps_imutils.update()
            fps_imutils.stop()
            print('imutils FPS: {}'.format(fps_imutils.fps()))
            num_frames = len(self.images)
            end = time()
            if self.MOT_TEST:
                self.evaluation(result, names)
            print("finished at " + str(int(num_frames / (end - start_time))) + " fps!")
            if self.GUI:
                show_result(self.images, self.PATH_OUTPUT)

        elif self.TRACK_TYPE == 'DEEP_SORT':
            nn_budget = None
            PATHFILE = read_file(self.PATH_TXT, self.images, self.TRACK_TYPE)
            for seq_dets_fn in glob.glob(PATHFILE):
                seq_dets = np.loadtxt(seq_dets_fn, delimiter=' ')
            deepsort = deepsort_rbc(max_cosine_distance=self.max_cosine_distance, wt_path=self.wt_path)
            metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, nn_budget)
            tracker = Tracker(metric, max_iou_distance=self.iou_threshold, max_age=self.max_age,
                              n_init=self.n_init)
            fps_imutils = imutils.video.FPS().start()
            count_frame = 0
            result = []
            names = []
            for frame in range(int(seq_dets[:, 0].max())):
                result.append({})
                frame += 1
                image = self.images[count_frame]
                boxes = seq_dets[seq_dets[:, 0] == frame, 2:6]
                confidence = seq_dets[seq_dets[:, 0] == frame, -1:]
                classes = seq_dets[seq_dets[:, 0] == frame, 1]
                count_frame += 1
                tracker, detections_class = deepsort.run_deep_sort(image, confidence, boxes)
                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    bbox = track.to_tlbr()  # Get the corrected/predicted bounding box
                    id_num = str(track.track_id)  # Get the ID for the particular track.
                    features = track.features  # Get the feature vector corresponding to the detection.
                    cv2.rectangle(self.images[frame - 1], (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                  self.colors_list[track.track_id], 2)
                    cv2.putText(self.images[frame - 1], "ID:" + id_num, (int(round((bbox[0] + bbox[2]) / 2)),
                                                                    int(round((bbox[1] + bbox[3]) / 2))), 1, 2,
                                (0, 0, 0), 2)
                    bbox = track.to_tlbr()
                    result[frame - 1]['{}'.format(track.track_id)] = [int(bbox[0]), int(bbox[1]), int(bbox[2]),
                                                                      int(bbox[3])]
                    names.append(frame)
                fps_imutils.update()
            fps_imutils.stop()
            print('imutils FPS: {}'.format(fps_imutils.fps()))
            if self.MOT_TEST:
                self.evaluation(result, names)
            if self.GUI:
                show_result(self.images, self.PATH_OUTPUT)
                cv2.destroyAllWindows()

        elif self.TRACK_TYPE == 'IOU':
            dets = read_file(self.PATH_TXT, self.images, self.TRACK_TYPE)
            t_min = 0
            tracks = []
            tracker = IOUTracker(sigma_l=self.sigma_l, sigma_h=self.sigma_h, sigma_iou=self.iou_threshold)
            fps_imutils = imutils.video.FPS().start()
            names = []
            for frame in range(len(self.images)):
                track = get_track(tracker, self.images[frame], dets[frame], self.sigma_l, self.sigma_h,
                                  self.iou_threshold, t_min)
                tracks.append(track)
                names.append(frame)
                fps_imutils.update()
            fps_imutils.stop()
            print('imutils FPS: {}'.format(fps_imutils.fps()))
            if self.MOT_TEST:
                self.evaluation(tracks, names)
            if self.GUI:
                show_result(self.images, self.PATH_OUTPUT)
                cv2.destroyAllWindows()

    def evaluation(self, result, names):
        seqs = get_perfect_bbox(self.path_file_test)  # получение ground truth результатов
        mot_accums = []
        mot_accum = mm.MOTAccumulator(auto_id=True)
        for i in range(len(result)):
            gt_ids, track_ids, distance = get_mot_accum(result[i], seqs[i])
            mot_accum.update(gt_ids, track_ids, distance)
            names.append(i)
            mot_accums.append(mot_accum)

        logging.info("Evaluation:")
        metrics = ('mota', 'motp', 'num_switches', 'idf1', 'precision', 'recall')
        mh = mm.metrics.create()
        summary = mh.compute(mot_accum, metrics=metrics, name=self.n)
        print(summary)
        filename_txt = 'results/metrics{}.txt'.format(self.TRACK_TYPE)
        with open(filename_txt, 'a') as f:
            f.write(str(summary) + '\n')
        # evaluate_mot_accums(mot_accums,
        #                     [str(s) for s in names],
        #                     generate_overall=True)

