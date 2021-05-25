def iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.
    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """

    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union


class IOUTracker():
    def __init__(self, sigma_l=0, sigma_h=0.9, sigma_iou=0.8, t_max=5, logging=False):
        self.sigma_l = sigma_l
        self.sigma_h = sigma_h
        self.sigma_iou = sigma_iou
        self.t_max = t_max
        self.frame = 0
        self.id_count = 0
        self.tracks_active = {}
        self.logging = logging
        self.log = {0: []}
        self.tracks_finished = []

    # Clear the old tracks
    def clean_old_tracks(self):
        target_frame = self.frame - self.t_max
        if (target_frame in self.tracks_active):
            if self.logging:
                self.log[self.frame] = ["[Log]: Tracks Deleted:{}".format(self.tracks_active[target_frame].keys())]
        # del(self.tracks_active[target_frame])

    # Retrieve tracks in an correct matching order
    def retrieve_tracks(self):
        tracks = []
        selected_tracks = {}
        frames = range(self.frame, self.frame - self.t_max, -1)
        for frame in frames:
            if frame in self.tracks_active:
                for id_, trackt in self.tracks_active[frame].items():
                    if id_ not in selected_tracks:
                        # tracks += (id_,track)
                        tracks += self.tracks_active[frame].items()
                        selected_tracks[id_] = trackt
                # tracks += self.tracks_active[frame].items()
        return tracks

    def track(self, detections, sigma_l, sigma_h, sigma_iou, t_min):
        self.frame += 1
        self.tracks_active[self.frame] = {}
        # Clear the tracks in old frame
        self.clean_old_tracks()
        dets = [det for det in detections if det['score'] >= self.sigma_l]

        for id_, trackr in self.retrieve_tracks():
            if len(dets) > 0:
                # get det with highest iou
                best_match = max(dets, key=lambda x: iou(trackr['bbox'], x['bbox']))
                if iou(trackr['bbox'], best_match['bbox']) >= self.sigma_iou:
                    self.tracks_active[self.frame][id_] = best_match
                    # remove from best matching detection from detections
                    del dets[dets.index(best_match)]

        # Create new tracks
        for det in dets:
            self.id_count += 1
            self.tracks_active[self.frame][self.id_count] = det

        # Return the current tracks
        return self.tracks_active[self.frame]

    def track_iou(self, detections, sigma_l, sigma_h, sigma_iou, t_min):

        tracks_active = []

        for frame_num, detections_frame in enumerate(detections, start=1):
            # apply low threshold to detections
            dets = [det for det in detections_frame if det['score'] >= sigma_l]

            updated_tracks = []
            for track in tracks_active:
                if len(dets) > 0:
                    # get det with highest iou
                    best_match = max(dets, key=lambda x: iou(track['bboxes'][-1], x['bbox']))
                    if iou(track['bboxes'][-1], best_match['bbox']) >= sigma_iou:
                        track['bboxes'].append(best_match['bbox'])
                        track['max_score'] = max(track['max_score'], best_match['score'])

                        updated_tracks.append(track)

                        # remove from best matching detection from detections
                        del dets[dets.index(best_match)]

                # if track was not updated
                if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
                    # finish track when the conditions are met
                    if track['max_score'] >= sigma_h and len(track['bboxes']) >= t_min:
                        self.tracks_finished.append(track)

            # create new tracks
            new_tracks = [
                {'bboxes': [det['bbox']], 'max_score': det['score'], 'start_frame': frame_num, 'classes': det['class']}
                for det in dets]
            tracks_active = updated_tracks + new_tracks

        # finish all remaining active tracks
        self.tracks_finished += [track for track in tracks_active
                                 if track['max_score'] >= sigma_h and len(track['bboxes']) >= t_min]

        track_id = 0
        for i in range(0, len(self.tracks_finished)):
            self.tracks_finished[i]['id'] = track_id
            track_id += 1

        return self.tracks_finished
