"""Lightweight object tracking across successive detection results.

Each detection is associated with at most one existing track of the SAME
class, using box overlap (IoU) and, when a box has moved too far to overlap,
how close the box centres are relative to the box size. Frames arrive about
every 1.5 seconds, so there is no motion model: this is plain association,
not ByteTrack.

Track IDs are temporary numbers, valid for one tracker (one awareness run).
A detection gets track_id None when it can't be associated reliably, i.e.
when two existing tracks fit it about equally well.
"""
from dataclasses import dataclass

MIN_SCORE = 0.15         # weakest association accepted
AMBIGUITY_MARGIN = 0.10  # two tracks this close in score for one detection -> no ID
MAX_AGE_SECONDS = 5.0    # a track not seen for this long is dropped (its ID is never reused)
MAX_CENTRE_SHIFT = 1.0   # centre may move up to this many box diagonals between frames
MAX_AREA_RATIO = 2.5     # boxes whose areas differ more than this are not the same object


@dataclass
class Track:
    track_id: int
    class_name: str
    box: tuple
    last_seen: float
    hits: int = 1


def iou(a, b):
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    overlap = width * height
    return overlap / (_area(a) + _area(b) - overlap)


def _area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _centre(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _diagonal(box):
    return ((box[2] - box[0]) ** 2 + (box[3] - box[1]) ** 2) ** 0.5


def association_score(track_box, box):
    """0 (not the same object) .. 1 (identical box)."""
    small, large = sorted((_area(track_box), _area(box)))
    if small <= 0 or large / small > MAX_AREA_RATIO:
        return 0.0
    (ax, ay), (bx, by) = _centre(track_box), _centre(box)
    reach = MAX_CENTRE_SHIFT * max(_diagonal(track_box), _diagonal(box))
    closeness = max(0.0, 1 - ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 / reach)
    return max(iou(track_box, box), 0.5 * closeness)


class ObjectTracker:
    def __init__(self, max_age_seconds=MAX_AGE_SECONDS, min_score=MIN_SCORE, ambiguity_margin=AMBIGUITY_MARGIN):
        self.max_age_seconds = max_age_seconds
        self.min_score = min_score
        self.ambiguity_margin = ambiguity_margin
        self.reset()

    def reset(self):
        self.tracks = []
        self._next_id = 1

    def update(self, detections, now):
        """Track IDs (int or None) for detections, in the same order.

        detections: dicts with "class_name" and "box" (x1, y1, x2, y2).
        now: time of this frame in seconds, from a clock that never goes back.
        """
        self.tracks = [t for t in self.tracks if now - t.last_seen <= self.max_age_seconds]
        boxes = [tuple(float(v) for v in d["box"]) for d in detections]

        # Every plausible (detection, track) pair, best first
        pairs = []
        for index, detection in enumerate(detections):
            for track in self.tracks:
                if track.class_name != detection["class_name"]:
                    continue
                score = association_score(track.box, boxes[index])
                if score >= self.min_score:
                    pairs.append((score, index, track))
        pairs.sort(key=lambda pair: (-pair[0], pair[1], pair[2].track_id))

        ids = [None] * len(detections)
        decided, used = set(), set()
        for score, index, track in pairs:
            if index in decided or track.track_id in used:
                continue
            # Another free track fits this detection about as well: don't guess between them
            rivals = [s for s, i, t in pairs
                      if i == index and t is not track and t.track_id not in used and score - s < self.ambiguity_margin]
            decided.add(index)
            if rivals:
                continue
            used.add(track.track_id)
            track.box, track.last_seen, track.hits = boxes[index], now, track.hits + 1
            ids[index] = track.track_id

        for index, detection in enumerate(detections):
            if index in decided:
                continue
            track = Track(self._next_id, detection["class_name"], boxes[index], now)
            self._next_id += 1
            self.tracks.append(track)
            ids[index] = track.track_id
        return ids
