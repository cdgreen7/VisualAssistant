"""ObjectTracker: associating detections across frames (no YOLO, no web)."""
import unittest

from vision.tracking import ObjectTracker, association_score, iou
from web.track_sessions import TrackSessions

STEP = 1.5  # seconds between awareness frames


def det(class_name, x1, y1, x2, y2):
    return {"class_name": class_name, "confidence": 0.9, "box": (x1, y1, x2, y2)}


def person(x, width=140):
    return det("person", x, 60, x + width, 340)


class ScoreTests(unittest.TestCase):
    def test_iou(self):
        self.assertEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)
        self.assertAlmostEqual(iou((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3)

    def test_association_score(self):
        box = (100, 100, 200, 300)
        self.assertEqual(association_score(box, box), 1.0)
        self.assertGreater(association_score(box, (210, 100, 310, 300)), 0.15)   # moved, no overlap, still near
        self.assertEqual(association_score(box, (500, 100, 600, 300)), 0.0)      # too far
        self.assertEqual(association_score(box, (100, 100, 400, 700)), 0.0)      # far too different in size
        self.assertEqual(association_score(box, (100, 100, 100, 300)), 0.0)      # empty box


class TrackerTests(unittest.TestCase):
    def setUp(self):
        self.tracker = ObjectTracker()

    def run_frames(self, frames, start=0.0):
        return [self.tracker.update(frame, start + i * STEP) for i, frame in enumerate(frames)]

    def test_one_object_keeps_its_id_across_consecutive_frames(self):
        ids = self.run_frames([[person(100)], [person(104)], [person(98)], [person(110)]])
        self.assertEqual(ids, [[1], [1], [1], [1]])

    def test_two_objects_of_the_same_class_keep_separate_ids(self):
        frames = [[person(40), person(400)], [person(60), person(380)], [person(380 - 30), person(80)]]
        self.assertEqual(self.run_frames(frames), [[1, 2], [1, 2], [2, 1]])  # order in the list doesn't matter

    def test_object_moving_between_image_regions_keeps_its_id(self):
        # walks left -> centre -> right, 110 px per frame (boxes barely or not overlapping)
        ids = self.run_frames([[person(x)] for x in (20, 130, 240, 350, 460)])
        self.assertEqual(ids, [[1]] * 5)

    def test_classes_are_never_mixed(self):
        here, there = (100, 60, 240, 340), (400, 60, 540, 340)
        frames = [[det("person", *here), det("chair", *there)],
                  [det("chair", *here), det("person", *there)]]
        first, second = self.run_frames(frames)
        self.assertEqual(first, [1, 2])
        self.assertEqual(sorted(second), [3, 4])  # a chair where the person was is a new object

    def test_object_temporarily_disappearing_keeps_its_id(self):
        ids = self.run_frames([[person(100)], [], [], [person(110)]])  # gone for 3 s, back at 4.5 s
        self.assertEqual(ids, [[1], [], [], [1]])

    def test_object_gone_too_long_gets_a_new_id_and_ids_are_not_reused(self):
        self.assertEqual(self.tracker.update([person(100)], 0.0), [1])
        self.assertEqual(self.tracker.update([person(100)], 6.0), [2])   # 6 s > 5 s
        self.assertEqual(len(self.tracker.tracks), 1)

    def test_new_object_entering_gets_the_next_id(self):
        ids = self.run_frames([[person(100)], [person(100), person(420)], [person(420), person(100), det("dog", 10, 250, 90, 340)]])
        self.assertEqual(ids, [[1], [1, 2], [2, 1, 3]])

    def test_reset_starts_again_from_one(self):
        self.run_frames([[person(100), person(420)]])
        self.tracker.reset()
        self.assertEqual(self.tracker.tracks, [])
        self.assertEqual(self.tracker.update([person(420)], 10.0), [1])

    def test_ambiguous_detection_gets_no_id(self):
        # two chair tracks almost on top of each other, then one detection exactly between them
        self.assertEqual(self.tracker.update([det("chair", 100, 100, 200, 200), det("chair", 320, 100, 420, 200)], 0.0), [1, 2])
        self.tracker.tracks[1].box = (120.0, 100.0, 220.0, 200.0)
        ids = self.tracker.update([det("chair", 110, 100, 210, 200)], STEP)
        self.assertEqual(ids, [None])                              # not guessed, and no ID invented
        self.assertEqual([t.track_id for t in self.tracker.tracks], [1, 2])
        self.assertEqual(self.tracker.tracks[0].box, (100.0, 100.0, 200.0, 200.0))  # tracks left untouched
        # once only one track fits, the ID comes back
        self.assertEqual(self.tracker.update([det("chair", 96, 100, 196, 200)], 2 * STEP), [1])

    def test_clear_winner_is_not_treated_as_ambiguous(self):
        self.run_frames([[person(100), person(200)]])              # overlapping people
        self.assertEqual(self.tracker.update([person(104), person(206)], STEP), [1, 2])

    def test_empty_frames(self):
        self.assertEqual(self.run_frames([[], [], []]), [[], [], []])
        self.assertEqual(self.tracker.tracks, [])


class TrackSessionTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.sessions = TrackSessions(idle_seconds=60, max_sessions=3, clock=lambda: self.now)

    def track(self, session, frame, detections, at=None):
        if at is not None:
            self.now = at
        return self.sessions.track(session, frame, detections)

    def test_concurrent_sessions_do_not_share_ids(self):
        self.assertEqual(self.track("phone-aaaa", 1, [person(100)], 0.0), ([1], "ok"))
        self.assertEqual(self.track("laptop-bbb", 1, [person(105), person(400)], 0.1), ([1, 2], "ok"))
        self.assertEqual(self.track("phone-aaaa", 2, [person(400), person(104)], 1.5), ([2, 1], "ok"))
        self.assertEqual(self.track("laptop-bbb", 2, [person(402)], 1.6), ([2], "ok"))

    def test_stale_and_repeated_frames_get_no_ids_and_change_nothing(self):
        self.track("phone-aaaa", 1, [person(100)], 0.0)
        self.track("phone-aaaa", 3, [person(110)], 3.0)
        self.assertEqual(self.track("phone-aaaa", 2, [person(500)], 3.1), ([None], "stale"))   # arrived late
        self.assertEqual(self.track("phone-aaaa", 3, [person(500)], 3.2), ([None], "stale"))   # repeated
        self.assertEqual(self.track("phone-aaaa", 4, [person(114)], 4.5), ([1], "ok"))         # track undisturbed

    def test_reset_forgets_a_session(self):
        self.track("phone-aaaa", 5, [person(100)], 0.0)
        self.assertTrue(self.sessions.reset("phone-aaaa"))
        self.assertFalse(self.sessions.reset("phone-aaaa"))
        self.assertEqual(self.track("phone-aaaa", 1, [person(400)], 1.0), ([1], "ok"))  # frame numbers restart too

    def test_idle_sessions_expire_and_the_number_of_sessions_is_capped(self):
        self.track("phone-aaaa", 1, [person(100)], 0.0)
        self.track("laptop-bbb", 1, [], 61.0)                      # 61 s later the phone session is gone
        self.assertEqual(len(self.sessions), 1)
        self.assertEqual(self.track("phone-aaaa", 1, [person(100)], 62.0), ([1], "ok"))
        for index, at in enumerate((63.0, 64.0, 65.0)):
            self.track(f"other-{index}xxx", 1, [], at)
        self.assertEqual(len(self.sessions), 3)                    # oldest ones were dropped


if __name__ == "__main__":
    unittest.main()
