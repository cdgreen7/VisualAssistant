"""Choosing which detected object to read (target selection).

main.py identifies "the" bottle in view. With several bottles, which one the user
means is not the same question as which one YOLO is most sure about. This module
scores each eligible detection from three normalized parts:

    S = wc * C + wa * A + wp * P          wc, wa, wp >= 0,  wc + wa + wp = 1

  C  YOLO confidence, used as reported (a probability-like score in [0, 1];
     never modified). A detection whose confidence is not a finite number in
     [0, 1] is rejected.
  A  box area as a fraction of the frame:  A = (w * h) / (W * H)  in (0, 1].
     The box is first clipped to the frame; a box with nothing left inside the
     frame is rejected. A is usually small (a bottle filling a tenth of the frame
     has A = 0.1), so area separates boxes weakly unless wa is large.
  P  closeness of the box centre (cx, cy) to the frame centre, per axis, so it
     doesn't depend on the resolution or aspect ratio:
         dx = (cx - W/2) / (W/2),   dy = (cy - H/2) / (H/2)      each in [-1, 1]
         P  = 1 - sqrt(dx^2 + dy^2) / sqrt(2)                       in [0, 1]
     P = 1 at the centre, 0 at a corner, 1 - 1/sqrt(2) = 0.29 at an edge midpoint.

Ties in S go to the higher confidence, then to the earlier detection (YOLO's
order), so the result is deterministic.

Two confidence-only selections exist and are kept apart:
  "confidence"            the ORIGINAL baseline, identification.select_best_detection:
                          maximum confidence among "bottle" detections, no validation.
                          main.py's default. Not implemented here.
  "confidence_validated"  this module with weights (1, 0, 0): the same choice whenever
                          every box and confidence is valid, but invalid detections
                          (zero-area or off-frame box, bad confidence) are skipped
                          instead of chosen. Any difference from the original comes from
                          that validation, not from geometric scoring.

Detection thresholds: the detections come from ObjectDetector, which already drops
boxes below its confidence threshold (0.5 by default, as in main.py). Selection never
adds, re-scores or re-thresholds detections; it only chooses among those it is given.

S is a ranking score for choosing what to read. It is not a probability that
the object is a medicine and has nothing to do with ProductMatcher's VERIFIED
decision or its confidence.

The weights are experimental starting values, not fitted or validated:
"weighted" = (0.40, 0.25, 0.35). They must not be tuned on synthetic scenarios.
The original baseline stays the default in main.py; every method here is opt-in
(see main.py and docs/phase-03-target-selection/README.md).
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

TARGET_CLASS = "bottle"          # same eligible class as vision/identification.py
SUM_TOLERANCE = 1e-9
SCORE_DECIMALS = 12              # scores equal to 12 decimals count as tied (float noise)


@dataclass(frozen=True)
class Weights:
    confidence: float
    area: float
    center: float

    def __post_init__(self):
        values = (self.confidence, self.area, self.center)
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0
                   for v in values):
            raise ValueError(f"weights must be finite, non-negative numbers: {values}")
        if abs(sum(values) - 1.0) > SUM_TOLERANCE:
            raise ValueError(f"weights must sum to 1, not {sum(values)}: {values}")


METHODS = {
    "confidence_validated": Weights(1.0, 0.0, 0.0),   # = select_best_detection, but invalid detections skipped
    "center": Weights(0.0, 0.0, 1.0),
    "area": Weights(0.0, 1.0, 0.0),
    "weighted": Weights(0.40, 0.25, 0.35),            # illustrative starting values, not optimized
}
# The original baseline, identification.select_best_detection (not in METHODS: it is not
# implemented here). main.py uses it unless VA_TARGET_SELECTION names a method above.
BASELINE = "confidence"
DEFAULT_METHOD = "confidence_validated"


@dataclass(frozen=True)
class Candidate:
    """Score breakdown for one detection (inspectable; nothing here changes the detection)."""
    index: int                    # position in the input list
    detection: dict
    eligible: bool
    reason: Optional[str]         # why it was rejected, if it was
    clipped_box: Optional[Tuple[int, int, int, int]] = None
    confidence: Optional[float] = None
    area: Optional[float] = None
    center: Optional[float] = None
    score: Optional[float] = None


@dataclass(frozen=True)
class Selection:
    method: str
    weights: Weights
    frame_size: Tuple[int, int]               # (width, height)
    detection: Optional[dict]                 # the chosen detection (the input dict itself), or None
    candidates: List[Candidate] = field(default_factory=list)   # every input detection, ranked eligible first

    @property
    def chosen(self):
        return next((c for c in self.candidates if c.eligible), None) if self.detection is not None else None


def resolve_weights(method=DEFAULT_METHOD, weights=None):
    """(name, Weights) for a named method, or custom weights (name "custom")."""
    if weights is not None:
        return "custom", weights if isinstance(weights, Weights) else Weights(*weights)
    if method == BASELINE:
        raise ValueError(f"{BASELINE!r} is the original identification.select_best_detection, not a scored "
                         f"method; use it directly, or 'confidence_validated'")
    if method not in METHODS:
        raise ValueError(f"unknown target-selection method {method!r}; choose from {sorted(METHODS)}")
    return method, METHODS[method]


def _frame_size(frame_size):
    try:
        width, height = (int(v) for v in frame_size)
    except (TypeError, ValueError):
        raise ValueError(f"frame_size must be (width, height), not {frame_size!r}")
    if width <= 0 or height <= 0:
        raise ValueError(f"frame_size must be positive, not {frame_size!r}")
    return width, height


def _clip(box, width, height):
    """Box clipped to the frame as ints, or (None, reason)."""
    if isinstance(box, (str, bytes)) or not hasattr(box, "__len__") or len(box) != 4:
        return None, "box is not four numbers"           # a string such as "1234" is not a box
    try:
        x1, y1, x2, y2 = (float(v) for v in box)
    except (TypeError, ValueError):
        return None, "box is not four numbers"
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None, "box has a non-finite coordinate"
    if x2 <= x1 or y2 <= y1:
        return None, "box has zero or negative area"
    cx1, cy1 = int(max(0, min(x1, width))), int(max(0, min(y1, height)))
    cx2, cy2 = int(max(0, min(x2, width))), int(max(0, min(y2, height)))
    if cx2 <= cx1 or cy2 <= cy1:
        return None, "box is outside the frame"
    return (cx1, cy1, cx2, cy2), None


def score_detection(detection, index, width, height, weights, class_name=TARGET_CLASS):
    """Candidate with its breakdown; ineligible detections carry the reason."""
    if not isinstance(detection, dict) or detection.get("class_name") != class_name:
        return Candidate(index, detection, False, f"class is not {class_name!r}")
    confidence = detection.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) \
            or not 0.0 <= confidence <= 1.0:
        return Candidate(index, detection, False, "confidence is not a number in [0, 1]")
    box, reason = _clip(detection.get("box"), width, height)
    if box is None:
        return Candidate(index, detection, False, reason)
    x1, y1, x2, y2 = box
    area = (x2 - x1) * (y2 - y1) / (width * height)
    dx = ((x1 + x2) / 2 - width / 2) / (width / 2)
    dy = ((y1 + y2) / 2 - height / 2) / (height / 2)
    center = max(0.0, 1.0 - math.hypot(dx, dy) / math.sqrt(2))
    score = weights.confidence * confidence + weights.area * area + weights.center * center
    return Candidate(index, detection, True, None, box, float(confidence), area, center, score)


def select_target(detections, frame_size, method=DEFAULT_METHOD, weights=None, class_name=TARGET_CLASS):
    """Selection of the eligible detection with the highest score S.

    detections: ObjectDetector.get_detections() output ({"class_name", "confidence",
    "box": (x1, y1, x2, y2)}), already filtered by the detector's confidence
    threshold. frame_size: (width, height) of the frame the boxes refer to.
    """
    name, chosen_weights = resolve_weights(method, weights)
    width, height = _frame_size(frame_size)
    scored = [score_detection(d, i, width, height, chosen_weights, class_name) for i, d in enumerate(detections or [])]
    eligible = sorted((c for c in scored if c.eligible),
                      key=lambda c: (-round(c.score, SCORE_DECIMALS), -c.confidence, c.index))
    rejected = [c for c in scored if not c.eligible]
    return Selection(name, chosen_weights, (width, height), eligible[0].detection if eligible else None,
                     eligible + rejected)


def make_selector(method=DEFAULT_METHOD, weights=None, class_name=TARGET_CLASS):
    """A function (detections, frame_size) -> detection or None, for identify_detection."""
    resolve_weights(method, weights)                  # fail early on a bad method or weights
    return lambda detections, frame_size: select_target(detections, frame_size, method, weights, class_name).detection


def describe(selection):
    """Readable score table, best first."""
    w = selection.weights
    lines = [f"target selection: {selection.method} (wc={w.confidence:.2f}, wa={w.area:.2f}, wp={w.center:.2f}), "
             f"frame {selection.frame_size[0]}x{selection.frame_size[1]}"]
    for c in selection.candidates:
        mark = "->" if selection.detection is not None and c is selection.candidates[0] else "  "
        if c.eligible:
            lines.append(f"{mark} #{c.index} S={c.score:.3f}  C={c.confidence:.3f} A={c.area:.3f} P={c.center:.3f} "
                         f"box={c.clipped_box}")
        else:
            label = c.detection.get("class_name") if isinstance(c.detection, dict) else None
            lines.append(f"   #{c.index} {label}: not eligible ({c.reason})")
    return "\n".join(lines)
