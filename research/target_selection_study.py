"""Phase 3 study: confidence-only vs geometric target selection.

    .venv/bin/python -m research.target_selection_study
    .venv/bin/python -m research.target_selection_study --evaluation research/results/evaluation-<time>.json --save

Three separate parts, never pooled:

1. Synthetic scenarios: hand-made boxes whose intended target is defined by the
   scenario itself. They show how each method behaves, including the cases it gets
   wrong; they are not a sample of real scenes and give no real-world accuracy.
2. Weight sensitivity: every weight combination on a 0.05 grid (wc + wa + wp = 1),
   on the synthetic scenarios, plus the exact decision boundary for a two-bottle case.
3. Saved real detections (unreviewed photos): which box each method picks and how
   often the choice changes across the weight grid. Descriptive only: without
   reviewed boxes nobody has said which box is the medicine.

Reviewed accuracy comes from vision/products/evaluate.py (selection_methods) once
reviewed boxes exist. Latency comes from research/benchmark.py
(selection_baseline, selection_weighted). Nothing here tunes the weights, and the
synthetic scenarios must never be used to choose them: their intended targets were
written by hand, so weights fitted to them would only reproduce those choices.
"""
import argparse
import json
import math
import os
import sys
import tempfile

from vision.target_selection import METHODS, Weights, select_target

W, H = 1920, 1080          # a webcam-sized frame for the synthetic scenarios
# The original baseline (identification.select_best_detection, main.py's default). Its
# validated counterpart is METHODS["confidence_validated"]: same choice on valid input,
# invalid boxes skipped. A difference between the two comes from validation alone.
ORIGINAL = "confidence_original"
GRID_STEP = 0.05


def bottle(confidence, box):
    return {"class_id": 39, "class_name": "bottle", "confidence": confidence, "box": box}


def other(class_name, confidence, box):
    return {"class_id": 0, "class_name": class_name, "confidence": confidence, "box": box}


def centred(width, height, dx=0, dy=0):
    """Box of the given size centred at (W/2 + dx, H/2 + dy)."""
    cx, cy = W // 2 + dx, H // 2 + dy
    return (cx - width // 2, cy - height // 2, cx + width // 2, cy + height // 2)


# (name, description, detections, intended index or None for "nothing should be read").
# Intended targets are defined by the scenario, not observed.
SCENARIOS = [
    ("single_bottle", "one bottle, off-centre", [bottle(0.71, centred(220, 500, dx=-500))], 0),
    ("confidence_vs_centre", "medicine centred (0.70), water bottle to the right (0.75), same size",
     [bottle(0.75, centred(220, 520, dx=620)), bottle(0.70, centred(220, 520))], 1),
    ("large_side_vs_small_centre", "medicine small and centred, larger bottle at the side, equal confidence",
     [bottle(0.70, centred(420, 900, dx=-650)), bottle(0.70, centred(160, 360))], 1),
    ("medicine_held_to_the_side", "medicine held at the left edge (0.78), another bottle centred (0.72)",
     [bottle(0.78, centred(230, 520, dx=-760)), bottle(0.72, centred(230, 520))], 0),
    ("medicine_close_up", "medicine close to the camera, background bottle more confident (0.86)",
     [bottle(0.86, centred(120, 300, dx=700, dy=-300)), bottle(0.66, centred(500, 1000, dx=60))], 1),
    ("medicine_small_in_background", "medicine far away and off-centre (0.80), large bottle in front (0.62)",
     [bottle(0.62, centred(450, 1000, dx=-80)), bottle(0.80, centred(110, 260, dx=650, dy=-250))], 1),
    ("three_bottles", "medicine slightly off-centre among two others",
     [bottle(0.74, centred(200, 480, dx=-700)), bottle(0.69, centred(240, 560, dx=150)),
      bottle(0.73, centred(200, 480, dx=750))], 1),
    ("distractor_cut_by_edge", "high-confidence bottle mostly outside the frame, medicine centred",
     [bottle(0.88, (1800, 200, 2300, 900)), bottle(0.70, centred(220, 520))], 1),
    ("invalid_box_present", "zero-area bottle box with the highest confidence, medicine centred",
     [bottle(0.95, (900, 300, 900, 800)), bottle(0.70, centred(220, 520))], 1),
    ("no_bottle", "person and cup only", [other("person", 0.9, centred(600, 1000)), other("cup", 0.8, centred(150, 200))],
     None),
]


def baseline_choice(detections):
    """main.py's default (identification.select_best_detection), as an index."""
    from vision.identification import select_best_detection
    chosen = select_best_detection(detections)
    return None if chosen is None else next(i for i, d in enumerate(detections) if d is chosen)


def choice(detections, weights, frame=(W, H)):
    selection = select_target(detections, frame, weights=weights)
    return None if selection.detection is None else selection.candidates[0].index


def synthetic_results():
    rows = []
    for name, description, detections, intended in SCENARIOS:
        picks = {ORIGINAL: baseline_choice(detections)}
        picks.update({method: choice(detections, weights) for method, weights in METHODS.items()})
        rows.append({"scenario": name, "description": description, "intended": intended, "picks": picks,
                     "correct": {method: pick == intended for method, pick in picks.items()}})
    return rows


def weight_grid(step=GRID_STEP):
    n = round(1 / step)
    return [Weights(round(i * step, 10), round(j * step, 10), round(1 - (i + j) * step, 10))
            for i in range(n + 1) for j in range(n + 1 - i)]


def sensitivity(rows_scenarios=SCENARIOS, step=GRID_STEP):
    """For each grid point, which synthetic scenarios pick their intended target."""
    grid = weight_grid(step)
    per_point = []
    for weights in grid:
        ok = [name for name, _, detections, intended in rows_scenarios if choice(detections, weights) == intended]
        per_point.append({"weights": [weights.confidence, weights.area, weights.center], "correct": ok})
    best = max(len(p["correct"]) for p in per_point)
    scenario_share = {name: round(sum(name in p["correct"] for p in per_point) / len(grid), 3)
                      for name, *_ in rows_scenarios}
    return {"grid_points": len(grid), "step": step, "max_scenarios_correct": best, "of": len(rows_scenarios),
            "points_reaching_max": sum(len(p["correct"]) == best for p in per_point),
            "share_of_grid_correct_per_scenario": scenario_share, "points": per_point}


def two_bottle_boundary(scenario="confidence_vs_centre"):
    """Exact condition under which the weighted score prefers the centred bottle.

    Prefer B over A  <=>  wc*(Cb - Ca) + wa*(Ab - Aa) + wp*(Pb - Pa) > 0."""
    _, _, detections, intended = next(s for s in SCENARIOS if s[0] == scenario)
    parts = select_target(detections, (W, H), "confidence_validated").candidates
    by_index = {c.index: c for c in parts}
    a, b = by_index[1 - intended], by_index[intended]
    d = {"confidence": b.confidence - a.confidence, "area": b.area - a.area, "center": b.center - a.center}
    weighted = METHODS["weighted"]
    margin = weighted.confidence * d["confidence"] + weighted.area * d["area"] + weighted.center * d["center"]
    # with wa = 0: centred bottle preferred when wp * dP > wc * (-dC)  ->  wp > wc * (-dC) / dP
    ratio = (-d["confidence"]) / d["center"] if d["center"] > 0 else math.inf
    return {"scenario": scenario, "deltas_intended_minus_other": {k: round(v, 4) for k, v in d.items()},
            "weighted_margin": round(margin, 4),
            "wa0_condition": f"wp/wc > {ratio:.4f} (centre weight at least {ratio / (1 + ratio):.3f} of wc+wp)"}


def saved_real_detections(evaluation_path=None, drafts_path=None):
    """[(source, image id, frame size, detections)] from a saved evaluation and/or drafts.json."""
    out = []
    if evaluation_path and os.path.exists(evaluation_path):
        with open(evaluation_path) as f:
            data = json.load(f)
        for image in data.get("images", []):
            if image.get("detections") is not None and image.get("size"):
                width, height = (int(v) for v in image["size"].split("x"))
                out.append(("evaluation (native resolution)", image["id"], (width, height),
                            [dict(d, box=tuple(d["box"])) for d in image["detections"]]))
    if drafts_path and os.path.exists(drafts_path):
        with open(drafts_path) as f:
            data = json.load(f)
        for item in data.get("items", []):
            obs = (item.get("suggestions") or {}).get("observations") or {}
            if obs.get("detections") is not None and obs.get("working_size"):
                name = os.path.basename(item.get("source", item["id"]))
                out.append(("drafts.json (1200x1600 working size; HEIC and JPEG copies are the same photo)",
                            name, tuple(obs["working_size"]),
                            [dict(d, box=tuple(d["box"])) for d in obs["detections"]]))
    return out


def real_descriptive(sources, step=GRID_STEP):
    """Which bottle each method picks on saved real detections (no correctness)."""
    grid = weight_grid(step)
    rows = []
    for source, image_id, frame, detections in sources:
        bottles = [i for i, d in enumerate(detections) if d.get("class_name") == "bottle"]
        picks = {ORIGINAL: baseline_choice(detections)}
        picks.update({method: choice(detections, weights, frame) for method, weights in METHODS.items()})
        grid_picks = {}
        for weights in grid:
            pick = choice(detections, weights, frame)
            grid_picks[pick] = grid_picks.get(pick, 0) + 1
        rows.append({"source": source, "image": image_id, "frame": list(frame), "bottles": len(bottles),
                     "bottle_boxes": {i: list(detections[i]["box"]) for i in bottles},
                     "bottle_confidences": {i: round(detections[i]["confidence"], 4) for i in bottles},
                     "picks": picks, "methods_agree": len(set(picks.values())) == 1,
                     "grid_share_by_pick": {str(k): round(v / len(grid), 3) for k, v in grid_picks.items()}})
    return rows


def latency(trials=200):
    """Selection latency from research/benchmark.py (separate processes, gated percentiles)."""
    from types import SimpleNamespace
    from research import benchmark
    with tempfile.TemporaryDirectory() as empty:
        args = SimpleNamespace(images=empty, trials=trials, timeout=120)
        return {name: benchmark.run_component(args, name, 0) for name in ("selection_baseline", "selection_weighted")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evaluation", help="a saved evaluate.py run (its YOLO detections are reused)")
    parser.add_argument("--drafts", default=os.path.join("research", "datasets", "medicine-v1", "drafts.json"),
                        help="drafts.json with earlier model predictions (optional, local only)")
    parser.add_argument("--trials", type=int, default=200, help="warm calls for the latency components")
    parser.add_argument("--save", action="store_true", help="write research/results/target-selection-<time>.json")
    args = parser.parse_args(argv)

    synthetic = synthetic_results()
    methods = [ORIGINAL] + list(METHODS)
    print("1. SYNTHETIC SCENARIOS (behaviour only; intended target defined by each scenario, not observed)")
    print("   scenario".ljust(34) + "".join(m[:15].ljust(16) for m in methods))
    for row in synthetic:
        cells = "".join(f"{row['picks'][m]!s:>2} {'ok' if row['correct'][m] else 'WRONG'}".ljust(16) for m in methods)
        print(f"   {row['scenario']:31}{cells}  (intended {row['intended']})")

    sens = sensitivity()
    boundary = two_bottle_boundary()
    print(f"\n2. WEIGHT SENSITIVITY on the {sens['of']} synthetic scenarios ({sens['grid_points']} weight points, "
          f"step {sens['step']})")
    print(f"   best any weight setting reaches: {sens['max_scenarios_correct']}/{sens['of']} "
          f"({sens['points_reaching_max']} grid points)")
    best_points = [p["weights"] for p in sens["points"] if len(p["correct"]) == sens["max_scenarios_correct"]]
    print(f"   weight settings (wc, wa, wp) reaching it: {best_points}")
    weighted = next(p for p in sens["points"] if p["weights"] == [0.4, 0.25, 0.35])
    print(f"   illustrative weights (0.40, 0.25, 0.35): {len(weighted['correct'])}/{sens['of']}")
    for name, share in sens["share_of_grid_correct_per_scenario"].items():
        print(f"   {name:31} intended target chosen at {share:.0%} of weight settings")
    print(f"   two-bottle boundary: {boundary}")

    real = real_descriptive(saved_real_detections(args.evaluation, args.drafts))
    print("\n3. SAVED REAL DETECTIONS (unreviewed: which box each method picks; nobody has said which is right)")
    for row in real:
        if row["bottles"] < 2:
            continue
        print(f"   {row['source']}, {row['image']}: bottles {row['bottle_confidences']} picks {row['picks']}"
              f"{'' if row['methods_agree'] else '  <- methods disagree'}; grid share {row['grid_share_by_pick']}")
    print(f"   images with 0 / 1 / 2+ bottles: "
          f"{sum(r['bottles'] == 0 for r in real)} / {sum(r['bottles'] == 1 for r in real)} / "
          f"{sum(r['bottles'] >= 2 for r in real)}")

    lat = latency(args.trials)
    print("\n4. SELECTION LATENCY (research/benchmark.py components, fixed 10 synthetic detections)")
    for name, result in lat.items():
        warm = result.get("warm_ms", {})
        fmt = lambda v: "<0.01" if v == 0 else v          # benchmark rounds to 0.01 ms
        print(f"   {name}: {result['status']}, warm median {fmt(warm.get('median'))} ms, p95 {fmt(warm.get('p95'))} ms, "
              f"max {warm.get('max')} ms, n={warm.get('n')}, cold {result.get('cold_ms')} ms")

    if args.save:
        from vision.products.evaluate import environment, save_results, DEFAULT_CACHE
        data = {"synthetic": synthetic, "sensitivity": sens, "two_bottle_boundary": boundary, "real_unreviewed": real,
                "latency": lat, "methods": {k: [w.confidence, w.area, w.center] for k, w in METHODS.items()},
                "evaluation_source": args.evaluation, "environment": environment(
                    args.evaluation or os.devnull, DEFAULT_CACHE) if args.evaluation else None}
        print("saved:", save_results(data, prefix="target-selection"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
