"""Independent human review of evaluation annotations and bounding boxes.

Each reviewer annotates on their own, without seeing anyone else's answers, and
their answers go to their own file (reviews/<reviewer>.json next to the labels).
`reconcile` then sets each image's review level in labels.json:

  no annotation             UNREVIEWED (truth fields empty)
  one reviewer              SINGLE_REVIEWED (exploratory metrics)
  two or more who agree     REVIEWED (primary metrics)
  reviewers who disagree    UNREVIEWED, with the disagreeing fields listed; discuss,
                            then one of you re-annotates with --replace

Nothing else sets a level: labels prepared by an assistant, model predictions
(drafts.json) and matcher output are never annotations, and software names are
refused as reviewers.

From the repository root:

    .venv/bin/python -m research.annotation_review status
    .venv/bin/python -m research.annotation_review annotate photo-IMG_9766 --reviewer "RJ"
    .venv/bin/python -m research.annotation_review preview photo-IMG_9766 --reviewer "RJ"
    .venv/bin/python -m research.annotation_review reconcile --dry-run
    .venv/bin/python -m research.annotation_review reconcile

`annotate` opens the image in an OpenCV window to draw boxes (drag a box, press
ENTER or SPACE to keep it, draw the next one, press ESC when finished), then asks
questions in the terminal. Box coordinates are pixels of the image as OpenCV
loads it (EXIF orientation applied), which is also what the evaluator uses.
`--labels` selects another dataset in the same format (default: photos-v1);
`--no-boxes` skips box drawing (label artwork).

Agreement rules: the same negative/readable answers, brand words, active
ingredients and strengths, package NDC, UPC and front-panel text (compared
case- and punctuation-insensitively), the same number of boxes per role with
each pair overlapping by IoU >= BOX_AGREEMENT_IOU (the agreed box is their mean),
and the same grouping of photos into physical products (the IDs themselves may
differ). The product name is free text and is not compared.
"""
import argparse
import datetime
import json
import os
import re
import sys

from vision.products.catalog import name_key
from vision.products.evaluate import (BOX_ROLES, file_sha256, formulation_key, iou, is_person_name, label_id,
                                      load_labels, match_boxes)
from vision.products.ndc import normalize_package_ndc

DEFAULT_LABELS = os.path.join("research", "datasets", "photos-v1", "labels.json")
BOX_AGREEMENT_IOU = 0.8
TRUTH_FIELDS = ("physical_product_id", "negative", "readable", "product", "brand_words", "ingredients",
                "formulation", "package_ndc", "upc", "visible_text", "target_boxes", "intended_target")
EMPTY_TRUTH = {"physical_product_id": None, "negative": None, "readable": None, "product": None, "brand_words": [],
               "ingredients": [], "formulation": [], "package_ndc": None, "upc": None, "visible_text": [],
               "target_boxes": None, "intended_target": None}
COMPARED = ("negative", "readable", "brand_words", "formulation", "package_ndc", "upc", "visible_text")


# Files

def reviews_dir(labels_path):
    return os.path.join(os.path.dirname(labels_path), "reviews")


def reviewer_slug(name):
    slug = re.sub(r"[^a-z0-9]+", "-", str(name).strip().lower()).strip("-")
    if not slug or not is_person_name(name):
        raise ValueError(f"{name!r} is not a person's name or initials")
    return slug


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def load_reviews(labels_path):
    """{reviewer name: {image id: annotation}}; files naming software are skipped."""
    folder, out = reviews_dir(labels_path), {}
    if not os.path.isdir(folder):
        return out
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(folder, name)) as f:
            data = json.load(f)
        reviewer = data.get("reviewer")
        if not is_person_name(reviewer):
            print(f"! skipped {name}: reviewer {reviewer!r} is not a person")
            continue
        out[reviewer] = data.get("annotations", {})
    return out


def save_annotation(labels_path, reviewer, image_id, annotation, replace=False):
    path = os.path.join(reviews_dir(labels_path), reviewer_slug(reviewer) + ".json")
    data = {"reviewer": reviewer, "annotations": {}}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        if data.get("reviewer", "").strip().lower() != reviewer.strip().lower():
            raise ValueError(f"{path} belongs to {data.get('reviewer')!r}")
    if image_id in data["annotations"] and not replace:
        raise ValueError(f"{reviewer} already annotated {image_id}; pass --replace to change it")
    data["annotations"][image_id] = annotation
    _write_json(path, data)
    return path


# Building and checking one annotation

def parse_ingredient(line):
    """'Ibuprofen | 200 mg' -> ['Ibuprofen', '200 mg']."""
    name, sep, strength = line.partition("|")
    if not sep or not name.strip() or not re.search(r"\d", strength):
        raise ValueError(f"write 'name | strength', e.g. 'Ibuprofen | 200 mg', not {line!r}")
    return [" ".join(name.split()), " ".join(strength.split())]


def build_annotation(answers, boxes, image_sha256, today=None):
    """Annotation dict from answers (dict of strings/lists/bools) and drawn boxes; raises ValueError."""
    negative = bool(answers["negative"])
    ingredients = [parse_ingredient(line) for line in answers.get("ingredients", []) if line.strip()]
    if not negative and not ingredients:
        raise ValueError("a medicine needs at least one active ingredient with its strength")
    if negative and ingredients:
        raise ValueError("a non-medicine image has no active ingredients")
    pid = str(answers.get("physical_product_id") or "").strip()
    if not pid:
        raise ValueError("physical_product_id is required")
    ndc = str(answers.get("package_ndc") or "").strip() or None
    if ndc and not normalize_package_ndc(ndc):
        raise ValueError(f"{ndc!r} is not a package NDC")
    upc = re.sub(r"\D", "", str(answers.get("upc") or "")) or None
    for box in boxes or []:
        if box.get("role") not in BOX_ROLES:
            raise ValueError(f"box role must be one of {BOX_ROLES}")
    medicine_boxes = sum(b["role"] == "medicine" for b in boxes or [])
    if boxes is not None and not negative and medicine_boxes != 1:
        raise ValueError(f"draw exactly one medicine box (drew {medicine_boxes})")
    if negative and medicine_boxes:
        raise ValueError("a non-medicine photo has no medicine box")
    return {
        "image_sha256": image_sha256,
        "date": (today or datetime.date.today()).isoformat(),
        "confirmation_source": "physical package" if answers.get("checked_package") else "image only",
        "physical_product_id": pid,
        "negative": negative,
        "readable": bool(answers["readable"]),
        "product": (answers.get("product") or "").strip() or None,
        "brand_words": [w for w in str(answers.get("brand_words") or "").split() if w],
        "ingredients": ingredients,
        "package_ndc": ndc,
        "upc": upc,
        "visible_text": [line.strip() for line in answers.get("visible_text", []) if line.strip()],
        "target_boxes": boxes,
        "intended_target": _intended(answers.get("intended_target"), boxes),
    }


def _intended(value, boxes):
    """The user's intended target: None (not annotated), "none" (nothing should be read)
    or an index into boxes. Separate from the medicine box on purpose."""
    if value is None or value == "":
        return None
    if value == "none":
        return "none"
    if boxes is None or isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < len(boxes):
        raise ValueError(f"intended target must be a drawn box number, 'none' or empty, not {value!r}")
    return value


def _intent_key(annotation):
    """("none",), (role, box) or None, for comparing intent across reviewers' own box lists."""
    intended = annotation.get("intended_target")
    if intended is None or intended == "none":
        return intended
    box = annotation["target_boxes"][intended]
    return box["role"], box["box"]


def intents_agree(a, b, threshold=BOX_AGREEMENT_IOU):
    first, second = _intent_key(a), _intent_key(b)
    if first is None or second is None or first == "none" or second == "none":
        return first == second
    return first[0] == second[0] and iou(first[1], second[1]) >= threshold


def merged_intent(first, merged_boxes):
    """Index of the first reviewer's intended box in the merged boxes (same role, best overlap)."""
    key = _intent_key(first)
    if key is None or key == "none":
        return key
    role, box = key
    options = [(iou(b["box"], box), i) for i, b in enumerate(merged_boxes) if b["role"] == role]
    return max(options)[1]


def catalog_has_formulation(catalog, ingredients):
    truth = formulation_key(ingredients)
    return any(formulation_key([(i["name"], i["strength"]) for i in r["active_ingredients"]]) == truth
               for r in catalog.records)


# Agreement

def _normalized(annotation, field):
    value = annotation.get(field)
    if field == "brand_words":
        return sorted(name_key(" ".join(value or [])).split())
    if field == "formulation":
        return sorted(formulation_key(annotation.get("ingredients") or []))
    if field == "package_ndc":
        return normalize_package_ndc(value) if value else None
    if field == "visible_text":
        return sorted(k for k in (name_key(line) for line in value or []) if k)
    return value


def boxes_agree(a, b, threshold=BOX_AGREEMENT_IOU):
    if (a is None) != (b is None):
        return False
    if a is None:
        return True
    for role in BOX_ROLES:
        first, second = [x["box"] for x in a if x["role"] == role], [x["box"] for x in b if x["role"] == role]
        if len(first) != len(second) or len(match_boxes(first, second, threshold)) != len(first):
            return False
    return True


def merge_boxes(annotations, threshold=BOX_AGREEMENT_IOU):
    """Mean box per agreed pair/group, matched to the first annotation's boxes."""
    if annotations[0]["target_boxes"] is None:
        return None
    merged = []
    for role in BOX_ROLES:
        base = [x["box"] for x in annotations[0]["target_boxes"] if x["role"] == role]
        groups = [[box] for box in base]
        for other in annotations[1:]:
            boxes = [x["box"] for x in other["target_boxes"] if x["role"] == role]
            for ti, pi, _ in match_boxes(base, boxes, threshold):
                groups[ti].append(boxes[pi])
        merged += [{"role": role, "box": [int(round(sum(b[k] for b in g) / len(g))) for k in range(4)]} for g in groups]
    return merged


def grouping_conflicts(reviews):
    """Images whose physical-product grouping differs between reviewers (IDs may differ)."""
    names = sorted(reviews, key=str.lower)
    conflicts = set()
    for other in names[1:]:
        common = set(reviews[names[0]]) & set(reviews[other])
        for image in common:
            same = lambda r: {j for j in common if reviews[r][j]["physical_product_id"] == reviews[r][image]["physical_product_id"]}
            if same(names[0]) != same(other):
                conflicts.add(image)
    return conflicts


def reconcile(labels, reviews):
    """New labels list (input unchanged) and a list of human-readable differences."""
    grouping = grouping_conflicts(reviews)
    out, report = [], []
    for label in labels:
        image_id, current = label_id(label), dict(label)
        found = []
        for reviewer in sorted(reviews, key=str.lower):
            annotation = reviews[reviewer].get(image_id)
            if annotation is None:
                continue
            if label.get("sha256") and annotation.get("image_sha256") != label["sha256"]:
                report.append(f"{image_id}: {reviewer}'s annotation is for a different file (sha256); ignored")
                continue
            found.append((reviewer, annotation))
        # Truth written by an earlier reconcile is cleared whenever it no longer holds;
        # fields prepared before any review (e.g. the assistant's DailyMed transcription,
        # which the pilot's regression replay uses) are left as they are.
        was_reviewed = (label.get("review") or {}).get("status", "UNREVIEWED") != "UNREVIEWED"
        if not found:
            if was_reviewed:
                current.update(EMPTY_TRUTH)
                current["review"] = {"status": "UNREVIEWED"}
            out.append(current)
            continue
        first = found[0][1]
        differences = [field for field in COMPARED
                       if any(_normalized(a, field) != _normalized(first, field) for _, a in found[1:])]
        if any(not boxes_agree(first["target_boxes"], a["target_boxes"]) for _, a in found[1:]):
            differences.append("target_boxes")
        elif any(not intents_agree(first, a) for _, a in found[1:]):
            differences.append("intended_target")
        if len(found) > 1 and image_id in grouping:
            differences.append("physical_product_id grouping")
        names = [r for r, _ in found]
        sources = sorted({a["confirmation_source"] for _, a in found})
        if differences:
            if was_reviewed:
                current.update(EMPTY_TRUTH)
            current["review"] = {"status": "UNREVIEWED", "conflicting_reviewers": names, "conflicts": differences}
            report.append(f"{image_id}: {', '.join(names)} disagree on {', '.join(differences)}")
            out.append(current)
            continue
        current.update({
            "physical_product_id": first["physical_product_id"], "negative": first["negative"],
            "readable": first["readable"], "product": first.get("product"), "brand_words": first["brand_words"],
            "ingredients": first["ingredients"], "formulation": first["ingredients"],
            "package_ndc": first["package_ndc"], "upc": first["upc"], "visible_text": first["visible_text"],
            "target_boxes": merge_boxes([a for _, a in found]),
        })
        current["intended_target"] = (None if current["target_boxes"] is None
                                      else merged_intent(first, current["target_boxes"]))
        current["review"] = {"status": "REVIEWED" if len(found) > 1 else "SINGLE_REVIEWED", "reviewers": names,
                             "date": max(a["date"] for _, a in found), "confirmation_sources": sources}
        out.append(current)
    return out, report


# Interactive parts (kept thin; the logic above is tested without a screen)

def ask_yes_no(ask, question):
    while True:
        answer = ask(question + " [y/n] ").strip().lower()
        if answer in ("y", "yes", "n", "no"):
            return answer.startswith("y")


def ask_lines(ask, question):
    print(question + " (one per line; empty line to finish)")
    lines = []
    while True:
        line = ask("  > ")
        if not line.strip():
            return lines
        lines.append(line)


def draw_boxes(image, title):
    """Boxes drawn in an OpenCV window, in full-image pixels."""
    import cv2
    scale = min(1.0, 900 / image.shape[0], 1400 / image.shape[1])
    shown = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image
    print(f"{title}: drag a box, ENTER/SPACE to keep it, draw the next, ESC when finished (none: ESC at once)")
    rects = cv2.selectROIs(title, shown, showCrosshair=False)
    cv2.destroyWindow(title)
    cv2.waitKey(1)
    boxes = []
    height, width = image.shape[:2]
    for x, y, w, h in (rects if len(rects) else []):
        x1, y1 = int(round(x / scale)), int(round(y / scale))
        x2, y2 = min(width, int(round((x + w) / scale))), min(height, int(round((y + h) / scale)))
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])
    return boxes


def ask_intended(ask, boxes):
    """The object the person meant the system to read. Recorded separately from the
    medicine box: a photo can show two medicines, or the user can mean something else.
    Empty = not known (then intent is simply not evaluated for this photo)."""
    for i, box in enumerate(boxes):
        print(f"  box {i}: {box['role']} {box['box']}")
    while True:
        answer = ask("Which box did the person intend the system to read? Box number, 'none' if nothing, "
                     "empty if you don't know: ").strip().lower()
        if answer in ("", "none"):
            return answer or None
        if answer.isdigit() and int(answer) < len(boxes):
            return int(answer)


def interview(ask, negative_hint=None):
    """Answers dict from the terminal."""
    print("Answer from the physical package where you can; otherwise from the photo. Don't look at anyone "
          "else's answers, model output or earlier annotations.")
    answers = {"negative": not ask_yes_no(ask, "Does this photo show a medicine package to identify?")}
    answers["physical_product_id"] = ask(
        "Short ID for the main package/object (same ID for every photo of the same physical item, e.g. P1): ")
    answers["checked_package"] = ask_yes_no(ask, "Did you check these answers against the physical package?")
    answers["readable"] = ask_yes_no(ask, "Is the identifying text (brand, ingredient, strength) readable in THIS photo?")
    if not answers["negative"]:
        answers["product"] = ask("Product name as printed (free text): ")
        answers["brand_words"] = ask("Brand words as printed (e.g. 'equate' or 'Advil PM'): ")
        answers["ingredients"] = ask_lines(ask, "Each active ingredient and strength as printed, 'name | strength', "
                                                "e.g. 'Ibuprofen | 200 mg' or 'Acetaminophen | 160 mg/5 mL'")
        answers["package_ndc"] = ask("Package NDC as printed (empty if not printed or unreadable): ")
        answers["upc"] = ask("UPC digits under the barcode (empty if none): ")
        answers["visible_text"] = ask_lines(ask, "Front-panel text exactly as printed, line by line "
                                                 "(used as perfect OCR; empty to skip)")
    return answers


def annotate(labels_path, image_id, reviewer, replace=False, with_boxes=True, ask=input, drawer=draw_boxes,
             catalog=None):
    import cv2
    reviewer_slug(reviewer)
    labels = {label_id(label): label for label in load_labels(labels_path)}
    if image_id not in labels:
        raise ValueError(f"no image {image_id}; run status to list them")
    label = labels[image_id]
    digest = file_sha256(label["image"])
    if label.get("sha256") and digest != label["sha256"]:
        raise ValueError(f"{label['image']} changed since the labels were prepared (sha256)")
    answers = interview(ask)
    boxes = None
    if with_boxes:
        image = cv2.imread(label["image"])
        boxes = [] if answers["negative"] else [{"role": "medicine", "box": b} for b in
                                                drawer(image, "Draw the MEDICINE package (one box)")]
        boxes += [{"role": "other_bottle", "box": b} for b in
                  drawer(image, "Draw every OTHER bottle (not the medicine)")]
        answers["intended_target"] = ask_intended(ask, boxes)
    annotation = build_annotation(answers, boxes, digest)
    if catalog is not None and annotation["ingredients"] and not catalog_has_formulation(catalog, annotation["ingredients"]):
        print("note: no catalog record has exactly these ingredients and strengths. Check the salt names "
              "(e.g. 'Diphenhydramine HCl') and liquid concentrations ('160 mg/5 mL').")
        if not ask_yes_no(ask, "Keep these ingredients as printed?"):
            raise ValueError("not saved; run annotate again")
    print(json.dumps({k: v for k, v in annotation.items() if k != "image_sha256"}, indent=2))
    if not ask_yes_no(ask, "Save this annotation?"):
        raise ValueError("not saved")
    return save_annotation(labels_path, reviewer, image_id, annotation, replace)


def preview(labels_path, image_id, reviewer, out_dir=None):
    """Draw one reviewer's own boxes on the image (local, gitignored folder)."""
    import cv2
    labels = {label_id(label): label for label in load_labels(labels_path)}
    annotation = load_reviews(labels_path).get(reviewer, {}).get(image_id)
    if annotation is None:
        raise ValueError(f"{reviewer} has no annotation for {image_id}")
    image = cv2.imread(labels[image_id]["image"])
    for box in annotation["target_boxes"] or []:
        colour = (0, 200, 0) if box["role"] == "medicine" else (0, 140, 255)
        cv2.rectangle(image, tuple(box["box"][:2]), tuple(box["box"][2:]), colour, max(3, image.shape[1] // 300))
    out_dir = out_dir or os.path.join(os.path.dirname(labels_path), "previews")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{image_id}-{reviewer_slug(reviewer)}.jpg")
    cv2.imwrite(path, image)
    return path


def status(labels_path):
    labels, reviews = load_labels(labels_path), load_reviews(labels_path)
    for label in labels:
        image_id = label_id(label)
        who = [r for r in sorted(reviews, key=str.lower) if image_id in reviews[r]]
        review = label.get("review") or {}
        extra = f" conflicts: {review['conflicts']}" if review.get("conflicts") else ""
        print(f"{image_id:22} {review.get('status', 'UNREVIEWED'):16} annotated by: {', '.join(who) or '-'}{extra}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("status", "annotate", "preview", "reconcile"))
    parser.add_argument("image_id", nargs="?")
    parser.add_argument("--reviewer")
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--no-boxes", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "status":
            status(args.labels)
        elif args.command in ("annotate", "preview"):
            if not args.image_id or not args.reviewer:
                parser.error(f"{args.command} needs an image id and --reviewer")
            if args.command == "preview":
                print("wrote", preview(args.labels, args.image_id, args.reviewer))
            else:
                catalog = None
                from vision.products.evaluate import DEFAULT_CACHE
                if os.path.exists(DEFAULT_CACHE):
                    from vision.products.catalog import ProductCatalog
                    catalog = ProductCatalog.from_file(DEFAULT_CACHE)
                print("saved", annotate(args.labels, args.image_id, args.reviewer, args.replace,
                                        not args.no_boxes, catalog=catalog))
        else:
            labels = load_labels(args.labels)
            new_labels, report = reconcile(labels, load_reviews(args.labels))
            for line in report:
                print(line)
            for label in new_labels:
                print(f"{label_id(label):22} {label['review']['status']}")
            if not args.dry_run:
                from vision.products.evaluate import validate_label
                problems = [f"{label_id(l)}: {p}" for l in new_labels for p in validate_label(l)]
                if problems:
                    raise ValueError("not saved:\n" + "\n".join(problems))
                _write_json(args.labels, new_labels)
                print("updated", args.labels)
    except ValueError as error:
        print(f"error: {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
