"""Presentation of existing matcher decisions, never a second verifier."""
from web import serialize
from vision.products.matcher import ingredient_key, ProductMatcher
from vision.products.catalog import strength_key


def explain(ocr, result):
    match = serialize.match_result(result)
    # Keep all observations, including low-confidence/conflicting label text.
    label = ' · '.join(r['text'] for r in ocr if r.get('text', '').strip()) or 'No readable label text'
    ingredients = sorted({e.value for e in result.evidence if e.kind == 'ingredient'})
    strengths = sorted({e.value for e in result.evidence if e.kind == 'strength'})
    supported = []
    for c in result.candidates:
        # An informative formulation requires every ingredient and strength and
        # no conflicting evidence. It is NOT an upgrade to matcher VERIFIED.
        if (result.conflicts or c.missing_ingredients or c.base_only_ingredients or c.missing_strengths):
            continue
        pairs = []
        for name, strength in c.ingredients:
            # Multi-ingredient amounts must occur with their ingredient in one
            # uncorrected OCR observation; unordered strength sets aren't enough.
            if len(c.ingredients) > 1 and not any(
                (' ' + ingredient_key(name) + ' ') in (' ' + ingredient_key(r['text']) + ' ') and
                strength_key(strength) in {s for s, corrected in ProductMatcher._strengths(r['text'].upper()) if not corrected} and
                sum((' ' + ingredient_key(n) + ' ') in (' ' + ingredient_key(r['text']) + ' ')
                    for n, _ in c.ingredients) == 1 for r in ocr
                if r.get('confidence', 0) >= 0.6):
                break
            pairs.append(f'{name} {strength}')
        else:
            forms = sorted({r.get('dosage_form') or '' for r in c.records} - {''})
            supported.append(' + '.join(pairs) + (' (' + forms[0] + ')' if len(forms) == 1 else ''))
    formulation = supported[0] if len(set(supported)) == 1 else 'Not established; review label evidence'
    # Never drop an observed second ingredient from the headline.
    if len(ingredients) > 1 and not supported:
        formulation = 'Combination label observed; formulation uncertain'
    if len(ingredients) > 1 and not supported and match['status'] == 'VERIFIED' and match['level'] != 'PACKAGE':
        # Stricter presentation guard, not a change to ProductMatcher: its
        # unordered strengths do not prove ingredient-to-amount association.
        match.update(status='UNCERTAIN', level='NONE', package=None,
            reason=match['reason'] + '; ingredient-to-strength association is not established for this combination')
    return {'label_recognized': label, 'observed_ingredients': ingredients,
            'observed_strengths': strengths, 'formulation': formulation,
            'formulation_supported': len(set(supported)) == 1,
            'exact_package': ('Verified: NDC ' + result.package['package_ndc'])
                if result.status.value == 'VERIFIED' and result.level.value == 'PACKAGE' and result.package
                else 'Not verified',
            'match': match}
