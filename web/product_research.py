"""On-demand official label research. Never writes registry or verifies a package."""
import copy
import json
import os
import re
import socket
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, quote
from urllib.request import urlopen

from vision.products.ndc import normalize_upc, normalize_package_ndc
from web.services import ServiceError

API = 'https://api.fda.gov/drug/label.json'


def clean(value):
    return re.sub(r'[^A-Z0-9]+', ' ', value.upper()).strip()


def strings(record, key):
    value = record.get(key, [])
    return [s for s in value if isinstance(s, str)][:50] if isinstance(value, list) else []


def amounts(text):
    # Preserve denominator: 200 mg/5 mL must never equal a 200 mg tablet.
    return {re.sub(r'\s+', '', m.lower()).removesuffix('/1') for m in re.findall(
        r'\b\d+(?:\.\d+)?\s*(?:mcg|mg|g|%)(?:\s*/\s*(?:\d+(?:\.\d+)?\s*(?:ml|g)?|ml))?', text, re.I)}


def paired_amounts(lines, ingredient, ingredients):
    found = set()
    for line in lines:
        # Require an isolated ingredient in this observation; ambiguous multi-
        # ingredient paragraphs are deliberately not strength association proof.
        present = [n for n in ingredients if ' '+clean(n)+' ' in ' '+clean(line)+' ']
        if present == [ingredient]:
            found.update(amounts(line))
    return found


def observations(payload):
    if not isinstance(payload, dict):
        raise ServiceError(400, 'invalid_evidence', 'Supply a JSON evidence object.')
    evidence = payload.get('evidence', [])
    ocr = payload.get('ocr', [])
    if not isinstance(evidence, list) or not isinstance(ocr, list) or len(evidence)>40 or len(ocr)>100:
        raise ServiceError(400, 'invalid_evidence', 'Too many or invalid observations.')
    result = {'ingredients': [], 'strengths': [], 'brands': [], 'ndcs': [], 'forms': [], 'label': [], 'reliable_label': []}
    keys = {'ingredient':'ingredients', 'strength':'strengths', 'brand':'brands', 'ndc':'ndcs', 'dosage_form':'forms'}
    for e in evidence:
        if not isinstance(e, dict) or not isinstance(e.get('value'), str) or len(e['value'])>160:
            raise ServiceError(400, 'invalid_evidence', 'Invalid label evidence.')
        if e.get('kind') in keys and e['value'].strip() and not e.get('corrected_from'):
            result[keys[e['kind']]].append(e['value'].strip())
    for row in ocr:
        if not isinstance(row, dict) or not isinstance(row.get('text'), str) or len(row['text'])>500:
            raise ServiceError(400, 'invalid_evidence', 'Invalid OCR observation.')
        result['label'].append(row['text'])
        confidence = row.get('confidence', 0)
        if isinstance(confidence, (int, float)) and 0.6 <= confidence <= 1:
            result['reliable_label'].append(row['text'])
    for key in keys.values():
        result[key] = list(dict.fromkeys(result[key]))
    upc = payload.get('upc', '')
    if not isinstance(upc, str) or (upc and not normalize_upc(upc)):
        raise ServiceError(400, 'invalid_upc', 'Use a valid UPC-A or EAN-13 with check digit.')
    result['upc'] = normalize_upc(upc) if upc else None
    for ndc in result['ndcs']:
        if not normalize_package_ndc(ndc):
            raise ServiceError(400, 'invalid_ndc', 'NDC must be an unambiguous package code.')
    if not (result['upc'] or result['ndcs'] or result['ingredients'] or result['brands']):
        raise ServiceError(400, 'missing_evidence', 'Scan a readable label or valid barcode first.')
    return result


def query_for(obs):
    def term(field, value):
        return field+':"'+re.sub(r'[^a-zA-Z0-9 .%/-]', ' ', value)+'"'
    if obs['upc']:
        values = {obs['upc'], obs['upc'][1:] if obs['upc'].startswith('0') else obs['upc']}
        return '('+' OR '.join(term('openfda.upc', v) for v in sorted(values))+')'
    if obs['ndcs']:
        # FDA indexes dashed package codes; canonical 11 digits may have several
        # ten-digit layouts. Query all, then compare normalized returned IDs.
        values = set(obs['ndcs'])
        for n in obs['ndcs']:
            n = normalize_package_ndc(n)
            parts = [n[:5], n[5:9], n[9:]]
            values.add('-'.join(parts))
            for i in range(3):
                if parts[i].startswith('0'):
                    p = parts.copy(); p[i] = p[i][1:]; values.add('-'.join(p))
        return '('+' OR '.join(term('openfda.package_ndc', v) for v in sorted(values))+')'
    terms = [term('openfda.substance_name', v) for v in obs['ingredients']]
    terms += [term('active_ingredient', re.sub(r'(\d)([a-zA-Z])', r'\1 \2', v.removesuffix('/1'))) for v in obs['strengths']]
    terms += [term('openfda.brand_name', v) for v in obs['brands']]
    terms += [term('dosage_forms_and_strengths', v) for v in obs['forms']]
    return ' AND '.join(terms)


def assess(record, obs):
    official = record.get('openfda', {})
    if not isinstance(official, dict): official = {}
    names = strings(official, 'substance_name')
    active = strings(record, 'active_ingredient')
    returned = {clean(n) for n in names}
    expected = {clean(n) for n in obs['ingredients']}
    conflicts, missing = [], []
    if expected and returned and expected != returned:
        conflicts.append('Active ingredient set differs (including additional ingredients).')
    elif expected and not returned:
        missing.append('Structured active ingredient names unavailable.')
    for name in obs['ingredients']:
        observed = paired_amounts(obs['reliable_label'], name, obs['ingredients'])
        retrieved = paired_amounts(active, name, names or obs['ingredients'])
        if not observed or not retrieved:
            missing.append(name+': ingredient-to-strength association incomplete.')
        elif observed != retrieved:
            conflicts.append(name+': observed and retrieved strengths differ.')
    brands = strings(official, 'brand_name')
    if obs['brands'] and not {clean(b) for b in obs['brands']}.intersection(clean(b) for b in brands):
        conflicts.append('Brand evidence differs from returned record.')
    forms = strings(record, 'dosage_forms_and_strengths')
    if obs['forms'] and not all(any(clean(f) in clean(r) for r in forms) for f in obs['forms']):
        missing.append('Dosage form not corroborated.')
    upcs = strings(official, 'upc')
    ndcs = strings(official, 'package_ndc')
    exact_upc = bool(obs['upc'] and obs['upc'] in {normalize_upc(u) for u in upcs})
    exact_ndc = bool(set(map(normalize_package_ndc, obs['ndcs'])) &
                     {normalize_package_ndc(n) for n in ndcs} - {None})
    if obs['upc'] and not exact_upc: missing.append('Exact UPC not present in returned record.')
    if obs['ndcs'] and not exact_ndc: missing.append('Exact package NDC not present in returned record.')
    identifier = record.get('id', '')
    identifier = identifier if isinstance(identifier, str) else ''
    source_url = API+'?'+urlencode({'search':'id:"'+identifier+'"'})
    set_id = record.get('set_id', '')
    daily = ('https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid='+quote(set_id)) if isinstance(set_id,str) and re.fullmatch(r'[a-fA-F0-9-]{36}',set_id) else None
    return {'source':'openFDA Drug Label', 'source_id':identifier, 'source_url':source_url,
            'dailymed_url':daily, 'effective_time':record.get('effective_time'),
            'brands':brands, 'ingredients':names, 'active_ingredient_text':active,
            'purpose':strings(record,'purpose'), 'dosage_form_text':forms,
            'package_ndcs':ndcs, 'upcs':upcs, 'exact_upc':exact_upc, 'exact_ndc':exact_ndc,
            'comparison':'CONFLICTING' if conflicts else ('SUPPORTED' if expected and not missing else 'INCOMPLETE'),
            'conflicts':conflicts, 'missing':missing,
            'exact_package':'Not verified; retrieved label association only'}


class ProductResearch:
    def __init__(self, fetch=None, clock=time.monotonic):
        self.fetch = fetch or self._fetch
        self.clock = clock
        self.cache = {}
        self.lock = threading.Lock()
        self.slots = threading.BoundedSemaphore(2)

    def _fetch(self, query):
        params = {'search':query, 'limit':10}
        if os.getenv('OPENFDA_API_KEY'): params['api_key'] = os.environ['OPENFDA_API_KEY']
        try:
            with urlopen(API+'?'+urlencode(params), timeout=12) as response:
                raw = response.read(2_000_001)
            if len(raw)>2_000_000: raise ValueError('response too large')
            return json.loads(raw)
        except HTTPError as e:
            if e.code == 404: return {'results':[], 'meta':{'results':{'total':0}}}
            raise ServiceError(429 if e.code==429 else 502, 'research_rate_limit' if e.code==429 else 'research_unavailable',
                               'Official lookup rate limited; try later.' if e.code==429 else 'Official lookup unavailable.') from e
        except (TimeoutError, socket.timeout) as e:
            raise ServiceError(504, 'research_timeout', 'Official lookup timed out; try again.') from e
        except URLError as e:
            timeout = isinstance(e.reason, (TimeoutError, socket.timeout))
            raise ServiceError(504 if timeout else 502, 'research_timeout' if timeout else 'research_unavailable', 'Official lookup could not be reached.') from e
        except (ValueError, UnicodeError) as e:
            raise ServiceError(502, 'research_invalid_response', 'Official lookup returned an invalid response.') from e

    def research(self, payload):
        if os.getenv('PRODUCT_LOOKUP_ENABLED','1') == '0':
            raise ServiceError(503, 'research_disabled', 'External product lookup is disabled.')
        start = self.clock()
        obs = observations(payload)
        query = query_for(obs)
        data, cached = self._retrieve(query)
        search_basis = 'identifier' if obs['upc'] or obs['ndcs'] else 'label_evidence'
        if not data['results'] and search_basis == 'identifier' and (obs['ingredients'] or obs['brands']):
            text_obs = dict(obs, upc=None, ndcs=[])
            data, cached = self._retrieve(query_for(text_obs))
            search_basis = 'label_evidence_after_identifier_not_found'
        records = [assess(r,obs) for r in data['results'][:10]]
        try:
            total = data.get('meta',{}).get('results',{}).get('total',len(records))
            if not isinstance(total, int) or total < len(records): raise ValueError()
        except (AttributeError, ValueError):
            raise ServiceError(502, 'research_invalid_response', 'Official lookup returned invalid result metadata.')
        return {'status':'RECORDS_FOUND' if records else 'NO_RESULTS', 'observed':obs,
                'records':records, 'total_records':total, 'truncated':total>len(records),
                'message':'Research only. Original identification and barcode status are unchanged. Exact package is not verified.',
                'search_basis':search_basis, 'cached':cached, 'processing_ms':round((self.clock()-start)*1000,1)}

    def _retrieve(self, query):
        start = self.clock()
        cached = False
        with self.lock:
            entry = self.cache.get(query)
            if entry and entry[0]>start: data = copy.deepcopy(entry[1]); cached = True
        if not cached:
            if not self.slots.acquire(blocking=False):
                raise ServiceError(409, 'research_busy', 'Other product lookups are running. Try again shortly.')
            try: data = self.fetch(query)
            finally: self.slots.release()
            if not isinstance(data, dict) or not isinstance(data.get('results'), list) or any(not isinstance(r,dict) or not isinstance(r.get('id'),str) or not r['id'] for r in data['results']):
                raise ServiceError(502, 'research_invalid_response', 'Official lookup returned an invalid response.')
            if data['results']:
                with self.lock:
                    if len(self.cache)>=64: self.cache.pop(next(iter(self.cache)))
                    self.cache[query] = (start+3600, copy.deepcopy(data))
        return data, cached
