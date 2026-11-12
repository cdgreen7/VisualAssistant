# Optional Intelligent Product Lookup

This is research, not a new verifier. ProductMatcher, FDA catalog and verified
UPC registry are unchanged. No medicine advice or automatic medicine speech is
added. Camera images stay in the existing local vision pipeline.

## Architecture and files

- `web/product_research.py`: separate official-source client, bounded requests,
  evidence comparison and memory cache; independent of YOLO/OCR/Qwen locks.
- `web/server.py`: `POST /api/products/research`, JSON evidence, 32 KB limit,
  injectable service for mocked tests and the existing structured error format.
- `web/static/app.js`: explicit research button on manual/automatic UNCERTAIN
  and NO_MATCH medicine results and decoded UNRESOLVED barcodes. Loading,
  retryable errors, expandable candidates, source links and evidence distinctions.
- `web/templates/index.html`: JavaScript cache version updated.
- `tests/test_product_research.py`: mocked service and API tests.
- `tests/js/product-research.test.js`: request/UI/concurrency/stale-result tests.
- `tests/test_frontend_js.py`: includes the new JavaScript suite.
- `web/README.md` and this document: setup and behavior.

No other vision algorithms or registry files are modified by this feature.

## Sources and query strategy

The backend calls the supported [openFDA Drug Label API](https://open.fda.gov/apis/drug/label/how-to-use-the-endpoint/).
Queries prefer an exact UPC, then an unambiguous package NDC, then observed
brand/ingredients/strengths/dosage form. The current matcher emits ingredient,
strength, brand and package NDC evidence; dosage form is supported if supplied.
Normalized NDCs are expanded to dashed layouts and compared after normalization.
Valid UPC-A/EAN-13 check digits are required. A UPC is never converted into an
invented NDC.

If an identifier finds no records and label evidence exists, one additional
text search is allowed and clearly labeled as approximate. Barcode-only
searches cannot do this until the user uses **Scan front label**. The result
shows up to 10 records and the total count; it does not pick the first result or
pretend that a truncated list is exhaustive.

Each candidate retains its openFDA record ID, effective date, source URL,
listed NDCs/UPCs, original active-ingredient text and purpose text. A DailyMed
link is added when an SPL set ID exists. [DailyMed's API](https://dailymed.nlm.nih.gov/dailymed/webservices-help/v2/spls_api.cfm)
also supports NDC/name/SPL discovery, but a second API adapter and XML label
parser are deferred: this implementation uses openFDA records and DailyMed
source links, not independent DailyMed corroboration.

## Evidence rules

- Original `VERIFIED`, `UNCERTAIN`, `NO_MATCH`, barcode statuses and registry
  contents remain unchanged. This service cannot verify or save a mapping.
- `SUPPORTED`, `INCOMPLETE`, `CONFLICTING` describe only comparison of observations
  against an individual retrieved record, never package verification.
- Compare the entire active ingredient set, including extra ingredients.
- Ingredient-to-strength association requires a separate observation containing
  that ingredient without another active ingredient, both in OCR and retrieved
  active-ingredient text. OCR confidence must be at least 0.6. Low-confidence
  observations remain visible. Digit-corrected matcher evidence is not searched.
- Preserve concentration denominators. Do not equate 200 mg/5 mL with 200 mg.
- Missing strength, mixed paragraphs, different strengths, brand discrepancies
  and unmatched dosage forms remain explicit. This intentionally favors
  incomplete comparisons over a guessed combination formulation.
- Exact UPC/NDC means that identifier occurs on that label record. A label can
  cover several packages; all candidates still say **Exact package: Not verified**.
- Purpose text is retrieved label information, not a recommendation for the
  package being held. Dosage/administration sections are excluded.
- Qwen observations are not transmitted, searched, summarized or validated here.
  Existing optional local Qwen reading remains unchanged. Deterministic evidence
  searches avoid introducing identifiers or latency from generated guesses.
- Unsupported web listings without official record identifiers are rejected.

## Configuration, network and costs

No extra dependencies or API key are required for the default implementation.
`PRODUCT_LOOKUP_ENABLED=0` disables external lookup on the server. Otherwise it
is available only on a button tap; there is no background/frame-by-frame search.
The UI explains that OCR text and/or barcode leave the Mac. No image is sent.

Optional `OPENFDA_API_KEY` is read from the server environment and never sent to
Safari or embedded in citation links. Obtain a free key through
[openFDA authentication](https://open.fda.gov/apis/authentication/). Without a key,
the documented limits are 240 requests/minute and 1,000/day per IP; see that
page for current keyed limits. HTTP 429 is shown as a rate-limit error; no retry
storm is generated.

Broader web search is **not connected**. A supported future option is
[Brave Search API](https://brave.com/blog/most-powerful-search-api-for-ai/), whose
published Search price is $5 per 1,000 requests with $5 monthly credit. Setup
would require an account, a server-side API key and a spending limit; check the
provider's current terms before subscribing. No key is requested, paid service
called, or Google results scraped by this implementation. Any future retailer
results must remain unverified discovery leads, separate from official records.

Each official call has a 12-second socket timeout and a 2 MB response cap. At
most two backend network requests run concurrently; the UI allows one research
request across cards. A fallback can require two sequential calls; the browser
aborts after 30 seconds. Network I/O occupies a Flask request thread, not a YOLO
or OCR lock. An abort does not forcibly cancel an in-progress Python socket.
Use the existing threaded Flask server. Successful queries are cached in memory
for one hour, up to 64 entries. Empty/error results are not cached. A cached
record is compared again against each request's observations. No images or
research results are written to disk. A detached result card ignores late replies.

## API example

```http
POST /api/products/research
Content-Type: application/json

{
  "evidence": [
    {"kind": "ingredient", "value": "IBUPROFEN"},
    {"kind": "strength", "value": "200mg"}
  ],
  "ocr": [{"text": "Ibuprofen 200 mg", "confidence": 0.95}]
}
```

For barcode-only research: `{"upc":"012345678905"}` (test fixture barcode;
not a claim of a real medicine mapping). For NDC, supply evidence kind `ndc`.

Response fields include `status` (`RECORDS_FOUND`/`NO_RESULTS`), `observed`,
`records`, `total_records`, `truncated`, `search_basis`, `cached`,
`processing_ms`, and a reminder that original statuses are unchanged.
Each record has ingredient/purpose text, comparison, conflicts/missing evidence,
exact identifier association flags and source links. Errors use the existing
`{"error":{"code":"research_timeout","message":"..."}}` format.

## Validation and measurements — November 11, 2026

Automated tests mock external services and cover exact NDC and UPC associations,
ingredient/strength and combination searches, swapped/missing/low-confidence
strengths, conflicting records, no results, HTTP/network errors, invalid JSON,
timeouts, rate limits, invalid evidence, rejected web listings, cache reuse and
expiry, bounded concurrency, opt-out, endpoint limits, and existing API health.
JavaScript tests cover explicit consent, excluding images/Qwen, duplicate clicks,
stale results, error retry, citations and all three entry points.

Final results: **256 Python unittest cases passed** (7.145 seconds), and
**65 JavaScript tests passed** (93 ms). This includes 24 new Python research
tests and 5 new JavaScript tests, plus the new unittest wrapper. Existing
matcher, manual identification, awareness, barcode, voice and Qwen tests pass.
`git diff --check` and Python/JavaScript syntax checks pass.

Commands:

```sh
.venv/bin/python -m unittest discover tests
node --test tests/js/*.test.js
```

Live service calls using generic public evidence (no camera photos):

| Query | Result | Network + comparison time |
|---|---|---|
| Ibuprofen 200 mg | 575 labels; first two comparisons supported | 1,025.8 ms |
| Package NDC 59779-438-12 from a returned label | One record, exact NDC association | 372.6 ms |
| Acetaminophen 250 mg + ibuprofen 125 mg | 69 labels; first two comparisons incomplete | 673.4 ms |
| Repeated cached text queries | Recompared observations | 2.2–2.3 ms |

The combination records retain both ingredients. Their mixed active-ingredient
paragraphs did not meet the conservative per-ingredient strength association
rule. These are discovery results, not identification of the user's bottles.
An earlier ibuprofen call took 1,249.1 ms. Live UPC association was not tested;
it is covered by mocks. No iPhone hardware test was performed in this task.

## Exact iPhone test steps

1. Restart the backend from the repository using your existing trusted certificate:
   `.venv/bin/python -m web.server --host 0.0.0.0 --cert web/certs/cert.pem --key web/certs/key.pem`.
   Keep the same certificate/hostname you already use. Internet access is needed
   on the Mac for research; it is not needed for existing local matching.
2. On the same Wi-Fi, reopen your existing `https://<Mac-address>:5443` URL in
   iPhone Safari and reload. Start the **iPhone** camera.
3. Enable automatic medicine scanning and hold the Equate or combination label
   steady, or use the existing manual **Identify medicine** action.
4. Expand an UNCERTAIN medicine card and tap **Search for product information**.
   Observe loading, then expand individual retrieved labels. Compare observed
   label text with retrieved ingredients/strengths/purpose and conflicts.
5. Verify combination observations retain acetaminophen AND ibuprofen. No card
   should claim that generic ibuprofen alone verifies the combination package.
6. Open a source link. Confirm it refers to that record; returning to the camera
   should preserve the local matcher result and should not speak medicine info.
7. Scan an unregistered UPC. Tap research on UNRESOLVED. If no official association
   exists, it stays unresolved. Use **Scan front label**, capture, then research
   the OCR result to enable an approximate text fallback.
8. Tap twice while loading: only one client request should run. Rescan while a
   request is pending: its old result must not replace the new card.
9. Disconnect the Mac from the internet and try an uncached query: expect a clear
   retryable error; existing local detection and matching should still operate.

Known limits: incomplete FDA UPC coverage, snapshot/cached label data, conservative
text parsing, no independent DailyMed fetch, no paid web search, no Qwen research,
no registry writes, no package verification upgrades, and no claim of real iPhone
validation for this new feature.
