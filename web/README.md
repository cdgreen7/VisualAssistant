# Visual Assistant web interface

A phone (or this Mac) shows a camera preview in the browser. Pressing a button
captures **one still image** and uploads it to a Flask server on the Mac, which
runs the existing pipeline in `vision/` and returns JSON. No video is streamed,
and YOLO/EasyOCR never run on the phone.

```
iPhone Safari -> camera -> still image -> Flask on the Mac -> vision/ modules -> JSON -> browser
```

## Endpoints

| Method | Path            | Purpose |
|--------|-----------------|---------|
| GET    | `/`             | The camera page |
| GET    | `/health`       | `{"status": "ok", "loaded": [...]}` |
| POST   | `/api/detect`   | Image -> YOLO -> every detected object with LEFT/CENTER/RIGHT (environmental awareness) |
| POST   | `/api/barcode`  | Image -> barcodes -> FDA lookup + local UPC registry -> reconciled result |
| POST   | `/api/identify` | Image -> YOLO bottle -> crop -> EasyOCR -> ProductMatcher |

All POST endpoints take a multipart form with one file field named `image`
(JPEG or PNG, at most 10 MB). Errors are JSON: `{"error": {"code", "message"}}`.

## Use it on this Mac (Mac webcam)

`localhost` counts as a secure context, so no certificate is needed:

```bash
.venv/bin/python -m web.server
```

Open <http://localhost:5443> in Safari or Chrome on the Mac and press
**Start camera**. The browser uses the Mac webcam; the camera menu lists other
cameras if there are any. Quit `main.py` first if it is using the webcam.

## Use it from an iPhone

iPhone Safari only allows camera access on HTTPS pages with a certificate the
phone trusts. The setup below creates a private certificate authority (CA) on
your Mac with [mkcert](https://github.com/FiloSottile/mkcert) and trusts it on
your own phone. Nothing is exposed to the internet: the phone and the Mac must
be on the same Wi-Fi network.

### One-time setup on the Mac

1. Install mkcert and create the local CA (asks for your Mac password, because
   it adds the CA to your keychain):

   ```bash
   brew install mkcert
   mkcert -install
   ```

2. Find the Mac's network name and address:

   ```bash
   scutil --get LocalHostName     # e.g. Ridays-MacBook-Pro  -> Ridays-MacBook-Pro.local
   ipconfig getifaddr en0         # e.g. 192.168.1.23
   ```

3. Create a certificate for those names (run from the repo root, replacing the
   two example values):

   ```bash
   mkdir -p web/certs
   mkcert -cert-file web/certs/cert.pem -key-file web/certs/key.pem Ridays-MacBook-Pro.local 192.168.1.23 localhost 127.0.0.1
   ```

   `web/certs/` is git-ignored. If the Mac's IP address changes, either use the
   `.local` name on the phone or re-run this step with the new address.

### One-time setup on the iPhone

1. On the Mac, run `mkcert -CAROOT` and open that folder. AirDrop **only**
   `rootCA.pem` to the iPhone. Never share `rootCA-key.pem`: anyone holding it
   could impersonate any website to devices that trust this CA.
2. On the iPhone: Settings -> "Profile Downloaded" -> Install.
3. Settings -> General -> About -> Certificate Trust Settings -> turn on full
   trust for the "mkcert ..." certificate.

To undo it later: Settings -> General -> VPN & Device Management -> remove the
mkcert profile.

### Every time

```bash
.venv/bin/python -m web.server --host 0.0.0.0 --cert web/certs/cert.pem --key web/certs/key.pem
```

macOS may ask whether Python may accept incoming connections: allow it. Then,
with the iPhone on the same Wi-Fi, open in Safari:

```
https://Ridays-MacBook-Pro.local:5443      (or https://192.168.1.23:5443)
```

Press **Start camera** and allow camera access. Use **Flip** for the front/rear
camera.

The server refuses to listen on the network without `--cert/--key`, because
the phone camera would not work over plain HTTP anyway.

### Who can reach the server

With `--host 0.0.0.0`, any device on the same network can open the page and
upload images while the server is running. There is no login. Run it only on a
network you trust (home Wi-Fi or a personal hotspot, not campus or cafe Wi-Fi),
and stop it with Ctrl-C when you are done.

### Why not a tunnel (ngrok, Cloudflare Tunnel, ...)?

A tunnel gives a public HTTPS address without installing a certificate, but it
publishes this development server on the internet: anyone with the URL could
upload images and use your Mac's CPU, every image passes through a third
party, and the app has no authentication. It is not used here. If you ever
need one, put the tunnel provider's own access control (login or IP allowlist)
in front of it first.

## Tests

```bash
.venv/bin/python -m unittest discover tests
```

`tests/test_web_api.py` uses fake YOLO/OCR objects, so it does not load the models.
