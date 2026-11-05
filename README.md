# Leaf classifier — live camera classification in the browser

**Open the page:** <https://grisposgiannis.github.io/mlr-ex08/>

Point your phone camera at a leaf and watch the probability of every class update
live. Inference runs entirely on-device in WebAssembly — no image is uploaded to
any server.

Works in Safari on iPhone (iOS 16.4 or newer, which is where WebAssembly SIMD
became available), and in current Chrome, Edge and Firefox on desktop and Android.

## What it does

`index.html` is a single self-contained page in plain JavaScript. It loads
`leaf_model.onnx` with [onnxruntime-web](https://github.com/microsoft/onnxruntime)
taken from the jsDelivr CDN (pinned to 1.21.0, WebAssembly backend) and runs the
rear camera through it frame by frame.

The four classes are `apple`, `cherry`, `chestnut` and `maple`. Each one gets a
labelled bar with its probability, refreshed every frame, plus the 96×96 image the
model actually received on its last pass.

A **Upload photo** button is the fallback for anywhere the camera is unavailable or
denied — the same pipeline runs over the chosen still.

## Preprocessing

Followed exactly as described in `model_info.json`, which the page re-reads at run
time:

| step | detail |
| --- | --- |
| crop | central square of the frame |
| resize | to `input_size` = 96 × 96 |
| scale | RGB bytes → 0…1 |
| standardize | per channel with `mean` `[0.485, 0.456, 0.406]`, `std` `[0.229, 0.224, 0.225]` |
| tensor | `float32`, layout `1 × 3 × 96 × 96` (RGB, channel-first) |

The model's output layer is named `logits`, so softmax is applied in the page to
turn them into probabilities.

Downscaling uses progressive halving so it stays alias-free in every browser, and
is arranged so that an input that is already 96×96 square passes through as a 1:1
copy.

## Self-test

`check_image.jpg` (which is exactly 96×96) and `check_expected.json` are both in
this repository, so the page can check itself against the reference numbers. Press
**Run self-test**, or append `?selftest=1` to the URL to run it on load.

Measured maximum absolute difference against `check_expected.json`:

| class | expected | this page | \|diff\| |
| --- | --- | --- | --- |
| apple | 0.829588 | 0.829591 | 2.73e-6 |
| cherry | 0.167952 | 0.167949 | 2.77e-6 |
| chestnut | 0.002456 | 0.002456 | 2.67e-8 |
| maple | 0.0000037 | 0.0000037 | 7.94e-11 |

The residual is the difference between the browser's and the reference decoder's
JPEG reconstruction, not a difference in the pipeline.

## QR code

The QR code at the bottom of the page is drawn onto a canvas by a vendored copy of
[qrcode-generator](https://www.npmjs.com/package/qrcode-generator) 1.5.0
(MIT, © Kazuhiko Arase), inlined into `index.html`. Nothing is fetched from a QR
image service and no network request is involved. It encodes the published address
above, so it keeps pointing at the deployed app even when the file is opened from
somewhere else, and there is a clickable text link beside it.

## Notes on iOS and GitHub Pages

* GitHub Pages serves no COOP/COEP headers, so the page has no
  `SharedArrayBuffer`. onnxruntime-web pins its WASM backend to a single thread in
  that situation; the page loads the `ort.wasm.min.js` bundle, which pulls a
  ~2.6 MB WASM binary rather than the ~4.4 MB one that only WebGPU needs.
* `getUserMedia` needs a secure origin, which Pages provides over HTTPS. The camera
  starts from a tap so the permission prompt follows a user gesture, the video
  element carries `playsinline`, and the capture constraints retry plainly if
  Safari rejects the preferred width and height.

## Files

| file | purpose |
| --- | --- |
| `index.html` | the whole application |
| `leaf_model.onnx` | the classifier (6.3 MB) |
| `model_info.json` | classes, input size, mean, std |
| `check_image.jpg` | reference input for the self-test |
| `check_expected.json` | reference probabilities for that input |
