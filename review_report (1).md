# Exercise 8 — Implementation Review

Live Page: https://grisposgiannis.github.io/mlr-ex08/
GitHub Repository: mlr-ex08

## 1. Mobile Testing

I tested the website on my phone by scanning the QR code. 

First, I tested the live camera by pointing it at a cherry leaf. The model recognized it correctly as cherry with 99.85% confidence, running at around 20 fps.

I also tested the photo upload option by selecting a picture of a maple leaf. The model correctly classified it as maple with 93.84% probability.

<p align="center">
  <img src="1000013792.jpg" alt="Mobile Live Camera Test" width="360"/>
  <img src="1000013793.jpg" alt="Mobile Upload Test" width="360"/>
</p>

## 2. Comparing the Results

I used the self-test button on the website to compare its output with the expected values from the notebook in check_expected.json.

- Apple: Notebook gave 0.829588, browser gave 0.829591.
- Cherry: Notebook gave 0.167952, browser gave 0.167949.
- Chestnut: Both gave approximately 0.002456.
- Maple: Both gave approximately 0.000004.

All differences are very small and well within the allowed tolerance of 0.001. The overall self-test result was PASS. 

These tiny differences happen because web browsers and Python handle image decoding and decompression slightly differently.

## 3. Code and Main Features

Looking through the index.html code, here is what I found for each requirement:

- Model and Runtime: The page uses ONNX Runtime Web to run leaf_model.onnx locally in the browser using WebAssembly. No photos or camera frames are sent to an external server.
- Preprocessing: The central square of each image is cropped and resized to 96 x 96 pixels. The RGB values are scaled to 0-1 and normalized using the mean and std values stored in model_info.json.
- Probabilities: The model outputs raw logits, and a softmax function inside index.html converts them into final probabilities.
- Camera and Interface: The page includes buttons to start, stop, and flip the camera, as well as an upload button. The probabilities are shown as live progress bars alongside an FPS counter and an input preview.
- Self-Test and Sharing: There is a self-test button that runs check_image.jpg automatically, and an in-browser QR code generator so the page can be opened quickly on a mobile phone.