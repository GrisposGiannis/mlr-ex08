# Exercise 8 --- Implementation Review

-   **Live Page:** https://grisposgiannis.github.io/mlr-ex08/
-   **GitHub Repository:** mlr-ex08

------------------------------------------------------------------------

## 1. Testing on a Mobile Phone

I tested the web application on my phone by scanning the QR code at the
bottom of the page. I tried both ways of providing an image: using the
live camera and uploading a photo.

### Live Camera

First, I used the phone's camera and pointed it at a cherry leaf. The
application processed the image directly in the browser and classified
it as **cherry**, with a confidence of **99.85%**. The camera was
running at around 20 fps during the test.

```{=html}
<p align="center">
```
`<img src="1000013792.jpg" alt="Mobile Live Camera Test" width="360"/>`{=html}
```{=html}
</p>
```
### Uploading a Photo

I also checked the upload option. For this test, I selected a photo of a
maple leaf. The prediction appeared straight away, and the model
correctly identified it as **maple**, with a probability of **93.84%**.

```{=html}
<p align="center">
```
`<img src="1000013793.jpg" alt="Mobile Upload Test" width="360"/>`{=html}
```{=html}
</p>
```

------------------------------------------------------------------------

## 2. Comparing the Probabilities (`check_image.jpg`)

I used the self-test button on the web page to check whether the
predictions from the browser were the same as the reference results
saved from the notebook in `check_expected.json`.

The results are shown below:

  ------------------------------------------------------------------------------
  Class             Notebook          Browser / Page    Absolute Difference
                    Probability       Probability       (\|diff\|)
  ----------------- ----------------- ----------------- ------------------------
  **apple**         0.829588          0.829591          $2.73 \times 10^{-6}$

  **cherry**        0.167952          0.167949          $2.77 \times 10^{-6}$

  **chestnut**      0.002456          0.002456          $2.67 \times 10^{-8}$

  **maple**         0.000004          0.000004          $7.94 \times 10^{-11}$
  ------------------------------------------------------------------------------

**Result: PASS.** All the differences are smaller than
$3 \times 10^{-6}$, so they are well below the allowed tolerance of
$10^{-3}$.

The small differences are most likely caused by the way the browser
reads JPEG images through the 2D canvas, compared with the way Python's
PIL library reads them. They do not indicate a problem with the model's
calculations.

------------------------------------------------------------------------

## 3. Checking the Implementation Requirements

I also went through `index.html` to check how the main requirements of
the assignment were covered.

1.  **Loading the model and running predictions in the browser**
    -   The application uses `onnxruntime-web` version 1.21.0, loaded
        through the `ort.wasm.min.js` file.
    -   The model runs in the browser using WebAssembly
        (`executionProviders: ["wasm"]`). This means that the image is
        processed on the user's device and is not sent to a server.
    -   I set `numThreads` to 1 because GitHub Pages does not provide
        the cross-origin isolation headers needed for
        `SharedArrayBuffer`. This avoids possible problems with browser
        threads.
2.  **Image preprocessing**
    -   The preprocessing settings are loaded from `model_info.json`.
        These include the target image size of $96 \times 96$ and the
        ImageNet mean and standard deviation.
    -   The image is cropped in the centre to make it square. It is then
        resized gradually by halving its dimensions, which helps avoid
        aliasing.
    -   The colour values are standardised using
        $(x - \text{mean}) / \text{std}$. A lookup table (LUT) is used
        to make this step faster.
    -   Before being passed to the model, the image is converted to a
        tensor with shape `[1, 3, 96, 96]`, using planar RGB format
        (channel-first).
3.  **Softmax**
    -   The ONNX model outputs raw logits rather than probabilities.
    -   For this reason, I added a separate `softmax()` function in
        JavaScript. It converts the logits into probabilities, which add
        up to 100%.
4.  **Interface and camera controls**
    -   The application can access the phone's rear camera through
        `navigator.mediaDevices.getUserMedia`, using
        `facingMode: "environment"`.
    -   There are controls to start, stop and switch the camera. Users
        can also upload an image from their device.
    -   The page shows the predicted probabilities with animated bars
        and highlights the class with the highest probability. It also
        displays the FPS and an inspect canvas, where the $96 \times 96$
        image that is passed to the model can be viewed.
5.  **Self-test and QR code**
    -   The **Run self-test** button processes `check_image.jpg` through
        the same frontend pipeline and compares the results with the
        values in `check_expected.json`.
    -   The page then displays the comparison in a table, together with
        a PASS or FAIL result.
    -   I also included a QR code generator written in JavaScript. It
        creates a QR code containing the GitHub Pages link, so the page
        can be opened more easily on a mobile phone without using an
        external QR service.
