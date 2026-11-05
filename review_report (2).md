I tested the website (https://grisposgiannis.github.io/mlr-ex08/) on my phone by scanning the QR code. Using the live camera on a cherry leaf gave a correct prediction of cherry with 99.85% confidence at around 20 fps. Uploading a picture of a maple leaf also worked correctly, predicting maple with 93.84% probability.

I also ran the self-test button on the page to compare with check_expected.json. For apple, the notebook gave 0.829588 and the page gave 0.829591. For cherry, the notebook gave 0.167952 and the page gave 0.167949. For chestnut, both were around 0.002456, and for maple both were around 0.000004. All differences are below the 0.001 limit, so the test passed. The small differences come from minor variations in how browsers and Python decode images.

Looking at index.html, the model runs locally using ONNX Runtime Web and WebAssembly, so no images leave the device. Incoming frames are cropped to a central square, resized to 96x96, and normalized using the mean and std values from model_info.json. The model outputs logits, which are converted to probabilities using softmax. The page also includes camera controls, an upload button, live probability bars, an FPS counter, a self-test feature, and a built-in QR code generator.

![Mobile Live Camera Test](1000013792.jpg)

![Mobile Upload Test](1000013793.jpg)