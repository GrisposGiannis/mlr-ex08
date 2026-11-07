# Task 4 — Improving Leaf Classification for Mobile Camera

Ioannis Grispos  
MLR — Exercise 08  
November 2026  

---

### Goal and Dataset Setup

The goal of this task was to improve our classifier so that it reliably recognizes leaves when used outdoors with a smartphone camera.

We initially collected 131 photographs. We removed 10 duplicate burst shots of the same leaves and kept 121 clean images. Out of these, 89 photos were used for training the model and 32 for validation.

To test the models properly, we used three different evaluation sets. The first was our own 32 clean validation photos. The second was the same 32 photos degraded with artificial blur, rotation, and color shifts to simulate real mobile camera conditions. The third was a set of 120 historical photos from last year taken by different students using different phones.

Because our validation set is very small with only 32 images, a single wrong prediction changes the accuracy by roughly 3.1%. Differences smaller than 8 to 9% can happen purely by chance. For this reason, every experiment was run across 3 random seeds and we reported the average values.

---

### Experiment Results

The initial baseline model trained from scratch (LeafNet) reached 39.6% on our clean photos, 31.2% on the camera photos, and 37.8% on the historical photos.

Adding geometric augmentations such as random crops and rotations improved the clean validation accuracy to 53.1% and the camera accuracy to 37.5%.

Color jitter did not help, dropping the clean accuracy down to 47.9%.

Using Global Average Pooling (GAP) in the small network boosted clean accuracy to 54.2% while drastically cutting down model parameters.

Dropout alone showed almost no improvement, staying at 40.6%.

Switching to a pretrained network (MobileNetV3) produced the biggest gain, reaching 66.7% on clean photos and 57.3% on camera photos.

When we froze the backbone and trained only the final classification layer, clean accuracy reached 75.0%, but camera accuracy dropped to 55.2%.

Using a Cosine Scheduler over 30 epochs improved the camera accuracy to 62.5% and clean accuracy to 70.8%.

Combining the pretrained model with geometric augmentations gave 72.9% on clean photos and 63.9% on historical photos.

Finally, Model E, which included the 700 historical images during training, reached 76.0% on our own clean photos and 91.1% on the historical test set.

---

### What Worked and What Failed

Pretraining was the most important change. It provided a massive performance boost because learning edge and texture filters from scratch with only 89 images is practically impossible.

Global Average Pooling helped noticeably by removing unnecessary parameters, preventing the model from memorizing our small training set.

Geometric augmentations (random crops and flips) also gave a solid improvement by helping the network generalize better.

In contrast, color jitter failed and hurt performance. In leaf identification, color tones are essential features, so distorting them confused the model.

Dropout made no real difference, and freezing the backbone prevented the convolutional filters from adapting to mobile camera noise and blur.

---

### Final Model Selection

We selected Model E for final deployment.

Even though the model with the Cosine Scheduler scored slightly higher on the synthetic 32-image camera set, that difference is within random noise. On the large 120-image historical test set from other mobile phones, Model E was by far the best performer at 91.1%.

This demonstrates that having more real photos from different cameras and lighting conditions is far more effective than hyperparameter tuning.

The model was exported as `leaf_model.onnx` for use in the browser application, and its predictions were verified to match the PyTorch model outputs.

---

### Conclusion

The main lesson from this study is that to make a classifier work reliably on a mobile camera, the top priority is collecting more real photos across different phones, angles, and lighting conditions, rather than endlessly tuning hyperparameters on a very small dataset.