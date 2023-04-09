## Smart mask labeling

Automated generation of the masks to support semantic segmentation is a very helpful direction to alleviate the efforts researchers must employ in real life settings. Within AI4LUC, the smart mask labeling module provides the automated mask generation and labeling. The masks are created via filters, thresholds, and morphological operations algorithms in order to segment the elements of the scene. The coordinates of the mask segments are used to access the image pixels, which are input data to CerraNetv3. The output of this classification label is used to replace the pixels of each segment of the mask. Details on this pipeline are presented in Figure below.

Each component, illustrated in the Figure, is integrated into the `main.py` file. Therefore, the main parameters of basic adjustments can be made in the file. For advanced tweaks, specific to part of the module, access the other `.py` files. For example, in the case of contextual classification neural network replacement, access `ai_ContexClassify.py`. To adjust the mask-generating functions, go to `filters.py`. To add access to sliding window functions or other module implementation utilities, change the `tools.py` file.

To use this module considering the CerraNetv3's weights to masks labeling, please introduce the path or paths of the image (patch) input in `main.py`. It is essecial remaind you, the input patch most be 256X256px and comprising one of those eight land use and land cover of the third version of [CerraData](https://www.kaggle.com/datasets/cerranet/cerradata-version3), with its false-color bands composition.

![image](../set_page/img/smartlabelmask.jpeg)
