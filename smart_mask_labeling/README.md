## Smart mask labeling

Automated generation of the masks to support semantic segmentation is a very helpful direction to alleviate the efforts researchers must employ in real life settings. Within AI4LUC, the smart mask labeling module provides the automated mask generation and labeling. The masks are created via filters, thresholds, and morpho- logical operations algorithms in order to segment the elements of the scene. The coordinates of the mask segments are used to access the image pixels, which are input data to CerraNetv3. The output of this classification label is used to replace the pixels of each segment of the mask. Details on this pipeline are presented in Figure below.

![image](../set_page/img/smartlabelmask.jpeg)
