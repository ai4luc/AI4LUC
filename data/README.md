## CerraData dataset

[CerraDatav3](https://www.kaggle.com/datasets/cerranet/cerradata-version3) has eight types of LULC, containing 80,000 patches, manually labeled via visual interpretation, as done in the second version [CerraDatav2](https://github.com/ai4luc/CerraData-code-data). The class names, as introduced in Figure below, are based on the thematic mapping of the [TerraClass](https://www.terraclass.gov.br/geoportal-cerrado/) project. In this version the samples were carefully audited by a committee composed of four LULC experts in the Cerrado biome, who are research members of TerraClass, DETER, and PRODES. There are 10,000 patches/class. This novel version can be accessed on [Kaggle platform](https://www.kaggle.com/datasets/cerranet/cerradata-version3).

However, to produce your own set of images access the `/data_engineering/` module to perform band compositing, `merge_bands.py`, cut the multispectral scene into 256X256 patches, `cut_and_filter_images.py, as well as remove those null-data patches. In addition to these utilities, other functions are available to help create the dataset.

![image](../set_page/img/datasets.jpeg)
