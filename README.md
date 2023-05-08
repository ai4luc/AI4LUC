# Artificial Intelligence for Land Use and Land Cover Classification (AI4LUC): Source Code

## Overview
The <i>Cerrado</i> biome is known mainly for the biodiversity of flora, as well as for its potential in agricultural production. Observing the different landscapes of land use and land cover (LULC) is essential, as the analysis of LULC data allows us to understand the social, economic, and environmental aspects related to causative factors and impacts of these activities, as well as their subjects. There are many efforts employing machine learning (ML) or deep learning (DL) techniques for restricted cases of study in the biome. However, a few datasets have images with high spatial resolution, representativeness, and a huge number of samples about the biome are available. For supervised learning DL training of deep (DL) or machine learning (ML) models, dataset samples must be labeled. This procedure is performed manually and requires much time and attention, whether used for contextual or pixel-based classification tasks. In line with this, there are proposals that suggest learning by weak supervision, for example, in order to speed up the labeling of samples. In terms of semantic segmentation, labeling time is longer, since generation and classification of reference masks, i.e., indicate which class the segment belongs to. Supported by these motivations, this master dissertation aims at contributing to the task of pixel-based classification (semantic segmentation) of LULC via DL and a \textit{Cerrado} dataset of satellite images. In order to meet this goal, a method called <b>Artificial Intelligence for Land Use and Land Cover Classification (AI4LUC)</b>, is proposed. Firstly, a dataset regarding the <i>Cerrado</i> biome was created, called CerraData, amounting to unlabeled 2.5 million patches with height and width of 256 pixels, and two meters of spatial resolution. The spectral bands were obtained from the Wide Panchromatic and Multispectral Camera (WPM) of the China-Brazil Earth Resources-4A (CBERS-4A) satellite. From this dataset, two novel labeled versions were designed. Secondly, a convolutional neural network (CNN), named as CerraNetv3, was created. CerraNetv3 and Google DeepLabv3plus are jointly to support the pixel-based classification task. A novel technique has also been proposed to automatically generate and label reference masks, using CerraNetv3, in order to support DeepLabv3plus training for pixel-based classification. AI4LUC was compared to other derived approaches for semantic segmentation and contextual classification to analyze its feasibility. With regard to results, CerraNetv3 reached the highest score in the contextual classification experiment, scoring with an F1-score of 0.9289. Detailing the mask generation and labeling method scores, based on F1-score metric, the over all was 0.6738. Whereas DeepLabv3plus, based on the same metric, scored 0.2805.

## AI4LUC
The Artificial Intelligence for Land Use and Land Cover Classification1 (AI4LUC) is a method based on the methodology of the DETER, TerraClass, and PRODES [projects](http://www.obt.inpe.br/OBT/assuntos/programas/amazonia/prodes/pdfs/Metodologia_Prodes_Deter_revisada.pdf), in terms of image interpretation criteria such as context information and texture, to classify every single pixel of the scene. In line with this, AI4LUC is arranged in three hierarchies: modules, components, and functions, as presented in Figure below.

![image](set_page/img/pipeline.jpeg)

The first module is indicated for the pre-processing of the images, regarding the composition of spectral bands, as well as clipping into patches, i.e., the satellite image scene is cut into pieces of 256x256 pixels (height and width) `data/data_engineering/`.In the second module,`AIModels/contextual_classification/`, CerraNetv3, a DL model for contextual classification, has been trained with CerraDatav3 dataset. Thus, the trained model was employed in the third `smart_mask_labeling/` module to support the mask labeling, afterward mask generator component. The fourth module, `AIModels/sematic_segmentation/`, uses the labeled masks, produced previously, to train the DeepLabv3plus\cite{ChenDeepLabv3plus} network to obtain a pixel-based model fitted to deploy in an application.  Therefore, the first three modules concern the preparation of a dataset to train a semantic segmentation model and implement it in an application that assists in the LULC pixel-based classification procedure.

AI4LUC is a general method but an instance of a method was developed based on the Python language, the Conda environment, and using geoprocessing packages for remote sensing images, like GDAL and earthpy. Within the smart mask labeling module, for morphological transformations were used based on scikit-image and OpenCV. Regarding DL and DNNs, frameworks/APIs such as Tensorflow, Keras, scikit-learn, and PyTorch were used. The experiments were run on the [SDumont](https://sdumont.lncc.br/) supercomputer, using Bull Sequana X1120 computing nodes where each one has 2 x Intel Xeon Skylake 6252 CPU, 4 x NVIDIA Volta V100 GPUs, and 384 GB of RAM. For experiments that do not require more GPU capacity, a second computer was used with 8GB of RAM, an Apple M1 processor with an 8-core CPU, 7-core GPU, and 16-core Neural Engine. Further details regarding the method documentation, access [here](https://drive.google.com/file/d/1ez-55LoOPiyaCJdD-LrTJ2-UNBuEPODW/view?usp=share_link)


## Get start!!

To simply run the source code, do the following steps:

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installer for conda;
2. Create a new environment
    ```
    conda create -n env_name
    ```
3. Activate it
    ```
    conda activate env_name
    ```

4. Install the env.yml: 
    
    ```
    python==3.9 (or later)
    CUDA
    torch
    torchvision
    tensorflow
    tensorflow-metal (Apple M chips)
    scikit-learn
    scikit-image
    Pillow
    GDAL
    earthpy
    imageio
    numpy
    imgaug
    rasterio
    numpy
    segmentation-models-pytorch
    tqdm
    ```
For users of Apple devices with any version of the M chip, use the [tensorflow-metal](https://developer.apple.com/metal/tensorflow-plugin/) additional GPU usage setting to train models written with TensorFlow package more efficiently and quickly.

5. **Clone** the reposotiry and **download** the dataset, By default the running code will search for the dataset in `/cerradov3/` directory within. Or cloning the repository:
    ```
    git clone https://github.com/ai4luc/AI4LUC.git
    ```

7. If you want to create your own database, consider starting from the first module of the pipeline. Label your dataset by assigning a label to the image patch. Then use CerraNetv3 to learn how to classify your dataset. After the model is trained, use the `.hdf5` file in the python `ai_ContextualClassificatio.py` file inside the `smart_mask_labeling module`.

The specific use instructions for the modules are described in each directory in this repository. 

## Acknowledgments

This research was developed within the [**IDeepS**](https://github.com/vsantjr/IDeepS) project which is supported by the Brazilian LNCC/MCTI via resources of the [SDumont supercomputer](http://sdumont.lncc.br). This research was also supported by the Brazilian agencies CAPES. 

## Reference
MIRANDA, M. S. AI4LUC: pixel-based classification of land use and land cover via deep learning and a cerrado image dataset. version: 2023-04-03. 101 p. IBI: <8JMKD3MGP3W34T/48QQB65>. Dissertação (Mestrado em Computação Aplicada) - Instituto Nacional de Pesquisas Espaciais (INPE), São José dos Campos, 2023. Available in: ...

```
    @MastersThesis{Miranda::PiClLa,
                   author = "Miranda, Mateus de Souza",
                    title = "AI4LUC: pixel-based classification of land use and land cover via 
                             deep learning and a cerrado image dataset",
                   school = "Instituto Nacional de Pesquisas Espaciais (INPE)",
                  address = "S{\~a}o Jos{\'e} dos Campos",
                    month = "2023/03/28",
                 keywords = "classifica{\c{c}}{\~a}o de imagem baseada em pixels, rede neural 
                             convolucional, cerrado, sensoriamento remoto, CBERS-4A, image 
                             pixel-based classification, convolutional neural network, cerrado, 
                             remote sensing, CBERS-4A.",
                committee = "Santiago J{\'u}nior, Valdivino Alexandre de and K{\"o}rting, 
                             Thales Sehn and Shiguemori, Elcio and Escada, Maria Isabel Sobral 
                             and Papa, Jo{\~a}o Paulo",
             englishtitle = "AI4LUC: classifica{\c{c}}{\~a}o baseada em pixels de uso e 
                             cobertura da Terra atrav{\'e}s de aprendizado profundo e um 
                             conjunto de imagens sobre o Cerrado",
                 language = "en",
                    pages = "101",
                      ibi = "8JMKD3MGP3W34T/48QQB65",
                      url = "http://urlib.net/ibi/8JMKD3MGP3W34T/48QQB65",
               targetfile = "Master's dissertation by Mateus de Souza Miranda CAP INPE 
                             Official_com marcas.pdf",
            urlaccessdate = "09 abr. 2023"
    }
```
