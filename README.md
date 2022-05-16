# Resumo sobre o projeto

O Bioma Cerrado é conhecido pela biodiversidade de fauna e flora, mas principalmente pelo potencial agrícola. Por isso, suas variadas paisagens de cobertura e uso do solo s\~ao estudadas a fim de compreender aspectos socioambientais, econ\^omicos, e ambientais. Haja vista, a comunidade de Sensoriamento Remoto, ou seja, àquela que dispõe de dados adquiridos por satélite, aviões, ou estações em solo para observação da Terra, tem utilizado imagens de satélite com alta resolução espacial, tal como o CBERS-4A, para monitorar e mapear essas atividades no bioma. Tendo em vista o volume de imagens de satélite necessários para cobrir a extensão do bioma, técnicas tradicionais de Aprendizado de  Máquina, como Random Forest, deixam de ser triviais. Um vez que para rotular uma vegetação deve ser considerado contexto, ou seja, a dinâmica entre os elementos presente na cena. Por isso, esta proposta de dissertação, consiste em desenvolver um método de fácil implementação e eficiente para classificar os tipos de uso e cobertura no Cerrado, a nível de pixel, usando Redes Neurais Convolucionais de Aprendizagem Profunda, as quais são inspiradas no córtex visual humano, capaz de extrair características de baixo e alto nível. Assim, o método se chama LUCai. Para tanto, constitui-se especialmente na integração de duas redes profundas, a primeira para classificar imagens pelo seu contexto, enquanto a segunda, pixel a pixel. Ambos modelos serão avaliados por métricas que discriminam a precisão dos rótulos sobre as imagens, bem como por comparação com outros métodos apresentado na literatura.


# Informações gerais

Neste repositório estão anexados:

1. Algoritmos de Engenharia de dados: <br>
1.1 Mesclagem de bandas espectrais <br>
1.2 Recorte da imagem em tile <br>
1.3 Filtro de imagens sem dados <br>
1.4 Algoritmo de Data Split <br>

2. CNN de classificação contextual
2.1 CerraNetv3 - algoritmo de treinamento  <br>
2.2 CerraNetv3 - algoritmo de teste e pesos treinados <br>
Acesse: https://drive.google.com/drive/folders/1KSidsIiosZdrhDM-IF9KLckTKLytV_hC?usp=sharing  <br>
2.3 Support Vector Machine - algoritmo de treinamento e teste <br>
2.4 Random Forest - algoritmo de treinamento e teste <br>

3. Dataset1 (50.000 amostras rotuladas)
3.1 Split de treinamento, validação e teste <br>
Acesse: https://drive.google.com/drive/folders/1ODKZXJBbH1VrQWAAGMnHdOI-r4t5brBX?usp=sharing <br>

4. Proposta de Dissertação
Acesse: https://drive.google.com/drive/folders/1UwTmBaxpyQR6m-Tw5fmQTBxwW4Ph6hai?usp=sharing <br>
5. Keynote <br>
Acesse: https://drive.google.com/drive/folders/1pHNjv2cAj-DSQEyCCeP1YgIy-YDWwfhK?usp=sharing  <br>


