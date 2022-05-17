# Resumo sobre o projeto

O Bioma Cerrado é conhecido principalmente pela biodiversidade de fauna e flora, bem como pelo potencial agrícola. Desse modo, suas variadas paisagens de cobertura e uso da terra são estudadas a fim de compreender aspectos sociais, econômicos, e ambientais. Haja vista, a comunidade de Sensoriamento Remoto têm utilizado imagens de satélite com alta resolução espacial para monitorar e mapear essas atividades. Tendo em vista o volume de imagens de satélite necessários para cobrir a extensão do bioma, técnicas de Aprendizado Profundo são adequadas e importantes para processá-las, devido a capacidade generalização de aprendizagem de máquina. Decerto para rotular um tipo de vegetação, o contexto e a dinâmica devem ser considerados. Por isso, esta proposta de dissertação, consiste em desenvolver um método para classificar os tipos de uso e cobertura no Cerrado, a nível de pixel, usando Redes Neurais Convolucionais de Aprendizagem Profunda. Chamado LUCai, termo em inglês atribuído à definição de Land Use and Land Cover in the Cerrado by the Deep Learning method. Para tanto, constitui-se especialmente na integração de duas redes profundas, a primeira para classificar imagens pelo seu contexto, enquanto a segunda, pixel a pixel; a região de interesse correponde a aproximadamente 44% da extensão total do Cerrado, cujos dados correpondem as imagens produzidas câmera WPM do Satélite CBERS-4A, no período de fevereiro de 2020 a fevereiro de 2022. Portanto, as redes serão aplicadas a volumoso dataset de imagens multiespectrais com alta resolução espacial. A avaliação do desepenho dos modelos seré dada pelas métricas Accuracy e F1-Score, e por comparação a outros métodos apresentado na literatura.


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


