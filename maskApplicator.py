# Pessoal, considerando a possível dificuldade de vocês para encontrar a parte correspondente nas imagens de referência, eu resolvi enviar essa base em anexo com as seguintes características:

# latent: as imagens de latentes (as mesmas da fase 02 do projeto)
# masks: as máscaras de segmentação da impressão digital nas imagens de latente. Essas máscaras foram obtidas pelo convex-hull dilatado das minúcias detectadas.
# aligned_reference: as imagens de referência alinhadas com as imagens das impressões digitais das latentes. O alinhamento foi realizado a partir das minúcias de ambas.

# A fase 01 do projeto pode ser facilitada da seguinte forma:

# Vocês podem aplicar a máscara em ambas imagens, latente e referência alinhada, para extrair as respectivas regiões de interesse com as partes das impressões 
# digitais que serão comparadas usando a rede neural da fase 01. 
# 
# Montem, portanto, novas pastas com essas regiões de interesse, 

# criem os arquivos de treino e teste que comparam genuínos (imagens com o mesmo basename) e impostores (imagens com basenames distintos), 

# treinem e avaliem a rede neural da fase 01 com esses arquivos, e busquem melhorar os resultados dela. 

# Depois, vocês podem fazer um script para que dada uma imagem de consulta (impressão latente), 
# o script verifique a acurácia de encontrar na base de referência essa imagem entre as k mais próximas para diferentes valores de k (1, 10, 100).

import os
from xmlrpc.client import Boolean
import cv2
from PIL import Image  
import PIL
from os.path import exists

maskPath = "in-masks/"
referencePath = "in-aligned_reference/"
latentPath = "in-latent/"
outReferencePath = "out-masked-aligned_reference/" 
outLatentPath = "out-masked-latent/"

def applyMask(maskName, imageName, outputPath):
    image = cv2.imread(imageName)
    mask = cv2.imread(maskName, 0)
    mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    image[mask!=255] = (0,0,0)
    cv2.imwrite(outputPath,image)

def checkHasReerenceAndLatentEquivalentToMask() -> Boolean:
    allIsOk = True
    for (root, dirs, file) in os.walk(maskPath):
        for f in file:
            if '.png' in f:
                if(not exists(referencePath + f)): 
                    print(referencePath + f)
                    allIsOk = False
                if(not exists(latentPath + f)):
                    print(latentPath + f)
                    allIsOk = False
    return allIsOk

# criando mascara + referencia e mascara + latente
if checkHasReerenceAndLatentEquivalentToMask():
    for (root, dirs, file) in os.walk(maskPath):
        print("------------------------------------------------")
        for f in file:
            if '.png' in f:
                applyMask(maskPath + f, referencePath + f, outReferencePath + f)
                applyMask(maskPath + f, latentPath + f, outLatentPath + f)

else:
    print("Existem mascaras sem referencia ou latente equivalente")













# image = cv2.imread('latent/b101-9_l.png')
# mask = cv2.imread('masks/b101-9_l.png', 0)
# mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
# image[mask!=255] = (0,0,0)

# cv2.imshow('image', image)
# cv2.imwrite('image2.png',image)

# cv2.waitKey()