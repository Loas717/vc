import cv2
import numpy as np

img = cv2.imread("4.1.01.tiff")
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def filtro_high_boost(imagem, ksize=(5, 5), sigma=1.0, A=1.3):
    img_float = imagem.astype(np.float32)
    
    passa_baixa = cv2.GaussianBlur(img_float, ksize, sigma)
    
    img_high_boost = (A * img_float) - passa_baixa
    
    img_high_boost = np.clip(img_high_boost, 0, 255).astype(np.uint8)
    
    return img_high_boost

passa_baixa_puro = cv2.GaussianBlur(grayImg.astype(np.float32), (5, 5), 1.0)
passa_alta_puro = grayImg.astype(np.float32) - passa_baixa_puro

passa_alta_vis = np.clip(passa_alta_puro + 128, 0, 255).astype(np.uint8)

resultado_high_boost = filtro_high_boost(grayImg, ksize=(5, 5), sigma=1.0, A=1.3)
soma_float = resultado_high_boost.astype(np.float32) + passa_alta_puro
res = np.clip(soma_float, 0, 255).astype(np.uint8)

cv2.imshow("1 - Original em Cinza", grayImg)
cv2.imshow("2 - Filtro Passa-Alta", passa_alta_vis)
cv2.imshow("3 - Filtro High-Boost", resultado_high_boost)
cv2.imshow("4 - test Filtro High-Boost", res)

cv2.waitKey(0)
cv2.destroyAllWindows()