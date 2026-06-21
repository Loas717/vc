import cv2
import numpy as np

img = cv2.imread("extest.jpg")

grayIm = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

altura, largura = grayIm.shape
img64tons = np.zeros((altura, largura), dtype=np.uint8)

img64tons = (grayIm // 4) * 4
tons_unicos_original = np.unique(grayIm)
tons_unicos_final = np.unique(img64tons)

print(f"Imagem Original: {len(tons_unicos_original)} tons de cinza diferentes.")
print(f"Imagem Final:    {len(tons_unicos_final)} tons de cinza diferentes.")

cv2.imshow('Original Cinza (256 tons)', grayIm)
cv2.imshow('Quantizada (Max 64 tons)', img64tons)
cv2.waitKey(0)
cv2.destroyAllWindows()