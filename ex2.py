import cv2
import numpy as np

base = cv2.imread("baseY.jpeg")
test = cv2.imread("testY.jpeg")

base_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

imgSub = base_gray - test_gray

#np.set_printoptions(threshold=np.inf)

#print(imgSub)

limiar_empirico = 200
img_binaria = np.where(imgSub < limiar_empirico, 255, 0).astype(np.uint8)

img_binaria_limpa = cv2.medianBlur(img_binaria, 9)

pixels_corpo = np.where(img_binaria_limpa == 255)
coordenadas_y = pixels_corpo[0]
coordenadas_x = pixels_corpo[1]

if len(coordenadas_x) > 0 and len(coordenadas_y) > 0:
    x_min, x_max = np.min(coordenadas_x), np.max(coordenadas_x)
    y_min, y_max = np.min(coordenadas_y), np.max(coordenadas_y)
    
    cv2.rectangle(test, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
    print(f"Corpo detectado: X=[{x_min} a {x_max}], Y=[{y_min} a {y_max}]")
else:
    print("Nenhum corpo detectado.")

cv2.imshow("1. Diferenca Absoluta", imgSub)
cv2.imshow("1. base", base_gray)
cv2.imshow("1. test", test_gray)
cv2.imshow("3. Resultado com Retângulo", img_binaria)
cv2.imshow("4. Resultado com Retângulo", test)

cv2.waitKey(0)
cv2.destroyAllWindows()