import cv2
import numpy as np

base = cv2.imread("base.jpeg")
test = cv2.imread("1.jpeg")

base_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

imgSub = cv2.subtract(base_gray, test_gray)

#np.set_printoptions(threshold=np.inf)

#print(imgSub)

limiar_empirico = 38
_, img_binaria = cv2.threshold(imgSub, limiar_empirico, 255, cv2.THRESH_BINARY)

pixels_corpo = np.where(img_binaria == 255)
coordenadas_y = pixels_corpo[0]
coordenadas_x = pixels_corpo[1]

img_resultado = cv2.cvtColor(imgSub, cv2.COLOR_GRAY2BGR)

if len(coordenadas_x) > 0 and len(coordenadas_y) > 0:
    x_min, x_max = np.min(coordenadas_x), np.max(coordenadas_x)
    y_min, y_max = np.min(coordenadas_y), np.max(coordenadas_y)
    
    cv2.rectangle(img_resultado, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
    
    print(f"Corpo detectado nas coordenadas: X=[{x_min} a {x_max}], Y=[{y_min} a {y_max}]")
else:
    print("Nenhum corpo detectado com o limiar atual.")

cv2.imshow("1. Diferenca Absoluta", imgSub)
cv2.imshow("1. base", base_gray)
cv2.imshow("1. test", test_gray)
cv2.imshow("3. Resultado com Retângulo", img_resultado)

cv2.waitKey(0)
cv2.destroyAllWindows()