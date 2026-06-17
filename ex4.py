from time import perf_counter
import cv2
import numpy as np
import scipy.signal

img = cv2.imread("4.1.01.tiff")
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#template = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
tamanho = 15
template = np.ones((tamanho, tamanho), dtype=np.float32)
template = template / (tamanho * tamanho)

conv_time = perf_counter()

convImg = scipy.signal.convolve2d(grayImg, template, mode="same", boundary="fill", fillvalue=0)

convImg = np.clip(convImg, 0, 255).astype(np.uint8)

conv_time = perf_counter() - conv_time
print(f"Tempo de convolução: {conv_time:.4f} segundos")

fourier_time = perf_counter()

templateFourier = np.fft.fft2(template, s=grayImg.shape)
imageFourier = np.fft.fft2(grayImg)

convFourier = templateFourier * imageFourier
convImgFourier = np.fft.ifft2(convFourier)

convImgFourier = np.clip(convImgFourier, 0, 255).astype(np.uint8)

fourier_time = perf_counter() - fourier_time
print(f"Tempo de convolução em Fourier: {fourier_time:.4f} segundos")

cv2.imshow("1 - Original em Cinza", grayImg)
cv2.imshow("2 - Convoluida", convImg)
cv2.imshow("3 - Fourier", convImgFourier)
cv2.waitKey(0)
cv2.destroyAllWindows()