import cv2 
import matplotlib.pyplot as plt
import numpy as np
bgr_image = cv2.imread('page_1.png')

hsv_image = cv2.cvtColor(bgr_image,cv2.COLOR_BGR2HSV)
h,s,v = hsv_image[:,:,0], hsv_image[:,:,1], hsv_image[:,:,2]
hist_h = cv2.calcHist([h],[0],None,[256],[0,256])
hist_s = cv2.calcHist([s],[0],None,[256],[0,256])
hist_v = cv2.calcHist([v],[0],None,[256],[0,256])
plt.plot(hist_h, color='r', label='H')
plt.plot(hist_s, color='g', label='S')
plt.plot(hist_v, color='b', label='V')
plt.legend()
plt.show()
# # HSVの個別チャンネルをプロット
# plt.figure(figsize=(10, 6))
# plt.subplot(1, 3, 1)
# plt.imshow(h, cmap='hsv')  # Hue
# plt.title("Hue")
# plt.subplot(1, 3, 2)
# plt.imshow(s, cmap='gray')  # Saturation
# plt.title("Saturation")
# plt.subplot(1, 3, 3)
# plt.imshow(v, cmap='gray')  # Value
# plt.title("Value")
# plt.tight_layout()
# plt.show()

#大津の手法
def threshold_otsu(hsv,min_value=0,max_value=255):
    hist = [np.sum(hsv == i) for i in range(256)]
    s_max = (0,-10)

    for th in range(256):
        # Count the sum of pixels in the two classes
        n1 = sum(hist[:th])
        n2 = sum(hist[th:])

        # Calculate the means of the two classes
        if n1 == 0 :
            mu1 = 0
        else: 
            mu1 = sum([i * hist[i] for i in range(0,th)]) / n1

        if n2 == 0 :
            mu2 = 0
        else: 
            mu2 = sum([i * hist[i] for i in range(th,150)]) / n2

        # Calcurlate the between-class variance
        s = n1 * n2 * (mu1 - mu2) ** 2

        # update the between - class variance and threshold if it is greater than the previous maximum
        if s > s_max[1]:
            s_max = (th,s)

    t = s_max[0]
    hsv = hsv.copy()
    hsv[hsv < t] = min_value
    hsv[hsv >= t] = max_value

    return hsv

def threshold_manual(img,t,min_value=0,max_value=255):
    img = img.copy()
    img[img < t] = min_value
    img[img >= t] = max_value
    return img
# s_bin = threshold_manual(s, 50)
# plt.figure(figsize=(10, 6))
# plt.imshow(s_bin, cmap='gray')
# plt.title("s(manual)")
# plt.show()

# h_otsu = threshold_otsu(h)
# s_otsu = threshold_otsu(s)
# v_otsu = threshold_otsu(v)
# plt.figure(figsize=(10, 6))
# plt.subplot(1, 1, 1)
# plt.imshow(v_otsu, cmap='gray')
# plt.title("v (Otsu)")

# plt.subplot(1, 2, 2)
# plt.imshow(s_otsu, cmap='gray')
# plt.title("S (Otsu)")

# plt.tight_layout()
# plt.show()


