import csv
import cv2
import numpy as np

img = cv2.imread('page_1.png')
h_channel = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 0]
s_channel = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
v_channel = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]

np.savetxt('h_value.csv', h_channel,delimiter= ',',fmt='%d')
np.savetxt('s_value.csv', s_channel,delimiter= ',',fmt='%d')
np.savetxt('v_value.csv', v_channel,delimiter= ',',fmt='%d')
print("HSVの各チャンネルの値をCSVファイルに保存しました。")