#Пункт 1
import cv2
import numpy as np

image = cv2.imread('F.png')
def rgb_to_cmyk(r, g, b):
    if (r == 0) and (g == 0) and (b == 0):
        return 0, 0, 0, 1
    r_prime, g_prime, b_prime = r / 255.0, g / 255.0, b / 255.0
    k = 1 - max(r_prime, g_prime, b_prime)
    c = (1 - r_prime - k) / (1 - k)
    m = (1 - g_prime - k) / (1 - k)
    y = (1 - b_prime - k) / (1 - k)
    return round(c * 255), round(m * 255), round(y * 255), round(k * 255)

def rgb_to_hsv(r, g, b):
    rgb_pixel = np.uint8([[[b, g, r]]])
    hsv_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_BGR2HSV)
    h, s, v = hsv_pixel[0][0]
    return h, s, v

def mouse_callback(event, x, y, flags, param):
    global windows_opened

    if event == cv2.EVENT_LBUTTONDOWN:
        bgr_color = image[y, x]
        b, g, r = int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2])

        c, m, y1, k = rgb_to_cmyk(r, g, b)
        h, s, v = rgb_to_hsv(r, g, b)

        print(
            f"Координаты: ({x}, {y}) | RGB: ({r}, {g}, {b}) | CMYK: ({c}, {m}, {y1}, {k}) | HSV: ({h}, {s}, {v})"
        )

        cmyk_image[:, :] = [k, y1, m]
        hsv_image[:, :] = [h, s, v]

        if not windows_opened:
            cv2.imshow("CMYK Image", cmyk_image)
            cv2.imshow("HSV Image", hsv_image)
            windows_opened = True
        else:
            cv2.imshow("CMYK Image", cmyk_image)
            cv2.imshow("HSV Image", hsv_image)

height, width = image.shape[:2]

cmyk_image = np.zeros((height, width, 3), dtype=np.uint8)
hsv_image = np.zeros((height, width, 3), dtype=np.uint8)

windows_opened = False

cv2.imshow("Image", image)
cv2.setMouseCallback("Image", mouse_callback)

cv2.waitKey(0)
cv2.destroyAllWindows()



#
# #Пункт 2 + задание из скриншота
# import cv2
#
# image = cv2.imread('F.png')
# height, width = image.shape[:2]
# center = (width // 2, height // 2)
# rotation_matrix = cv2.getRotationMatrix2D(center, 75, 1.0)
# image = cv2.warpAffine(image, rotation_matrix, (width, height))
#
# image = cv2.resize(image, (int(width * 0.75), int(height * 0.75)))
# new_height, new_width = image.shape[:2]
#
# cv2.line(image, (new_width // 2, 0), (new_width // 2, new_height), (255, 0, 0), 2)
#
# text = "%#?!!a1T"
# font = cv2.FONT_HERSHEY_SIMPLEX
# font_scale = 1
# color = (255, 0, 0)
# thickness = 2
# position = (100, new_height - 10)
# cv2.putText(image, text, position, font, font_scale, color, thickness)
#
# cv2.imshow("Result", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
#
#
#
# ####Пункт 3
# import cv2
#
# image_path = "F.png"
# image = cv2.imread(image_path)
#
# def apply_positive_filter(img, alpha=1.5, beta=50):
#     return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
#
# def apply_negative_filter(img):
#     return cv2.bitwise_not(img)
#
# positive_filtered_image = apply_positive_filter(image)
# negative_filtered_image = apply_negative_filter(image)
#
# cv2.imshow("Original Image", image)
# cv2.imshow("Positive Filter Image", positive_filtered_image)
# cv2.imshow("Negative Filter Image", negative_filtered_image)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()


