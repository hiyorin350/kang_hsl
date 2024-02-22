import numpy as np
import cv2
from conversions import *

def angle_to_normal_vector(angle):
    """
    2次元空間において、指定された角度での直線の法線ベクトルを計算する。
    
    :param angle: 直線がX軸と成す角度（度単位）
    :return: 法線ベクトル（numpy配列）
    """
    # 角度をラジアンに変換
    angle_rad = np.radians(angle)
    
    # 直線の法線ベクトルの計算
    # 直線が成す角度に90度を加える（垂直な方向）
    nx = np.cos(angle_rad + np.pi / 2)
    ny = np.sin(angle_rad + np.pi / 2)
    
    return np.array([0, nx, ny])

def project_pixels_to_color_plane(image, u):
    """
    射影された画像を返す関数。

    :param image: 入力画像（CIE L*a*b* 色空間）
    :param u: 色平面の法線ベクトル
    :return: 射影された画像
    """
    # 画像の形状を取得
    height, width, _ = image.shape

    # 射影された画像を格納するための配列を初期化
    projected_image = np.zeros_like(image)

    # 各画素に対して射影を行う
    for i in range(height):
        for j in range(width):
            # 画素の色ベクトルを取得
            color_vector = image[i, j, :]

            # 色ベクトルを色平面に射影
            projected_vector = color_vector - np.dot(color_vector, u) * u

            # 射影された色ベクトルを保存
            projected_image[i, j, :] = projected_vector

    return projected_image

def clip_image_within_rgb_gamut(image_lab):
    """
    画像全体のL*a*b*色空間の色をRGB色空間に収まるように調整する。
    この際、L（明度）とC（彩度）のみを調整し、色相（H）は保持する。

    :param image_lab: L*a*b*色空間で表された入力画像
    :return: 調整されたL*a*b*画像
    """
    # 彩度Cの計算
    a = image_lab[:, :, 1].astype(np.float32)
    b = image_lab[:, :, 2].astype(np.float32)
    C = np.sqrt(a**2 + b**2) 
    rgb = lab_to_rgb(lab_image)
    
    # RGBが[0, 255]の範囲内に収まるか確認し、収まらない場合は彩度を調整
    mask = np.logical_or(rgb < 0, rgb > 1)
    while np.any(mask):
        # 彩度Cを減少させる
        C *= 0.99
        image_lab[:, :, 1] = C / np.maximum(C, np.finfo(float).eps) * a
        image_lab[:, :, 2] = C / np.maximum(C, np.finfo(float).eps) * b
        
        # RGBに再変換して確認
        
        rgb = lab_to_rgb(lab_image)
        mask = np.logical_or(rgb < 0., rgb > 1.1)
    
    return image_lab

# 画像の読み込み
image = cv2.imread('/Users/hiyori/kang/images/Chart26_kang_rotate.ppm')
lab_image = rgb_to_lab(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

height, width, _ = image.shape
N = height * width

lab_process = np.zeros_like(image)
lab_uncripped = np.zeros_like(image)
lab_cripped = np.zeros_like(image)
lab_out = np.zeros_like(image)

u = angle_to_normal_vector(90 + 11.8)#2色覚平面

rotate_image = project_pixels_to_color_plane(lab_image, u)

lab_cripped = clip_image_within_rgb_gamut(rotate_image)

img_out_rgb = lab_to_rgb(lab_cripped)

if img_out_rgb.dtype == np.float32 or img_out_rgb.dtype == np.float64:
    # 最大値が1.0を超えない場合、255を掛ける
    if img_out_rgb.max() <= 1.0:
        img_out_rgb = (img_out_rgb * 255).astype(np.uint8)

# 射影された画像を表示
print("done!")

img_out_rgb = (img_out_rgb * 255).astype(np.uint8)

img_out_bgr = cv2.cvtColor(img_out_rgb, cv2.COLOR_RGB2BGR)

cv2.imwrite('/Users/hiyori/kang/images/Chart26_kang_rotate.ppm',img_out_bgr)
cv2.imshow('lab_cripped', img_out_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
