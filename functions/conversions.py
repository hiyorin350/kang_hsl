import numpy as np
import cv2

def rgb_to_hsl(image):
    # imageは(高さ, 幅, 3)の形状のNumPy配列と仮定
    # dtypeをfloatに変換して計算を行う
    image = image.astype(np.float32) / 255.0

    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    max_color = np.maximum(np.maximum(R, G), B)
    min_color = np.minimum(np.minimum(R, G), B)

    L = (max_color + min_color) / 2

    delta = max_color - min_color
    S = np.zeros_like(L)
    
    # 彩度の計算
    S[delta != 0] = delta[delta != 0] / (1 - np.abs(2 * L[delta != 0] - 1))

    H = np.zeros_like(L)
    # 色相の計算
    # Rが最大値
    idx = (max_color == R) & (delta != 0)
    H[idx] = 60 * (((G[idx] - B[idx]) / delta[idx]) % 6)

    # Gが最大値
    idx = (max_color == G) & (delta != 0)
    H[idx] = 60 * (((B[idx] - R[idx]) / delta[idx]) + 2)

    # Bが最大値
    idx = (max_color == B) & (delta != 0)
    H[idx] = 60 * (((R[idx] - G[idx]) / delta[idx]) + 4)

    # 彩度と輝度をパーセンテージに変換
    S = S * 100
    L = L * 100

    return np.stack([H, S, L], axis=-1)

def hsl_to_rgb(hsl_image):
    H, S, L = hsl_image[:, :, 0], hsl_image[:, :, 1], hsl_image[:, :, 2]
    H /= 360  # Hを0から1の範囲に正規化
    S /= 100  # Sを0から1の範囲に正規化
    L /= 100  # Lを0から1の範囲に正規化

    def hue_to_rgb(p, q, t):
        # tが0より小さい場合、1を加算
        t[t < 0] += 1
        # tが1より大きい場合、1を減算
        t[t > 1] -= 1
        # t < 1/6の場合
        r = np.copy(p)
        r[t < 1/6] = p[t < 1/6] + (q[t < 1/6] - p[t < 1/6]) * 6 * t[t < 1/6]
        # 1/6 <= t < 1/2の場合
        r[(t >= 1/6) & (t < 1/2)] = q[(t >= 1/6) & (t < 1/2)]
        # 1/2 <= t < 2/3の場合
        r[(t >= 1/2) & (t < 2/3)] = p[(t >= 1/2) & (t < 2/3)] + (q[(t >= 1/2) & (t < 2/3)] - p[(t >= 1/2) & (t < 2/3)]) * (2/3 - t[(t >= 1/2) & (t < 2/3)]) * 6
        # t >= 2/3の場合、rは変更なし（pの値を保持）
        
        return r

    rgb_image = np.zeros_like(hsl_image)
    q = np.where(L < 0.5, L * (1 + S), L + S - L * S)
    p = 2 * L - q

    rgb_image[:, :, 0] = hue_to_rgb(p, q, H + 1/3)  # R
    rgb_image[:, :, 1] = hue_to_rgb(p, q, H)        # G
    rgb_image[:, :, 2] = hue_to_rgb(p, q, H - 1/3)  # B

    return np.clip(rgb_image * 255, 0, 255).astype(np.uint8)

def hsl_to_cartesian(h, s, l):
    """
    HSL値を直交座標系に変換する。
    :param h: 色相 (0-360度)
    :param s: 彩度 (0-1)
    :param l: 輝度 (0-1)
    :return: (x, y, z) 直交座標系における座標
    """
    # 色相をラジアンに変換
    h_rad = np.deg2rad(h)
    
    # 彩度を半径としてx, y座標を計算
    x = s * np.cos(h_rad)
    y = s * np.sin(h_rad)
    
    # 輝度はz座標としてそのまま使用
    z = l
    
    return x, y, z