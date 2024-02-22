import numpy as np
def rgb_to_lab(rgb):
    """
    RGBからL*a*b*色空間への変換を行う。
    rgbは[0, 255]の範囲の値を持つ3要素のリストまたはNumPy配列。
    """
    # RGBを[0, 1]の範囲に正規化
    rgb = rgb / 255.0
    
    # sRGBからリニアRGBへの変換
    def gamma_correction(channel):
        return np.where(channel > 0.04045, ((channel + 0.055) / 1.055) ** 2.4, channel / 12.92)
    
    rgb_linear = gamma_correction(rgb)
    
    # リニアRGBからXYZへの変換
    mat_rgb_to_xyz = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    xyz = np.dot(mat_rgb_to_xyz, rgb_linear)
    
    # XYZからL*a*b*への変換
    def xyz_to_lab(t):
        delta = 6/29
        return np.where(t > delta ** 3, t ** (1/3), t / (3 * delta ** 2) + 4/29)
    
    xyz_ref_white = np.array([0.95047, 1.00000, 1.08883])
    xyz_normalized = xyz / xyz_ref_white
    
    L = 116 * xyz_to_lab(xyz_normalized[1]) - 16
    a = 500 * (xyz_to_lab(xyz_normalized[0]) - xyz_to_lab(xyz_normalized[1]))
    b = 200 * (xyz_to_lab(xyz_normalized[1]) - xyz_to_lab(xyz_normalized[2]))
    
    return np.array([L, a, b])

def lab_to_rgb(lab):
    """
    L*a*b*からRGB色空間への逆変換を行う。
    labはL, a, bの値を持つ3要素のリストまたはNumPy配列。
    """
    # L*a*b*からXYZへの変換
    def lab_to_xyz(l, a, b):
        y = (l + 16) / 116
        x = a / 500 + y
        z = y - b / 200

        xyz = np.array([x, y, z])
        mask = xyz > 6/29
        xyz[mask] = xyz[mask] ** 3
        xyz[~mask] = (xyz[~mask] - 16/116) / 7.787

        # D65光源の参照白
        xyz_ref_white = np.array([0.95047, 1.00000, 1.08883])
        xyz = xyz * xyz_ref_white
        return xyz

    xyz = lab_to_xyz(*lab)

    # XYZからリニアRGBへの変換
    mat_xyz_to_rgb = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ])
    rgb_linear = np.dot(mat_xyz_to_rgb, xyz)

    # リニアRGBからsRGBへのガンマ補正
    def gamma_correction(channel):
        return np.where(channel > 0.0031308, 1.055 * (channel ** (1/2.4)) - 0.055, 12.92 * channel)

    rgb = gamma_correction(rgb_linear)

    # RGB値を[0, 255]の範囲にクリッピングして整数に変換
    rgb_clipped = np.clip(rgb * 255, 0, 255).astype(np.uint8)

    return rgb_clipped