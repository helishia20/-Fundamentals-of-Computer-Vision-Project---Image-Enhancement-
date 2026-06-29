import numpy as np

def HistogramEqualize(img):
    """
    img : تصویر خاکستری با مقادیر 0 تا 255
    خروجی: تصویر پس از Histogram Equalization
    """

    # ابعاد تصویر
    h, w = img.shape
    total_pixels = h * w

    # محاسبه هیستوگرام
    hist = np.zeros(256, dtype=np.int32)

    for i in range(h):
        for j in range(w):
            hist[img[i, j]] += 1

    # محاسبه CDF (تابع توزیع تجمعی)
    cdf = np.zeros(256, dtype=np.int32)
    cdf[0] = hist[0]

    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + hist[i]

    # پیدا کردن اولین مقدار غیر صفر CDF
    cdf_min = 0
    for value in cdf:
        if value != 0:
            cdf_min = value
            break

    # ساخت جدول نگاشت (LUT)
    lut = np.zeros(256, dtype=np.uint8)

    for i in range(256):
        lut[i] = round(
            ((cdf[i] - cdf_min) / (total_pixels - cdf_min)) * 255
        ) if cdf[i] > 0 else 0

    # اعمال نگاشت روی تصویر
    result = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            result[i, j] = lut[img[i, j]]

    return result