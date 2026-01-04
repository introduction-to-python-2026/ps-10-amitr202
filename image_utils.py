from PIL import Image
import numpy as np
from scipy.signal import convolve2d
import skimage.filters
import skimage.morphology

def load_image(path):
    # טעינה והמרה למערך float כדי למנוע שגיאות עיגול
    return np.array(Image.open(path).convert('L')).astype(float)

def ball(radius):
    return skimage.morphology.ball(radius)

def median(image, footprint):
    # שימוש ב-uint8 עבור פילטר החציון כפי ש-skimage דורשת
    return skimage.filters.median(image.astype(np.uint8), footprint)

def edge_detection(image):
    # מטריצות Sobel
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    # קונבולוציה מדויקת
    Gx = convolve2d(image, Kx, mode='same', boundary='symm')
    Gy = convolve2d(image, Ky, mode='same', boundary='symm')
    
    # חישוב עוצמה ונרמול
    mag = np.sqrt(Gx**2 + Gy**2)
    if mag.max() > 0:
        mag = (mag / mag.max()) * 255
        
    return mag.astype(np.uint8)
