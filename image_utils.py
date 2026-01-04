from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(path):
    """
    טעינת תמונה והמרתה למערך numpy בגווני אפור
    """
    img = Image.open(path).convert('L')
    return np.array(img)

def edge_detection(image):
    """
    זיהוי קצוות באמצעות אופרטור Sobel
    """
    # הגדרת מטריצות Sobel לזיהוי שינויים בציר X ובציר Y
    Kx = np.array([[-1, 0, 1], 
                   [-2, 0, 2], 
                   [-1, 0, 1]])
    
    Ky = np.array([[ 1,  2,  1], 
                   [ 0,  0,  0], 
                   [-1, -2, -1]])

    # ביצוע קונבולוציה
    Gx = convolve2d(image, Kx, mode='same', boundary='fill', fillvalue=0)
    Gy = convolve2d(image, Ky, mode='same', boundary='fill', fillvalue=0)

    # חישוב עוצמת הקצוות (מגניטודה)
    g_magnitude = np.sqrt(Gx**2 + Gy**2)

    # נרמול הערכים לטווח 0-255 כדי שנוכל להציג כתמונה
    if g_magnitude.max() > 0:
        g_magnitude = (g_magnitude / g_magnitude.max()) * 255
        
    return g_magnitude.astype(np.uint8)
