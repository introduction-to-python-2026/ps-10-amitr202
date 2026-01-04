from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(path):
    """טעינת תמונה והמרתה למערך numpy בגווני אפור"""
    img = Image.open(path).convert('L')
    return np.array(img)

def ball(radius):
    """יצירת אלמנט מבני עבור פילטר חציון"""
    from skimage.morphology import ball as sk_ball
    return sk_ball(radius)

def median(image, footprint):
    """הפעלת פילטר חציון להפחתת רעשים"""
    from skimage.filters.rank import median as sk_median
    return sk_median(image, footprint)

def edge_detection(image):
    """זיהוי קצוות באמצעות אופרטור Sobel ונרמול ל-0-255"""
    # הגדרת מטריצות Sobel
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    # חישוב קונבולוציה
    Gx = convolve2d(image, Kx, mode='same', boundary='fill', fillvalue=0)
    Gy = convolve2d(image, Ky, mode='same', boundary='fill', fillvalue=0)
    
    # חישוב עוצמת הקצוות
    g_magnitude = np.sqrt(Gx**2 + Gy**2)
    
    # נרמול לטווח 0-255 - קריטי כדי לעבור את הסף (edge > 50)
    if g_magnitude.max() > 0:
        g_magnitude = (g_magnitude / g_magnitude.max()) * 255
        
    return g_magnitude.astype(np.uint8)
