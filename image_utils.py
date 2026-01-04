from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(path):
    """
    טעינת תמונה מהנתיב והמרתה למערך numpy בגווני אפור.
    """
    img = Image.open(path).convert('L') # 'L' ממיר ישירות לגווני אפור
    return np.array(img)

def edge_detection(image):
    """
    זיהוי קצוות באמצעות אופרטור Sobel ופעולת קונבולוציה.
    """
    # הגדרת קרנלים של Sobel לציר X ולציר Y
    # אלו מטריצות שמזהות שינויים בחדות (נגזרות)
    Kx = np.array([[-1, 0, 1], 
                   [-2, 0, 2], 
                   [-1, 0, 1]])
    
    Ky = np.array([[ 1,  2,  1], 
                   [ 0,  0,  0], 
                   [-1, -2, -1]])

    # ביצוע קונבולוציה למציאת שינויים אופקיים ואנכיים
    Gx = convolve2d(image, Kx, mode='same', boundary='fill', fillvalue=0)
    Gy = convolve2d(image, Ky, mode='same', boundary='fill', fillvalue=0)

    # חישוב עוצמת הגרדיאנט (שילוב של שני הצירים)
    # $G = \sqrt{Gx^2 + Gy^2}$
    g_magnitude = np.sqrt(Gx**2 + Gy**2)

    # נרמול הערכים לטווח של 0-255 והמרה ל-uint8
    g_magnitude = (g_magnitude / g_magnitude.max()) * 255
    return g_magnitude.astype(np.uint8)
