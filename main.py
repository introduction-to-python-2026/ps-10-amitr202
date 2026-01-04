from image_utils import load_image, edge_detection
from PIL import Image

def main():
    # 1. טעינת התמונה המקורית
    original_path = 'my_image.jpg' # וודא שיש לך תמונה בשם הזה
    img_array = load_image(original_path)
    
    # 2. הרצת פונקציית זיהוי הקצוות
    edges_array = edge_detection(img_array)
    
    # 3. שמירת התוצאה כתמונה
    # הפיכת המערך בחזרה לאובייקט תמונה של PIL
    edge_image = Image.fromarray(edges_array)
    edge_image.save('edge_detected.png')
    
    print("Edge detection complete. Image saved as edge_detected.png")

if __name__ == "__main__":
    main()
