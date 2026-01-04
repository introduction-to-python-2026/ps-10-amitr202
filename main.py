from image_utils import load_image, edge_detection
from PIL import Image

def main():
    # שלב 1: הגדרת שם התמונה המקורית (וודאי שהתמונה נמצאת באותה תיקייה)
    # את יכולה לשנות את השם ל- 'input_image.jpg' אם זה השם שנתת לתמונה ששמרת
    input_filename = 'input_image.jpg' 
    output_filename = 'edge_detected_result.png'

    try:
        # שלב 2: טעינת התמונה
        print(f"Loading image: {input_filename}...")
        img_array = load_image(input_filename)

        # שלב 3: הפעלת זיהוי הקצוות
        print("Processing edge detection...")
        edges = edge_detection(img_array)

        # שלב 4: שמירת התוצאה
        result_img = Image.fromarray(edges)
        result_img.save(output_filename)
        print(f"Success! The result was saved as: {output_filename}")
        
        # שלב 5: (אופציונלי) הצגת התמונה על המסך
        result_img.show()

    except FileNotFoundError:
        print(f"Error: The file '{input_filename}' was not found. Please make sure the image is in the folder.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
