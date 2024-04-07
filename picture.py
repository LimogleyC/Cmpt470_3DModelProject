import cv2
import os

def capture_screenshot_on_spacebar():
    cam = cv2.VideoCapture(0)  # Initialize the camera
    cv2.namedWindow("Camera Feed")  # Create a named window for display
    img_counter = 1  # Counter for image filenames
    output_folder = "checkerboard"  # Folder to save grayscale images
    # output_folder = "chest"
    # output_folder = "rock"
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    while True:
        ret, frame = cam.read()  # Read a frame from the camera
        if not ret:
            print("Failed to grab frame. Exiting.")
            break

        cv2.imshow("Camera Feed", frame)  # Display the frame

        key = cv2.waitKey(1)  # Wait for a key press
        if key == 27:  # If ESC key is pressed, exit the loop
            print("Escape key pressed. Closing...")
            break
        elif key == 32:  # If SPACE key is pressed, capture and save the image
            img_name = f"image_{img_counter}.jpg"
            # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            cv2.imwrite(os.path.join(output_folder, img_name), frame)
            print(f"{img_name} saved in {output_folder}!")
            img_counter += 1

    cam.release()  # Release the camera
    cv2.destroyAllWindows()  # Close all windows

if __name__ == "__main__":
    capture_screenshot_on_spacebar()