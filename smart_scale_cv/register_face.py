import cv2
import os

def register_user():
    # Ensure the storage directory exists
    output_dir = "known_users"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get the user's name
    print("=== Smart Scale Face Registration ===")
    user_name = input("Enter the name of the person to register: ").strip()
    if not user_name:
        print("❌ Name cannot be empty. Exiting.")
        return

    filename = os.path.join(output_dir, f"{user_name}.jpg")

    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not access the webcam.")
        return

    print("\n📸 Camera opening...")
    print("--> Position your face clearly in the frame.")
    print("--> Press [SPACEBAR] to capture your portrait.")
    print("--> Press [ESC] to cancel.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame from camera.")
            break

        # Show the live feed in a window
        cv2.imshow("Register Face - Press Space to Capture", frame)

        # Wait for keypresses
        key = cv2.waitKey(1) & 0xFF
        
        # Spacebar pressed
        if key == 32:
            # Save the raw frame as your profile picture
            cv2.imwrite(filename, frame)
            print(f"\n🎯 Success! Saved profile for '{user_name}' to {filename}")
            break
        
        # ESC key pressed
        elif key == 27:
            print("\n❌ Registration cancelled.")
            break

    # Clean up window and camera handle
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    register_user()
