from app import TrackingApp

if __name__ == "__main__":
    print("=== Pose Detection System ===")
    print("Using pose model for accurate person detection with keypoints")
    print()
    
    print("=== Tracking Mode Selection ===")
    print("1. Head tracking - Track head position for precise targeting")
    print("2. Body tracking - Track center of body")
    print()
    
    mode_choice = input("Enter tracking mode (1 or 2): ").strip()
    
    if mode_choice == "1":
        ui_mode = "head"
        print("Using head tracking mode")
    else:
        ui_mode = "body"
        print("Using body tracking mode")
    
    print(f"\nStarting pose detection application with {ui_mode} tracking...")
    TrackingApp(ui_mode=ui_mode).run()