import cv2

def draw_signal_status(frame, green_lane, phase="green"):
    """
    Draw traffic signal lights (Red/Yellow/Green) for lanes A, B, C, D on the video frame.
    Only one lane can be green/yellow, others remain red.
    """

    # Fixed positions for signal lights on screen
    positions = {
        "A": (50, 50),
        "B": (200, 50),
        "C": (350, 50),
        "D": (500, 50),
    }

    for lane, pos in positions.items():
        if lane == green_lane:
            if phase == "green":
                color = (0, 255, 0)      # Green
            elif phase == "yellow":
                color = (0, 255, 255)    # Yellow
            else:
                color = (0, 0, 255)      # Red (rare case: all red)
        else:
            color = (0, 0, 255)          # Red for non-active lanes

        # Draw signal light
        cv2.circle(frame, pos, 20, color, -1)

        # Draw lane label below each light
        cv2.putText(frame, lane, (pos[0] - 15, pos[1] + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Overlay status text at bottom
    status_text = f"Current Lane: {green_lane} | Phase: {phase.upper()}"
    height = frame.shape[0]
    cv2.putText(frame, status_text, (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return frame
