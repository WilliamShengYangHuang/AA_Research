import cv2
import numpy as np

# set game board (grid) size
chessboard_size = (8, 8)

# Set the corner detection criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Define a function to find and draw chessboard corners
def find_chessboard_corners(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 使用简单的二值化处理替代 adaptive thresholding
    _, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    
    if ret:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        frame = cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)
        
        # Get the four corners (top-left, top-right, bottom-right, bottom-left)
        top_left = (int(corners[0][0][0]), int(corners[0][0][1]))
        top_right = (int(corners[chessboard_size[0] - 1][0][0]), int(corners[chessboard_size[0] - 1][0][1]))
        bottom_right = (int(corners[-1][0][0]), int(corners[-1][0][1]))
        bottom_left = (int(corners[-chessboard_size[0]][0][0]), int(corners[-chessboard_size[0]][0][1]))
        
        vertices = [top_left, top_right, bottom_right, bottom_left]
        
        # Draw the vertices on the image
        for i, vertex in enumerate(vertices):
            cv2.circle(frame, vertex, 5, (0, 0, 255), -1)
            cv2.putText(frame, f'V{i+1}', vertex, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame, True
    else:
        return frame, False

# launch webcam
cap = cv2.VideoCapture(0)

# Set the webcam frame rate
cap.set(cv2.CAP_PROP_FPS, 15)

# Set the webcam resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Processing interval (in frames)
frame_interval = 1
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count % frame_interval == 0:
        # 查找和绘制棋盘角点
        frame, detected = find_chessboard_corners(frame)
        
        # If未检测到棋盘，则显示提示信息
        if not detected:
            cv2.putText(frame, "No chessboard detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Display the result
    cv2.imshow('Chessboard Detection', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
