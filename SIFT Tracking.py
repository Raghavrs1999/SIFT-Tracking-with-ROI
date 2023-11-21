import cv2
import numpy as np

rect_start = None
rect_end = None
drawing = False
roi_selected = False

cap = cv2.VideoCapture(0)
cv2.namedWindow('Live Stream')

start_roi = 0

def draw_rectangle(event, x, y, flags, param):
    global rect_start, rect_end, drawing, roi_selected

    if event == cv2.EVENT_LBUTTONDOWN:
        rect_start = (x, y)
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        rect_end = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        rect_end = (x, y)
        drawing = False
        roi_selected = True


cv2.setMouseCallback('Live Stream', draw_rectangle)

sift = cv2.SIFT_create(10000)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if rect_start and rect_end:
        if not roi_selected:
            cv2.rectangle(frame, rect_start, rect_end, (0, 255, 0), 2)

        if roi_selected:
            start_roi += 1

            if start_roi == 1:
                x, y = min(rect_start[0], rect_end[0]), min(rect_start[1], rect_end[1])
                width, height = abs(rect_start[0] - rect_end[0]), abs(rect_start[1] - rect_end[1])
                x = max(0, x)
                y = max(0, y)
                width = min(width, frame.shape[1] - x)
                height = min(height, frame.shape[0] - y)

                roi = frame[y:y + height, x:x + width]

                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                kp, descriptors = sift.detectAndCompute(gray_roi, None)

                roi_with_keypoints = cv2.drawKeypoints(roi, kp, None, color=(0, 255, 0), flags=0)

                saved_kp = kp
                saved_descriptors = descriptors

                cv2.imshow('ROI with Keypoints', roi_with_keypoints)

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            kp_live, descriptors_live = sift.detectAndCompute(gray_frame, None)

            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)

            matches = flann.knnMatch(saved_descriptors, descriptors_live, k=2)

            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

            # If there are enough good matches, draw a rectangle around the matched region
            if len(good_matches) > 10:
                src_pts = np.float32([saved_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_live[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Compute the transformation matrix
                M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                # Get the corners of the ROI in the live stream frame
                h, w = gray_roi.shape
                corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

                # Transform the corners using the homography matrix
                transformed_corners = cv2.perspectiveTransform(corners, M)

                
                # Draw a rectangle around the transformed corners
                frame = cv2.polylines(frame, [np.int32(transformed_corners)], True, (0, 255, 0), 2)

            matched_frame = frame.copy()

            #matched_frame = cv2.drawMatches(roi, saved_kp, frame, kp_live, good_matches, matched_frame,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            #cv2.imshow('Live Stream with Matches', matched_frame)

    cv2.imshow('Live Stream', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
