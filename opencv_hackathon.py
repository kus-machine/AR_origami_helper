import cv2
import numpy as np


def step_1(img, pts_src, number_of_good_frames):
    pts_dst = np.array([[210, 0], [210, 297], [0, 297], [0, 0]])
    array_of_line_length = np.zeros(4)
    for i in range(4):
        array_of_line_length[i] = ((pts_src[i, 0] - pts_src[(i + 1) % 4, 0]) ** 2) + (
                (pts_src[i, 1] - pts_src[(i + 1) % 4, 1]) ** 2)
    pts_src = np.roll(pts_src, -np.argmax(array_of_line_length), axis=0)
    h, status = cv2.findHomography(pts_src, pts_dst, method=cv2.RANSAC, confidence=0.9999)
    h_inv, status_inv = cv2.findHomography(pts_dst, pts_src, method=cv2.RANSAC, confidence=0.9999)
    points = np.array([[105, 0, 1], [105, 297, 1]], np.float64)
    if sum(status) != 4:
        return img, 0
    else:
        points[0] = (h_inv.astype(np.float64)).dot(points[0])
        points[1] = (h_inv.astype(np.float64)).dot(points[1])
        points[0] = points[0] / points[0, -1]
        points[1] = points[1] / points[1, -1]
        if number_of_good_frames > 4:
            img = cv2.line(img, (int(points[0, 0]), int(points[0, 1])), (int(points[1, 0]), int(points[1, 1])),
                           (0, 255, 255), 3)
        return img, number_of_good_frames + 1


def step_2(img, pts_src, number_of_good_frames):
    pts_dst = np.array([[105, 0], [105, 297], [0, 297], [0, 0]])
    array_of_line_length = np.zeros(4)
    for i in range(4):
        array_of_line_length[i] = ((pts_src[i, 0] - pts_src[(i + 1) % 4, 0]) ** 2) + (
                (pts_src[i, 1] - pts_src[(i + 1) % 4, 1]) ** 2)
    pts_src = np.roll(pts_src, -np.argmax(array_of_line_length), axis=0)
    h, status = cv2.findHomography(pts_src, pts_dst, method=cv2.RANSAC, confidence=0.9999)
    if (sum(status) != 4):
        return img, 0
    else:
        return img, number_of_good_frames + 1


def step_3(img, pts_src, number_of_good_frames):
    pts_dst = np.array([[210, 0], [210, 297], [0, 297], [0, 0]])
    array_of_line_length = np.zeros(4)
    for i in range(4):
        array_of_line_length[i] = ((pts_src[i, 0] - pts_src[(i + 1) % 4, 0]) ** 2) + (
                (pts_src[i, 1] - pts_src[(i + 1) % 4, 1]) ** 2)
    pts_src = np.roll(pts_src, -np.argmax(array_of_line_length), axis=0)
    h, status = cv2.findHomography(pts_src, pts_dst, method=cv2.RANSAC, confidence=0.9999)
    h_inv, status_inv = cv2.findHomography(pts_dst, pts_src, method=cv2.RANSAC, confidence=0.9999)
    points = np.array([[105, 0, 1], [0, 105, 1]], np.float64)
    if (sum(status) != 4):
        return img, 0
    else:
        if ((pts_src[0, 0]) ** 2 + (pts_src[0, 1]) ** 2) < ((pts_src[2, 0]) ** 2 + (pts_src[2, 1]) ** 2):
            points = np.array([[105, 0, 1], [0, 105, 1]], np.float64)
        else:
            points = np.array([[105, 297, 1], [105, 192, 1]], np.float64)

        points[0] = (h_inv.astype(np.float64)).dot(points[0])
        points[1] = (h_inv.astype(np.float64)).dot(points[1])
        points[0] = points[0] / points[0, -1]
        points[1] = points[1] / points[1, -1]
        if number_of_good_frames > 4:
            img = cv2.line(img, (int(points[0, 0]), int(points[0, 1])), (int(points[1, 0]), int(points[1, 1])),
                           (0, 255, 255), 3)
        return img, number_of_good_frames + 1


def step_4(img, pts_src, number_of_good_frames):
    pts_dst = np.array([[0, 210], [297, 210], [297, 0], [105, 0], [0, 105]])
    array_of_line_length = np.zeros(5)
    for i in range(5):
        array_of_line_length[i] = ((pts_src[i, 0] - pts_src[(i + 1) % 5, 0]) ** 2) + (
                (pts_src[i, 1] - pts_src[(i + 1) % 5, 1]) ** 2)
    pts_src = np.roll(pts_src, -np.argmax(array_of_line_length), axis=0)
    h, status = cv2.findHomography(pts_src[0:4], pts_dst[0:4], method=cv2.RANSAC, confidence=0.9)
    h_inv, status_inv = cv2.findHomography(pts_dst[0:4], pts_src[0:4], method=cv2.RANSAC, confidence=0.9)
    points = np.array([[105, 210, 1], [0, 105, 1]], np.float64)
    if (sum(status) != 4):
        return img, 0
    else:
        points[0] = (h_inv.astype(np.float64)).dot(points[0])
        points[1] = (h_inv.astype(np.float64)).dot(points[1])
        points[0] = points[0] / points[0, -1]
        points[1] = points[1] / points[1, -1]
        if number_of_good_frames > 4:
            img = cv2.line(img, (int(points[0, 0]), int(points[0, 1])), (int(points[1, 0]), int(points[1, 1])),
                           (0, 255, 255), 3)
        return img, number_of_good_frames + 1


def step_5(img, pts_src, number_of_good_frames):
    pts_dst = np.array([[0, 105], [105, 210], [297, 210], [297, 0], [105, 0]])
    for i in range(5):
        pts_src = np.roll(pts_src, i, axis=0)
        h, status = cv2.findHomography(pts_src, pts_dst, method=cv2.RANSAC, confidence=0.9999)
        h_inv, status_inv = cv2.findHomography(pts_dst, pts_src, method=cv2.RANSAC, confidence=0.9999)
        if (sum(status) == 5):
            points = np.array([[0, 105, 1], [253, 0, 1]], np.float64)
            points[0] = (h_inv.astype(np.float64)).dot(points[0])
            points[1] = (h_inv.astype(np.float64)).dot(points[1])
            points[0] = points[0] / points[0, -1]
            points[1] = points[1] / points[1, -1]
            if number_of_good_frames > 1:
                img = cv2.line(img, (int(points[0, 0]), int(points[0, 1])), (int(points[1, 0]), int(points[1, 1])),
                               (0, 255, 255), 3)
            return img, number_of_good_frames + 1
    return img, 0


def step_6(img, pts_src, number_of_good_frames):
    pts_dst = np.array([[297, 0], [253, 0], [0, 105], [105, 210], [297, 210]])
    array_of_line_length = np.zeros(5)
    for i in range(5):
        array_of_line_length[i] = ((pts_src[i, 0] - pts_src[(i + 1) % 5, 0]) ** 2) + (
                (pts_src[i, 1] - pts_src[(i + 1) % 5, 1]) ** 2)
    pts_src = np.roll(pts_src, -np.argmin(array_of_line_length), axis=0)
    h, status = cv2.findHomography(pts_src[0:4], pts_dst[0:4], method=cv2.RANSAC, confidence=0.9999)
    h_inv, status_inv = cv2.findHomography(pts_dst[0:4], pts_src[0:4], method=cv2.RANSAC, confidence=0.9999)
    points = np.array([[0, 105, 1], [253, 210, 1]], np.float64)
    if (sum(status) != 4):
        return img, 0
    else:
        points[0] = (h_inv.astype(np.float64)).dot(points[0])
        points[1] = (h_inv.astype(np.float64)).dot(points[1])
        points[0] = points[0] / points[0, -1]
        points[1] = points[1] / points[1, -1]
        if number_of_good_frames > 5:
            img = cv2.line(img, (int(points[0, 0]), int(points[0, 1])), (int(points[1, 0]), int(points[1, 1])),
                           (0, 255, 255), 3)
        return img, number_of_good_frames + 1


def step_7(img, pts_src, number_of_good_frames):
    pts_dst = np.array([[253, 0], [0, 105], [253, 210], [297, 210], [297, 0]])
    array_of_line_length = np.zeros(5)
    for i in range(5):
        array_of_line_length[i] = ((pts_src[i, 0] - pts_src[(i + 1) % 5, 0]) ** 2) + (
                (pts_src[i, 1] - pts_src[(i + 1) % 5, 1]) ** 2)
    first_max = np.argmax(array_of_line_length)
    array_of_line_length[first_max] = 0
    second_max = np.argmax(array_of_line_length)
    if (first_max + second_max == 4):
        max_element = 4
    else:
        max_element = min(first_max, second_max)
    pts_src = np.roll(pts_src, -max_element, axis=0)
    h, status = cv2.findHomography(pts_src[0:4], pts_dst[0:4], method=cv2.RANSAC, confidence=0.9999)
    h_inv, status_inv = cv2.findHomography(pts_dst[0:4], pts_src[0:4], method=cv2.RANSAC, confidence=0.9999)
    points = np.array([[0, 105, 1], [297, 105, 1]], np.float64)
    if sum(status) != 4:
        return img, 0
    else:
        points[0] = (h_inv.astype(np.float64)).dot(points[0])
        points[1] = (h_inv.astype(np.float64)).dot(points[1])
        points[0] = points[0] / points[0, -1]
        points[1] = points[1] / points[1, -1]
        if number_of_good_frames > 5:
            img = cv2.line(img, (int(points[0, 0]), int(points[0, 1])), (int(points[1, 0]), int(points[1, 1])),
                           (0, 255, 255), 3)
        return img, number_of_good_frames + 1


def step_8(img, pts_src, number_of_good_frames, max_length):
    pts_dst = np.array([[253, 0], [0, 105], [297, 105], [297, 0]])
    # pts_dst = np.array([[297, 0], [0, 0], [253, 210], [297, 210]])
    array_of_line_length = np.zeros(4)
    for i in range(4):
        array_of_line_length[i] = ((pts_src[i, 0] - pts_src[(i + 1) % 4, 0]) ** 2) + (
                (pts_src[i, 1] - pts_src[(i + 1) % 4, 1]) ** 2)
    first_max = np.argmax(array_of_line_length)
    max_element_length = array_of_line_length[first_max]
    array_of_line_length[first_max] = 0
    second_max = np.argmax(array_of_line_length)
    if first_max + second_max == 3 and first_max * second_max == 0:
        max_element = 3
    else:
        max_element = min(first_max, second_max)
    pts_src = np.roll(pts_src, -max_element, axis=0)
    h, status = cv2.findHomography(pts_src, pts_dst, method=cv2.RANSAC, confidence=0.9999)
    h_inv, status_inv = cv2.findHomography(pts_dst, pts_src, method=cv2.RANSAC, confidence=0.9999)
    points = np.array([[0, 105, 1], [297, 46, 1]], np.float64)
    if (sum(status) != 4):
        return img, 0, max_length
    else:
        points[0] = (h_inv.astype(np.float64)).dot(points[0])
        points[1] = (h_inv.astype(np.float64)).dot(points[1])
        points[0] = points[0] / points[0, -1]
        points[1] = points[1] / points[1, -1]
        if number_of_good_frames > 5:
            img = cv2.line(img, (int(points[0, 0]), int(points[0, 1])), (int(points[1, 0]), int(points[1, 1])),
                           (0, 255, 255), 3)
        return img, number_of_good_frames + 1, max_element_length


def step_9(img, pts_src, number_of_good_frames, current_step):
    pts_dst9 = np.array([[297, 105], [314, 88], [297, 46], [297, 0], [253, 0], [0, 105]])
    pts_dst10 = np.array([[314, 17], [297, 0], [0, 0], [253, 105], [297, 105], [297, 59]])

    array_of_line_length = np.zeros(6)
    for i in range(6):
        array_of_line_length[i] = ((pts_src[i, 0] - pts_src[(i + 1) % 6, 0]) ** 2) + (
                (pts_src[i, 1] - pts_src[(i + 1) % 6, 1]) ** 2)
    pts_src = np.roll(pts_src, -np.argmin(array_of_line_length), axis=0)
    array_of_line_length = np.roll(array_of_line_length, -np.argmin(array_of_line_length))

    if current_step == 10:
        h, status = cv2.findHomography(pts_src[0:4], pts_dst10[0:4], method=cv2.RANSAC, confidence=0.9999)
        if sum(status) == 4:
            h_inv, status_inv = cv2.findHomography(pts_dst10[0:4], pts_src[0:4], method=cv2.RANSAC, confidence=0.9999)
            points = np.array([[0, 0, 1], [297, 59, 1]], np.float64)
            points[0] = (h_inv.astype(np.float64)).dot(points[0])
            points[1] = (h_inv.astype(np.float64)).dot(points[1])
            points[0] = points[0] / points[0, -1]
            points[1] = points[1] / points[1, -1]
            if number_of_good_frames > 5:
                img = cv2.line(img, (int(points[0, 0]), int(points[0, 1])), (int(points[1, 0]), int(points[1, 1])),
                               (0, 255, 255), 3)
            return img, number_of_good_frames + 1, 10
        else:
            return img, 0, 10
    else:
        if (array_of_line_length[5] > array_of_line_length[1] and current_step == 9):
            h, status = cv2.findHomography(pts_src[0:4], pts_dst9[0:4], method=cv2.RANSAC, confidence=0.9999)
            if (sum(status) == 4):
                return img, number_of_good_frames + 1, 9
            else:
                return img, 0, 9
        elif (array_of_line_length[5] < array_of_line_length[1] and current_step == 9):
            h, status = cv2.findHomography(pts_src[0:4], pts_dst10[0:4], method=cv2.RANSAC, confidence=0.9999)
            if sum(status) == 4:
                h_inv, status_inv = cv2.findHomography(pts_dst10[0:4], pts_src[0:4], method=cv2.RANSAC,
                                                       confidence=0.9999)
                points = np.array([[0, 0, 1], [297, 59, 1]], np.float64)
                points[0] = (h_inv.astype(np.float64)).dot(points[0])
                points[1] = (h_inv.astype(np.float64)).dot(points[1])
                points[0] = points[0] / points[0, -1]
                points[1] = points[1] / points[1, -1]
                if number_of_good_frames > 5:
                    img = cv2.line(img, (int(points[0, 0]), int(points[0, 1])), (int(points[1, 0]), int(points[1, 1])),
                                   (0, 255, 255), 3)
                return img, number_of_good_frames + 1, 10
        else:
            return img, 0, current_step


def step_11(img, pts_src, number_of_good_frames):
    pts_dst = np.array([[314, 17], [297, 0], [0, 0], [297, 59]])
    array_of_line_length = np.zeros(4)
    for i in range(4):
        array_of_line_length[i] = ((pts_src[i, 0] - pts_src[(i + 1) % 4, 0]) ** 2) + (
                (pts_src[i, 1] - pts_src[(i + 1) % 4, 1]) ** 2)
    pts_src = np.roll(pts_src, -np.argmin(array_of_line_length), axis=0)
    h, status = cv2.findHomography(pts_src, pts_dst, method=cv2.RANSAC, confidence=0.9999)
    h_inv, status_inv = cv2.findHomography(pts_dst, pts_src, method=cv2.RANSAC, confidence=0.9999)
    if (sum(status) != 4):
        return img, 0
    else:
        return img, number_of_good_frames + 1


def biggestContour(contours, n_ang=4, min_line_lenth=0.008):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 5000:
            perimeter = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, min_line_lenth * perimeter, True)
            if area > max_area and len(approx) == n_ang:
                biggest = approx
                max_area = area
    return biggest, max_area


def paste_image(backgrownd, first_tip_image, second_tip_image, scale=6):
    first_smaler_image = cv2.resize(first_tip_image, (backgrownd.shape[1] // scale, backgrownd.shape[0] // scale))
    second_smaler_image = cv2.resize(second_tip_image, (backgrownd.shape[1] // scale, backgrownd.shape[0] // scale))

    vert_start = int(backgrownd.shape[0] * 0.2 * (first_smaler_image.shape[0] / backgrownd.shape[0]))
    vert_end = vert_start + first_smaler_image.shape[0]

    horiz_start_1 = int(backgrownd.shape[1] * (1 - 2.2 * first_smaler_image.shape[1] / backgrownd.shape[1]))
    horiz_end_1 = horiz_start_1 + first_smaler_image.shape[1]

    horiz_start_2 = int(backgrownd.shape[1] * (1 - 1.1 * second_smaler_image.shape[1] / backgrownd.shape[1]))
    horiz_end_2 = horiz_start_2 + second_smaler_image.shape[1]

    backgrownd[vert_start:vert_end, horiz_start_1:horiz_end_1] = backgrownd[vert_start:vert_end,
                                                                 horiz_start_1:horiz_end_1] * 0.4 + first_smaler_image * 0.6
    backgrownd[vert_start:vert_end, horiz_start_2:horiz_end_2] = backgrownd[vert_start:vert_end,
                                                                 horiz_start_2:horiz_end_2] * 0.4 + second_smaler_image * 0.6

    cv2.putText(backgrownd, "CURRENT",
                (horiz_start_1 + second_smaler_image.shape[1] // scale, vert_end + second_smaler_image.shape[0] // 4),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2 * (backgrownd.shape[0] / 1080), (255, 50, 255),
                4 * int(backgrownd.shape[0] / 1080))

    cv2.putText(backgrownd, "NEXT",
                (horiz_start_2 + second_smaler_image.shape[1] // scale, vert_end + second_smaler_image.shape[0] // 4),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2 * (backgrownd.shape[0] / 1080), (255, 50, 255),
                4 * int(backgrownd.shape[0] / 1080))
    return backgrownd


def put_text_tip(img, text, color=(0, 255, 0)):
    cv2.putText(img, text, (res_frame.shape[0] // 10, res_frame.shape[0] // 10),
                cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5 * (res_frame.shape[0] / 1080),
                color=color, thickness=int(4 / (1080/img.shape[0])))
    return img


img_tips = []
for i in range(12):
    img_tips.append(cv2.imread("images_of_steps/" + str(i) + ".png"))

default_tip = "Show your result"
text_tips = ["Step 0: Show me A4 paper",
             "Step 1: Fold the sheet in half",
             "Step 2: Unfold the sheet back ",
             "Step 3: Fold the corner ",
             "Step 4: Fold the second corner ",
             "Step 5: Fold the edge along the line ",
             "Step 6: Fold the second edge along the line ",
             "Step 7: Fold in the middle along the line ",
             "Step 8: Fold along the line",
             "Step 9: Turn around",
             "Step 10: Fold along the line ",
             "Step 11: Well done"]

# _________________________________________________________________________________________________________________________________
# _________________________________________________________________________________________________________________________________
# _________________________________________________________________________________________________________________________________
# _________________________________________________________________________________________________________________________________

is_camera = False
if is_camera is True:
    cap = cv2.VideoCapture(0)
elif is_camera is False:
    cap = cv2.VideoCapture("123.mp4")


cv2.namedWindow('bar')

scale_for_resize = 1
n_good_frames = 0
current_step = 1
A4_area = 0
t_waiting = 1
max_length = 0
n_frame = 0
res, frame = cap.read()
is_write_video = False
if is_write_video is True:
    res_frame = cv2.resize(frame, (frame.shape[1] // scale_for_resize, frame.shape[0] // scale_for_resize))
    video_Writter = cv2.VideoWriter("result_video_90fps.avi", cv2.VideoWriter_fourcc(*'MJPG'), 90, (res_frame.shape[1], res_frame.shape[0]))
while True:
    res, frame = cap.read()
    if res is True:
        if is_camera is True:
            res_frame = cv2.flip(cv2.resize(frame, (frame.shape[1] // scale_for_resize, frame.shape[0] // scale_for_resize)), 1)
        elif is_camera is False:
            res_frame = cv2.resize(frame, (frame.shape[1] // scale_for_resize, frame.shape[0] // scale_for_resize))
        # res_frame = cv2.flip(cv2.resize(frame, (frame.shape[1] // scale_for_resize, frame.shape[0] // scale_for_resize)), 1)

    # res_frame = cv2.resize(frame, (frame.shape[1] // scale_for_resize, frame.shape[0] // scale_for_resize))

    n_frame+=1
    print(n_frame)
    gray = cv2.cvtColor(res_frame, cv2.COLOR_BGR2GRAY)
    im_gray_blur = cv2.GaussianBlur(gray, (3, 3), 9)
    edges = cv2.Canny(im_gray_blur, threshold1=50, threshold2=150, apertureSize=3)

    kernel = np.ones((3, 3), 'uint8')
    dilate_img = cv2.dilate(edges, kernel, iterations=2)
    erode_image = cv2.erode(dilate_img, kernel)
    contours, hir = cv2.findContours(erode_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.drawContours(res_frame, contours, -1, (0, 0, 0), 2)
    # # --------------------------------------------------------------------------------------

    if current_step == 1:
        biggest_cont, max_area1 = biggestContour(contours, 4)
        cv2.drawContours(res_frame, biggest_cont, -1, (0, 255, 0), 10)
        if len(biggest_cont) == 4:
            res_frame, n_good_frames = step_1(res_frame, np.array(biggest_cont).reshape(
                (biggest_cont.shape[0], biggest_cont.shape[2])), n_good_frames)
        else:
            n_good_frames = 0
        if n_good_frames > 4:
            A4_area = max_area1
            put_text_tip(res_frame, text_tips[1])
        else:
            put_text_tip(res_frame, default_tip, color=(20, 20, 255))
            # cv2.putText(res_frame, "Show me clean A4 paper!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.2,
            #             color=(0, 0, 255))
        if np.abs(A4_area / 2 - max_area1) < A4_area * 0.05:
            current_step = 2
    # --------------------------------------------------------------------------------------

    if current_step == 2:
        biggest_cont, max_area2 = biggestContour(contours, 4)
        cv2.drawContours(res_frame, biggest_cont, -1, (0, 255, 0), 10)
        if len(biggest_cont) == 4:
            res_frame, n_good_frames = step_2(res_frame, np.array(biggest_cont).reshape(
                (biggest_cont.shape[0], biggest_cont.shape[2])), n_good_frames)
        else:
            n_good_frames = 0
        if n_good_frames > 4:
            put_text_tip(res_frame, text_tips[2])
        else:
            put_text_tip(res_frame, default_tip, color=(20, 20, 255))
        if np.abs(A4_area - max_area2) < A4_area * 0.05:
            current_step = 3
    # --------------------------------------------------------------------------------------
    if current_step == 3:
        biggest_cont, max_area3 = biggestContour(contours, 4)
        cv2.drawContours(res_frame, biggest_cont, -1, (0, 255, 0), 10)
        if len(biggest_cont) == 4:
            res_frame, n_good_frames = step_3(res_frame, np.array(biggest_cont).reshape(
                (biggest_cont.shape[0], biggest_cont.shape[2])), n_good_frames)
        else:
            n_good_frames = 0
        if n_good_frames > 4:
            put_text_tip(res_frame, text_tips[3])
        else:
            put_text_tip(res_frame, default_tip, color=(20, 20, 255))
        # !!!чтобы перейти на след шаг с другим количеством углов, добавляю доп поиск контура с этим количеством углов
        biggest_cont, max_area3 = biggestContour(contours, 5)

        if len(biggest_cont) == 5 and np.abs(A4_area * 0.9116 - max_area3) < A4_area * 0.05:
            current_step = 4
    # --------------------------------------------------------------------------------------

    if current_step == 4:
        biggest_cont, max_area4 = biggestContour(contours, 5)
        cv2.drawContours(res_frame, biggest_cont, -1, (0, 255, 0), 10)
        if len(biggest_cont) == 5:
            res_frame, n_good_frames = step_4(res_frame, np.array(biggest_cont).reshape(
                (biggest_cont.shape[0], biggest_cont.shape[2])), n_good_frames)
        else:
            n_good_frames = 0
        if n_good_frames > 4:
            put_text_tip(res_frame, text_tips[4])
        else:
            put_text_tip(res_frame, default_tip, color=(20, 20, 255))
        if np.abs(A4_area * 0.8232 - max_area4) < A4_area * 0.05:
            current_step = 5
    # --------------------------------------------------------------------------------------

    if current_step == 5:
        biggest_cont, max_area5 = biggestContour(contours, 5)
        cv2.drawContours(res_frame, biggest_cont, -1, (0, 255, 0), 10)
        if len(biggest_cont) == 5:
            res_frame, n_good_frames = step_5(res_frame, np.array(biggest_cont).reshape(
                (biggest_cont.shape[0], biggest_cont.shape[2])), n_good_frames)
        else:
            n_good_frames = 0
        if n_good_frames > 1:
            put_text_tip(res_frame, text_tips[5])
        else:
            put_text_tip(res_frame, default_tip, color=(20, 20, 255))
        if np.abs(A4_area * 0.69 - max_area5) < A4_area * 0.05:
            current_step = 6
    # --------------------------------------------------------------------------------------

    if current_step == 6:
        biggest_cont, max_area6 = biggestContour(contours, 5)
        cv2.drawContours(res_frame, biggest_cont, -1, (0, 255, 0), 10)
        if len(biggest_cont) == 5:
            res_frame, n_good_frames = step_6(res_frame, np.array(biggest_cont).reshape(
                (biggest_cont.shape[0], biggest_cont.shape[2])), n_good_frames)
        else:
            n_good_frames = 0
        if n_good_frames > 4:
            put_text_tip(res_frame, text_tips[6])
        else:
            put_text_tip(res_frame, default_tip, color=(20, 20, 255))
        if np.abs(A4_area * 0.57 - max_area6) < A4_area * 0.05:
            current_step = 7
        # --------------------------------------------------------------------------------------
    if current_step == 7:
        biggest_cont, max_area7 = biggestContour(contours, 5)
        cv2.drawContours(res_frame, biggest_cont, -1, (0, 255, 0), 10)
        if len(biggest_cont) == 5:
            res_frame, n_good_frames = step_7(res_frame, np.array(biggest_cont).reshape(
                (biggest_cont.shape[0], biggest_cont.shape[2])), n_good_frames)
        else:
            n_good_frames = 0
        if n_good_frames > 4:
            put_text_tip(res_frame, text_tips[7])
        else:
            put_text_tip(res_frame, default_tip, color=(20, 20, 255))
        biggest_cont, max_area7 = biggestContour(contours, 4)

        if len(biggest_cont) == 4 and np.abs(A4_area * 0.287 - max_area7) < A4_area * 0.05:
            current_step = 8

        # --------------------------------------------------------------------------------------
    if current_step == 8:
        biggest_cont, max_area8 = biggestContour(contours, 4, min_line_lenth=0.01)
        cv2.drawContours(res_frame, biggest_cont, -1, (0, 255, 0), 10)
        if len(biggest_cont) == 4:
            res_frame, n_good_frames, max_length = step_8(res_frame, np.array(biggest_cont).reshape(
                (biggest_cont.shape[0], biggest_cont.shape[2])), n_good_frames, max_length)
        else:
            n_good_frames = 0
        if n_good_frames > 4:
            put_text_tip(res_frame, text_tips[8])
        else:
            put_text_tip(res_frame, default_tip, color=(20, 20, 255))
        biggest_cont, max_area8 = biggestContour(contours, 5, min_line_lenth=0.02)
        cv2.drawContours(res_frame, biggest_cont, -1, (0, 255, 0), 10)
        if len(biggest_cont) == 5:
            array_of_line_length = np.zeros(len(biggest_cont))
            help_array = np.array(biggest_cont).reshape((biggest_cont.shape[0], biggest_cont.shape[2]))
            for i in range(len(biggest_cont)):
                array_of_line_length[i] = ((help_array[i, 0] - help_array[(i + 1) % len(help_array), 0]) ** 2) + (
                        (help_array[i, 1] - help_array[(i + 1) % len(help_array), 1]) ** 2)
        if len(biggest_cont) == 5 and np.abs(
                A4_area * 0.2936 - max_area8) < A4_area * 0.05 and n_good_frames == 0 and abs(
            np.max(array_of_line_length) - max_length) < 0.05 * max_length:
            current_step = 9
        # --------------------------------------------------------------------------------------
    if current_step == 9:
        biggest_cont, max_area9 = biggestContour(contours, 6, min_line_lenth=0.01)
        cv2.drawContours(res_frame, biggest_cont, -1, (255, 0, 0), 10)
        if len(biggest_cont) == 6:
            res_frame, n_good_frames, current_step = step_9(res_frame, np.array(biggest_cont).reshape(
                (biggest_cont.shape[0], biggest_cont.shape[2])), n_good_frames, current_step)
        else:
            n_good_frames = 0
        if n_good_frames > 4:
            put_text_tip(res_frame, text_tips[9])
        else:
            put_text_tip(res_frame, default_tip, color=(20, 20, 255))
        # --------------------------------------------------------------------------------------
    if current_step == 10:
        biggest_cont, max_area10 = biggestContour(contours, 6, min_line_lenth=0.01)
        cv2.drawContours(res_frame, biggest_cont, -1, (255, 0, 0), 10)

        if len(biggest_cont) == 6:
            res_frame, n_good_frames, current_step = step_9(res_frame, np.array(biggest_cont).reshape(
                (biggest_cont.shape[0], biggest_cont.shape[2])), n_good_frames, current_step)
        else:
            n_good_frames = 0
        if n_good_frames > 4:
            put_text_tip(res_frame, text_tips[10])
        else:
            put_text_tip(res_frame, default_tip, color=(20, 20, 255))
        biggest_cont, max_area10 = biggestContour(contours, 4)

        if np.abs(A4_area * 0.14748276414 - max_area10) < A4_area * 0.05:
            current_step = 11

        # --------------------------------------------------------------------------------------
    if current_step == 11:
        # t_waiting = 0
        biggest_cont, max_area3 = biggestContour(contours, 4)
        cv2.drawContours(res_frame, biggest_cont, -1, (0, 255, 0), 10)
        if len(biggest_cont) == 4:
            res_frame, n_good_frames = step_11(res_frame, np.array(biggest_cont).reshape(
                (biggest_cont.shape[0], biggest_cont.shape[2])), n_good_frames)
        else:
            n_good_frames = 0
        if n_good_frames > 1:
            current_step = 12
        else:
            put_text_tip(res_frame, default_tip, color=(20, 20, 255))
        # --------------------------------------------------------------------------------------
    if current_step == 12:
        cv2.putText(res_frame, "WELL DONE", (res_frame.shape[0] // 2, res_frame.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=3 * (res_frame.shape[0] / 1080),
                    color=(0, 255, 0), thickness=3)

    # ______________________________________________________________________
    if current_step < 11:
        paste_image(res_frame, img_tips[current_step], img_tips[current_step + 1], scale=8)
    cv2.putText(res_frame, "Current step: " + str(current_step), (50, res_frame.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5 * (res_frame.shape[0] / 1080), color=(255, 255, 255))
    cv2.imshow("bar", res_frame)
    if is_write_video is True:
        video_Writter.write(res_frame)
    key = cv2.waitKey(t_waiting)
    if key == ord('p'):
        t_waiting = 0
    if key == ord('c'):
        t_waiting = 1
