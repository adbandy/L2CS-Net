import cv2
import numpy as np
import math

def draw_gaze(gaze_origin, image_in, endpoint, color, thickness=2):
    """Draw gaze angle on given image with a given eye positions.
    :param gaze_origin: point of gaze source
    :param image_in: frame used
    :param endpoint: final point of gaze
    :param color: coloring of gaze arrow
    :param thickness: thickness of gaze arrow
    :return: a frame with an arrow of the gaze
    """
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    cv2.arrowedLine(image_out, tuple(np.round(gaze_origin).astype(int)),
                    tuple(np.round(endpoint).astype(int)), color,
                    thickness, cv2.LINE_AA, tipLength=0.18)
    return image_out

def draw_bbox(frame: np.ndarray, bbox: np.ndarray):
    """
    Draws Boundary Box
    :param frame: frame to draw on
    :param bbox: Boundary box
    :return: Boundary box on frame
    """
    x_min=int(bbox[0])
    if x_min < 0:
        x_min = 0
    y_min=int(bbox[1])
    if y_min < 0:
        y_min = 0
    x_max=int(bbox[2])
    y_max=int(bbox[3])

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1) #BGR colour

    return frame

def render(frame_count: int, frame: np.ndarray,
           results, predefined_bboxes,
           highlight_intersections=False, scale=20.0):

    """

    :param frame_count: current frame iteration
    :param frame: The frame to be annotated.
    :param results: A GazeResultContainer containing pitch,yaw,bboxes etc.
    :param highlight_intersections: A boolean flag indicating whether to color the intersecting gaze
    :param scale: Option to change relative size of the gaze arrows
    :return: frame with gaze, booleans for gaze intersections and the bboxes involved
    """
    gaze_source_bbox = []
    gaze_target_bbox = []
    predefined_gaze_intersections = []

    if results:

        for i, bbox in enumerate(results.bboxes):
            frame = draw_bbox(frame, bbox)
            pitch = results.pitch[i]
            yaw = results.yaw[i]

            # dx = -np.sin(pitch) * np.cos(yaw)
            # dy = -np.sin(yaw)
            """
            Positive yaw should point rightwards.
            Negative yaw should point leftwards.
            Positive pitch should point upwards.
            Negative pitch should point downwards."""
            #native yaw was negative for right
            dx = np.cos(pitch) * np.sin(yaw)
            dy = -np.sin(pitch)

            scale_factor = max(abs(dx), abs(dy)) * scale
            dx, dy = scale_factor * dx, scale_factor * dy
            gaze_origin = (int((bbox[0] + bbox[2]) / 2.0), int((bbox[1] + bbox[3]) / 2.0))
            end_point_x = int(gaze_origin[0] + (scale_factor*dx))
            end_point_y = int(gaze_origin[1] + (scale_factor*dy))
            end_point = (end_point_x, end_point_y)



            gaze_colour = (255,0,0) # default blue as cv2 is BGR not RGB
            if highlight_intersections and abs(yaw)>0.4 and abs(pitch)<0.3:
                for other_bbox_index, other_bbox in enumerate(results.bboxes):
                    if other_bbox_index != i:
                        intersection_results = is_gaze_in_bbox(gaze_origin, end_point, other_bbox)
                        if intersection_results:
                            x, y = intersection_results[0], intersection_results[1]
                            gaze_colour = (0,255,0) #green
                            gaze_source_bbox.append(i)
                            gaze_target_bbox.append(other_bbox_index)

            for idx, predefined_bbox in enumerate(predefined_bboxes):
                if is_gaze_in_bbox(gaze_origin, end_point, predefined_bbox):
                    predefined_gaze_intersections.append((i, idx))
                    # gaze_colour = (0, 0, 255)  # red for predefined boundary box intersection
                    break

            draw_gaze(gaze_origin, frame, end_point,color=gaze_colour)
    return frame, gaze_source_bbox, gaze_target_bbox, predefined_gaze_intersections


def is_gaze_in_bbox(gaze_origin, end_point, target_bbox):
    """
    Checks if the gaze line intersects with the given bounding box.

    Args:
        gaze_origin (tuple): The (x, y) coordinates of the gaze origin.
        end_point (tuple): The (x, y) coordinates of the end point of the gaze line.
        bbox (tuple): The bounding box in the form (x1, y1, x2, y2).

    Returns:
        bool: True if the gaze intersects with the bbox, False otherwise.
    """
    x1, y1, x2, y2 = target_bbox
    #add error margin
    error_margin = 0.1
    x_margin = abs(x2-x1) * error_margin
    y_margin = abs(y2-y1) * error_margin
    x1, y1, x2, y2 = x1 - x_margin, y1 - y_margin, x2 + x_margin, y2 + y_margin
    x1, y1, x2, y2 = [int(float(i)) for i in [x1, y1, x2, y2]]
    bbox_lines = [
        [(x1, y1), (x2, y1)],  # top edge
        [(x2, y1), (x2, y2)],  # right edge
        [(x2, y2), (x1, y2)],  # bottom edge
        [(x1, y2), (x1, y1)]  # left edge
    ]

    for bbox_line in bbox_lines:
        intersection = line_intersection(gaze_origin, end_point, bbox_line[0], bbox_line[1])
        if intersection:
            return intersection

    return False

def line_intersection(line1_start, line1_end, line2_start, line2_end):
    """
    Finds the intersection point of two lines.

    Args:
        line1_start: A tuple representing the start point of line 1 (x1, y1).
        line1_end: A tuple representing the end point of line 1 (x2, y2).
        line2_start: A tuple representing the start point of line 2 (x1, y1).
        line2_end: A tuple representing the end point of line 2 (x2, y2).

    Returns:
        A tuple representing the intersection point (x, y) or None if no intersection.
    """
    xdiff = (line1_start[0] - line1_end[0], line2_start[0] - line2_end[0])
    ydiff = (line1_start[1] - line1_end[1], line2_start[1] - line2_end[1])

    def det(a, b):
        assert isinstance(a, tuple) and len(a) == 2, f"Expected tuple of length 2 for a, got {a}"
        assert isinstance(b, tuple) and len(b) == 2, f"Expected tuple of length 2 for b, got {b}"
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None  # Lines don't intersect


    d = (det(line1_start, line1_end), det(line2_start, line2_end))
    div = det(xdiff, ydiff)
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    def is_between(a, b, c):
        #determines if intersection coordinate is in boundary box edges
        return min(a, b) <= c <= max(a, b)

    def is_in_direction(start, end, point):
        return ((end[0] - start[0]) * (point[0] - start[0]) +
                (end[1] - start[1]) * (point[1] - start[1])) >= 0

    if is_between(line2_start[0], line2_end[0], x) and is_between(line2_start[1], line2_end[1], y):
        if is_in_direction(line1_start, line1_end, (x, y)):
            return x, y


    return None


"""
analyze_video looks nice but I have a dataframe called df.
Each row in df is a frame, with columns of 'path' (for the file path), 'pitches', 'yaws', 'bboxes'
Pitches and yaws are lists with the first item belonging to the first person seen from left to right from the cameras perspective. 

bboxes is a list of lists, each sublist contains [x1,y1,x2,y2]. 
"""