def get_horizontal_position(box, frame_width):
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    left_boundary = frame_width / 3
    right_boundary = (frame_width * 2 ) / 3

    if center_x < left_boundary:
        return "LEFT"
    elif center_x < right_boundary:
        return "CENTER"
    else:
        return "RIGHT"