from vision.spatial import get_horizontal_position

WIDTH = 300  # 0–100 LEFT, 100–200 CENTER, 200–300 RIGHT


def box_centered_at(x):
    return (x - 10, 0, x + 10, 50)


def test_left():
    assert get_horizontal_position(box_centered_at(50), WIDTH) == "LEFT"


def test_center():
    assert get_horizontal_position(box_centered_at(150), WIDTH) == "CENTER"


def test_right():
    assert get_horizontal_position(box_centered_at(250), WIDTH) == "RIGHT"


def test_left_center_boundary():
    assert get_horizontal_position(box_centered_at(99), WIDTH) == "LEFT"
    assert get_horizontal_position(box_centered_at(100), WIDTH) == "CENTER"


def test_center_right_boundary():
    assert get_horizontal_position(box_centered_at(199), WIDTH) == "CENTER"
    assert get_horizontal_position(box_centered_at(200), WIDTH) == "RIGHT"


def test_wide_box_uses_center():
    # spans all three thirds, center is at 150
    assert get_horizontal_position((0, 0, 300, 50), WIDTH) == "CENTER"