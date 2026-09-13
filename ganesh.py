import cv2
import numpy as np
import turtle
import time


# =========================================================
# SETTINGS
# =========================================================

IMAGE_FILE = "bappa.jpg"

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800

# Drawing speed
DRAW_DELAY = 0.0015

# Edge detection
CANNY_LOW = 30
CANNY_HIGH = 100

# Ignore extremely tiny noise
MIN_CONTOUR_LENGTH = 8


# =========================================================
# 1. LOAD GANESH IMAGE
# =========================================================

img = cv2.imread(IMAGE_FILE)

if img is None:
    print()
    print("❌ ERROR!")
    print("bappa.jpg was not found.")
    print()
    print("Make sure your folder contains:")
    print("    bappa.py")
    print("    bappa.jpg")
    print()
    input("Press Enter to close...")
    exit()


# =========================================================
# 2. RESIZE IMAGE
# =========================================================

height, width = img.shape[:2]

max_width = 600
max_height = 600

scale = min(
    max_width / width,
    max_height / height
)

new_width = int(width * scale)
new_height = int(height * scale)

img = cv2.resize(
    img,
    (new_width, new_height),
    interpolation=cv2.INTER_AREA
)


# =========================================================
# 3. GRAYSCALE
# =========================================================

gray = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2GRAY
)


# =========================================================
# 4. IMPROVE CONTRAST
# =========================================================

gray = cv2.equalizeHist(gray)


# =========================================================
# 5. REDUCE SMALL NOISE
# =========================================================

gray = cv2.GaussianBlur(
    gray,
    (3, 3),
    0
)


# =========================================================
# 6. DETECT EDGES
# =========================================================

edges = cv2.Canny(
    gray,
    CANNY_LOW,
    CANNY_HIGH
)


# =========================================================
# 7. FIND ALL CONTOURS
# =========================================================

contours, hierarchy = cv2.findContours(
    edges,
    cv2.RETR_LIST,
    cv2.CHAIN_APPROX_NONE
)


# =========================================================
# 8. KEEP IMPORTANT DETAILS
# =========================================================

good_contours = []

for contour in contours:

    length = cv2.arcLength(
        contour,
        False
    )

    if length >= MIN_CONTOUR_LENGTH:
        good_contours.append(contour)


# =========================================================
# 9. SORT CONTOURS
# =========================================================

# Larger contours first,
# then smaller facial details.

good_contours.sort(
    key=cv2.contourArea,
    reverse=True
)


print()
print("================================")
print("🐘 GANESH DRAWING")
print("================================")
print()
print("Image loaded successfully!")
print("Image size:", new_width, "x", new_height)
print("Contours detected:", len(good_contours))
print()
print("Starting drawing...")
print()


# =========================================================
# 10. CREATE TURTLE WINDOW
# =========================================================

screen = turtle.Screen()

screen.setup(
    width=WINDOW_WIDTH,
    height=WINDOW_HEIGHT
)

screen.title(
    "Drawing Bappa using Python ✨"
)

screen.bgcolor(
    "#11131c"
)

# Prevent Turtle animation from becoming slow
screen.tracer(
    0,
    0
)


# =========================================================
# 11. CREATE DRAWING PEN
# =========================================================

pen = turtle.Turtle()

pen.hideturtle()

pen.speed(0)

pen.pensize(2)

# Orange glowing-style line
pen.pencolor(
    "#ff7043"
)

pen.penup()


# =========================================================
# 12. DRAW GANESH OUTLINE
# =========================================================

for contour_number, contour in enumerate(good_contours):

    points = contour.reshape(
        -1,
        2
    )

    if len(points) < 2:
        continue


    # Start point of this contour
    x0 = int(points[0][0])
    y0 = int(points[0][1])

    turtle_x = (
        x0
        - new_width / 2
    )

    turtle_y = (
        new_height / 2
        - y0
    )


    pen.penup()

    pen.goto(
        turtle_x,
        turtle_y
    )

    pen.pendown()


    # Draw the contour
    for point in points:

        x = int(point[0])
        y = int(point[1])


        # Convert OpenCV coordinates
        # into Turtle coordinates.

        turtle_x = (
            x
            - new_width / 2
        )

        turtle_y = (
            new_height / 2
            - y
        )


        pen.goto(
            turtle_x,
            turtle_y
        )


        screen.update()

        time.sleep(
            DRAW_DELAY
        )


    pen.penup()


# =========================================================
# 13. FINAL UPDATE
# =========================================================

screen.update()

print()
print("================================")
print("✨ DRAWING COMPLETED!")
print("================================")
print()
print("Ganesh outline is complete.")
print("Both eyes and smaller details are included.")
print()

# =========================================================
# 14. KEEP WINDOW OPEN
# =========================================================

screen.mainloop()