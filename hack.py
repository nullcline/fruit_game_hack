import cv2
import numpy as np
import pyautogui
import pytesseract
import random

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

while True:
    screenshot = pyautogui.screenshot(region=(460, 280, 990, 580))
    frame_bgr = np.array(screenshot)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = frame_rgb

    rows = []
    try:
        # hardcoded values cause im a bum
        width_of_column = 21
        col_gap = 37
        height_of_rows = 32
        row_gap = 26

        for i in range(0, image.shape[0], height_of_rows + row_gap):
                row = image[i:i + height_of_rows,:]
                rows.append(row)

        cells = []

        for row in rows:
            nya = []
            for i in range(0, image.shape[1], width_of_column + col_gap):
                column = row[:, i:i + width_of_column]
                if len(nya)+1 <= 17:
                    nya.append(column)
            cells.append(nya)

        matrix = np.zeros((len(cells), len(cells[0])), dtype=object)
        # turn blanks into 0s
        for i in range(len(cells)):
            for j in range(len(cells[0])):
                # if cell is mostly black replace it with a picture of a 0
                if np.mean(cells[i][j]) > 220:
                    matrix[i,j] = cv2.imread("0.png")
                else:
                    matrix[i,j] = cells[i][j]

        # concatenate all the cells into one image
        concat_rows = []
        for i in range(matrix.shape[0]):
            # Concatenate images along columns (axis=1)
            concat_rows.append(np.concatenate(matrix[i, :], axis=1))
        image = np.concatenate(concat_rows, axis=0)

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        common_errors = [("B","8"), ("T","7"), ("§","5"), ("()","0"), ("("," "), (")"," "), (" ","")]
        text = pytesseract.image_to_string(thresh, config='--psm 6')
        for error in common_errors:
            text = text.replace(error[0], error[1])

        try:
            matrix = [[int(char) for char in line] for line in text.split('\n') if line.strip() != '']
            matrix = np.array(matrix)
        except Exception as e:
            print(e)

        # points already used in a pair
        visited = set()
        tens = []
        # different kernels to try for summing

        # horizontal
        for i in range(matrix.shape[0]):
            for rectange_size in range(1,matrix.shape[1]+1):
                for j in range(matrix.shape[1] - rectange_size + 1):
                    square = matrix[i, j:j+rectange_size]
                    if (i,j) in visited or (i,j+rectange_size) in visited:
                        continue
                    # if all the points in this rectangle add to 10, add them to the list
                    if np.sum(square) == 10:
                        points = [(i,j+k) for k in range(rectange_size)]
                        tens.append(points)
                        visited.update(points)
        # vertical
        for j in range(matrix.shape[1]):
            for rectange_size in range(1,matrix.shape[0]+1):
                for i in range(matrix.shape[0] - rectange_size + 1):
                    square = matrix[i:i+rectange_size, j]
                    if np.sum(square) == 10:
                        points = [(i+k,j) for k in range(rectange_size)]
                        tens.append(points)
                        visited.update(points)

        # box
        kernel_size = 2
        for i in range(matrix.shape[0] - kernel_size + 1):
            for j in range(matrix.shape[1] - kernel_size + 1):
                square = matrix[i:i+kernel_size, j:j+kernel_size]
                if np.sum(square) == 10:
                    points = [(i+k,j+l) for k in range(kernel_size) for l in range(kernel_size)]
                    tens.append(points)
                    visited.update(points)

        cell_height = thresh.shape[0] // len(matrix)
        cell_width  = thresh.shape[1] // len(matrix[0])
        # highlight the image
        for ten in tens:
            overlay = np.zeros((cell_height, cell_width, 3))
            overlay[:] = (random.randint(100,255), random.randint(100,255), random.randint(100,255))
            for point in ten:
                row, col = point

                # Define the coordinates for the region corresponding to the grid cell
                y1 = row * cell_height
                y2 = (row + 1) * cell_height
                x1 = col * cell_width
                x2 = (col + 1) * cell_width
                # Change hue based on the index of the cell
                image[y1:y2, x1:x2] = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_RGB2HSV)
    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        break

    cv2.imshow('frame', image)
    cv2.waitKey(1)