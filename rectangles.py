# определение нового, охватывающего прямоугольника
def find_enclosisng_rectangle(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    x = min(x1, x2)
    y = min(y1, y2)
    width = max(x1 + w1, x2 + w2) - x
    height = max(y1 + h1, y2 + h2) - y

    return [x, y, width, height]

# определение, являются ли прямоугольники пересекающимися
# def find_crossing_rectangles(rect1, rect2):
#     enclosing_rect = find_enclosisng_rectangle(rect1, rect2)
#     if enclosing_rect[2] >= rect1[2] and enclosing_rect[3] >= rect1[3] and enclosing_rect[2] >= rect2[2] and enclosing_rect[3] >= rect2[3]:
#         return True, enclosing_rect
#     else:
#         return False, []
    
def find_crossing_rectangles(rect1, rect2):

    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2



    enclosing_rect = find_enclosisng_rectangle(rect1, rect2)
    if enclosing_rect[2] >= rect1[2] and enclosing_rect[3] >= rect1[3] and enclosing_rect[2] >= rect2[2] and enclosing_rect[3] >= rect2[3]:
        return True, enclosing_rect
    else:
        return False, []



import cv2
import numpy as np
import copy

# Функция для определения пересечения двух контуров
def check_contour_intersection(contour1, contour2):
    # Создание пустого изображения
    img = cv2.drawContours(np.zeros((1, 1, 3), dtype=np.uint8), [contour1, contour2], -1, (255, 255, 255), cv2.FILLED)
    cv2.imwrite('process-img/ex'+str(contour1[0])+'.png', img)
    # Преобразование изображения в оттенки серого
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('process-img/gray-ex'+str(contour1[0])+'.png', img)
    # Подсчет количества ненулевых пикселей
    nonzero_pixels = cv2.countNonZero(gray)

    # Если количество ненулевых пикселей больше 0, значит контуры пересекаются
    return nonzero_pixels > 0

def contours_intersect(img, contour1, contour2):
    contours = [contour1, contour2]
    blank = np.zeros(img.shape[0:2])

    new_img = copy.deepcopy(img)

    image1 = cv2.drawContours(new_img, [contours[0]], -1, 1)
    image2 = cv2.drawContours(new_img, [contours[1]], -1, 1)

    intersection = np.logical_and(image1, image2)

    return intersection.any()


def get_rectangle_coords(coords1, coords2):

    x,y,w,h = 0,0,0,0
    if coords1[2] > coords2[2]:
        w = coords1[2]
    else:
        w = coords2[2]
    if coords1[3] > coords2[3]:
        h = coords1[3]
    else:
        h = coords2[3]

    if coords1[0] < coords2[0]:
        x = coords1[0]
    else:
        x = coords2[0]

    if coords1[1] < coords2[1]:
        y = coords1[1]
    else:
        y = coords2[1]

    return x,y,w,h


if __name__ == '__main__':
    get_rectangle_coords([4,1,2,5], [2,2,7,3])