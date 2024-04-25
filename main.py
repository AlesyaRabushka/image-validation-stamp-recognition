import cv2
import copy
from rectangles import *
import random



# получение маски изображения
def get_image_mask(img, threshold = -1):
    # размытие изображения
    img_blured = cv2.medianBlur(img, 3)
    cv2.imwrite('process-img/blured.png', img_blured)
    
    # изображение в чб
    img_gray = cv2.cvtColor(img_blured, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('process-img/gray.png', img_gray)

    # получение ?цветной части изображения?
    adaptiveThreshold = threshold if threshold >= 0 else cv2.mean(img)[0]
    color = cv2.cvtColor(img_blured, cv2.COLOR_BGR2HLS)
    mask = cv2.inRange(color, (0, int(adaptiveThreshold / 6), 60), (180, adaptiveThreshold, 255))

    # создание маски цветной части изображения
    result_mask = cv2.bitwise_and(img_gray, img_gray, mask=mask)
    cv2.imwrite('process-img/mask.png', result_mask)

    return result_mask

# полчуение контура с печатью на изображении
def get_stamp_contour(img, mask):
    # получение копии изображения для поиска контуров
    img_copy = copy.deepcopy(img)

    # поиск контуров на копии
    cont, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    img_contoured = cv2.drawContours(img_copy, cont, -1, 255, 3)
    cv2.imwrite('process-img/contoured.png', img_contoured)

    c = max(cont, key = cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),1)
    

    # new_contrs = copy.deepcopy(cont)
    # for i in range(1, len(cont)):
    #     x1,y1,w1,h1 = cv2.boundingRect(new_contrs[i - 1])
    #     x2,y2,w2,h2 = cv2.boundingRect(new_contrs[i])

    #     fact, cords = find_crossing_rectangles([x1,y1,w1,h1], [x2,y2,w2,h2])
    #     cv2.rectangle(img, (cords[0],cords[1]), (cords[0] + cords[2], cords[1] + cords[3]), (125,255,0), 1)


    # для поиска нескольких контуров
    # new = []
    # for c in cont:

    #     area = cv2.contourArea(c)
            
    #     if area >= 1303:
    #         # print('c', area)
    #         print('---', area)
    #         x,y,w,h = cv2.boundingRect(c)

    #         # все вершины прямоугольника по часовой стрелке
    #         # x y
    #         # x y+h
    #         # x+w y+h
    #         # x+w y
    #         new.append(c)

    #         print(x,y,x+w,y+h)
    #         # отрисовка найденного контура печати на оригинальном изображении
    #         cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 1)
            


    # for i in range(0, len(new)):
    #     for j in range(0, len(new)):
    #         if i != j:
    #             result = contours_intersect(img, new[i], new[j])
    #             print(result)
    # #     x1,y1,w1,h1 = cv2.boundingRect(new[i - 1])
    # #     x2,y2,w2,h2 = cv2.boundingRect(new[i])

    # #     fact, cords = find_crossing_rectangles([x1,y1,w1,h1], [x2,y2,w2,h2])
    # #     print(fact)
    #             if result:
    #                 coords1 = cv2.boundingRect(new[i])
    #                 coords2 = cv2.boundingRect(new[j])
                    
    #                 x,y,w,h = get_rectangle_coords(coords1, coords2)

    #                 cv2.rectangle(img, (x,y), (x+w, y+h), (125,240,0), 3)
    #                 # cv2.imwrite('process-img/stamp_rect_img'+str(i) + str(j) +'.png', img)
    


    cv2.imwrite('process-img/stamp_rect_img.png', img)


def stamp_recognition(img_path):
    img = cv2.imread(img_path)

    # получение маски изображения
    mask = get_image_mask(img)

    #
    get_stamp_contour(img, mask)


from ultralytics import YOLO



def get_class(id):
    if id == 0:
        return 'dog'

if __name__ == '__main__':
    img_path = 'data/images/val/n02085620_1502.jpg'

    model = YOLO('/home/aleksa/UNIVER/image-validation-stamp-recognition/runs/detect/train24/weights/best.pt')
    
    results = model.predict(source=img_path)

    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

    img = cv2.imread(img_path)

    # проходим по результатам распознавания
    for result in results:
        # определяем все рамки, в которых находятся распознанные объекты
        boxes = result.boxes.cpu().numpy()

        # список координат верхнего левого угла каждой рамки
        coordinates_list = boxes.xyxy

        # список вероятностей правильного распознавания сообтветствующих рамок
        confidences_list = boxes.conf

        # список идентификаторов классов
        classes_list = boxes.cls

    for result_id in range(0, len(confidences_list)):

        # отметаем все "объекты" с низким уроовнем уверенности
        if confidences_list[result_id] >= 0.5:

            # получаем имя класса по его идентификатору
            result_class_name = get_class(classes_list[result_id])

            # получаем показатель степени уверенности в верном распознавании объекта
            result_conf = str(confidences_list[result_id])

            # получаем координаты рамки
            result_coords = coordinates_list[result_id]

            x = int(result_coords[0])
            y = int(result_coords[1])
            w = int(result_coords[2])
            h = int(result_coords[3])

            # рандомно определяем цвет рамки
            color = colors[random.randint(1,100) % len(colors)]

            # отображаем рамку на изображении
            cv2.rectangle(img, (x,y), (w,h), (color), 3)

            # отображаем название класса и степень уверенности над рамкой объекта на итоговом изображении
            cv2.putText(img, f'{result_class_name}, {result_conf}%', (x, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
            
    # сохраняем изменения в итоговое изображение
    cv2.imwrite('new_img.png', img)
    