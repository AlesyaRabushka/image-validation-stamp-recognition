import cv2
import copy

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
    cont, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_contoured = cv2.drawContours(img_copy, cont, -1, 255, 3)
    cv2.imwrite('process-img/contoured.png', img_contoured)

    c = max(cont, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    # отрисовка найденного контура печати на оригинальном изображении
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 1)
    cv2.imwrite('process-img/stamp_rect_img.png', img)


def stamp_recognition(img_path):
    img = cv2.imread(img_path)

    # получение маски изображения
    mask = get_image_mask(img)

    #
    get_stamp_contour(img, mask)



if __name__ == '__main__':
    stamp_recognition('/home/aleksa/UNIVER/image-validation-stamp-recognition/img/1.jpg')