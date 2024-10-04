import cv2
import random


from ultralytics import YOLO



    
class ObjectDetection:
     
    """
    класс предназначен для распознавания наличия печати
    на изображении документа
    """
    def __init__(self, weight_path, img_path):
        # путь к изображению для распознания печатей на нем
        self.img_path = img_path

        # путь к весовым коэффициентам
        self.weight_path = weight_path

        # яркие цвета для отображения областей печатей
        self.colors = [(255, 0, 255), (255, 255, 0), (0, 255, 0), (255, 165, 0),
                        (0, 255, 255), (255, 0, 0), (128, 0, 128), (0, 0, 255)]

        # модель для распознавания
        self.model = YOLO(self.weight_path)

        # количество печатей
        self.stamps = 0


     # получение названия класса, исходя из его идентификатора
    def get_class(self, id):
        if id == 0:
            return 'stamp'


    # метод распознавания печатей
    def predict(self):
        self.results = self.model.predict(source=self.img_path)
          
   

    # отрисовка прямоугольников с печатями
    # на изображении
    def draw_rectangles(self):
         # преобразование картинки для распознавания
        img = cv2.imread(self.img_path)
        print(self.results)

        # проходим по результатам распознавания
        for result in self.results:
            # определяем все рамки, в которых находятся распознанные объекты
            boxes = result.boxes.cpu().numpy()

            # получаем количество обнаруженных печатей
            self.stamps = len(boxes)

            # список координат верхнего левого угла каждой рамки
            coordinates_list = boxes.xyxy

            # список вероятностей правильного распознавания сообтветствующих рамок
            confidences_list = boxes.conf

            # список идентификаторов классов
            classes_list = boxes.cls
        

        for result_id in range(0, len(confidences_list)):

                # получаем имя класса по его идентификатору
                result_class_name = self.get_class(classes_list[result_id])

                # получаем показатель степени уверенности в верном распознавании объекта
                result_conf = str(round(confidences_list[result_id] * 100, 3))

                # получаем координаты рамки
                result_coords = coordinates_list[result_id]

                x = int(result_coords[0])
                y = int(result_coords[1])
                w = int(result_coords[2])
                h = int(result_coords[3])

                # рандомно определяем цвет рамки
                color = self.colors[random.randint(1,100) % len(self.colors)]

                # отображаем рамку на изображении
                cv2.rectangle(img, (x,y), (w,h), (color), 3)

                # отображаем название класса и степень уверенности над рамкой объекта на итоговом изображении
                cv2.putText(img, f'{result_class_name}, {result_conf}%', (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2, cv2.LINE_AA)
                
        # сохраняем изменения в итоговое изображение
        cv2.imwrite('result-img/new_img.png', img)

        if self.stamps == 0:
             print("\nПечатей не обнаружено")
        else:
            print(f"\nОбнаружено печатей в количестве: {self.stamps} ")
    
         

if __name__ == '__main__':
    weights =  'best.pt'
    image_path = 'img/1.jpg'

    detection = ObjectDetection(weights, image_path)
    detection.predict()
    detection.draw_rectangles()
   