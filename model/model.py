from ultralytics import YOLO

class Model():
    """
    класс предназначен для хранения готовой модели
    и её обучения для определенного датасета
    """
    def __init__(self):
        # переменная, содержащая модель
        self.model = YOLO("yolo8s.pt")

    def train(self, data='config.yaml', epochs=30):
        # обучение модели
        result = self.model.train(data, epochs)

        return result




# model = YOLO("yolov8s.pt")

# result = model.train(data='config.yaml', epochs=30)

# class ObjectDetection():
#     def __init__(self, capture_index):
#         self.capture_index = capture_index

#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         print('Using device: ', self.device)

#         self.model = self.load_model()

#     def load_model(self):
#         model = YOLO('yolo8s.pt')
#         model.fuse()

#         return model
    
#     def predict(self, img):
#         results = self.model(img)

#         return results
    
#     def plot_bboxes(self, results, img):
#         xyxys = []
#         confidences = []
#         class_ids = []

#         for result in results:
#             boxes = result.boxes.cpu().numpy()

#             xyxys = boxes.xyxy

#         for xyxy in xyxys:
#             cv2.rectangle(img, (int(xyxy[0], int(xyxy[1])), (int(xyxy[2], int(xyxy[3])), (0,255,0))))
        
#         cv2.imwrite('new_img.png', img)