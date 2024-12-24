import cv2
import os
import numpy as np
import pywt
from sklearn.neighbors import KNeighborsClassifier

# === 1. Загрузка предобученной модели MobileNet SSD ===
def load_model():
    net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')
    return net

def detect_objects(image, net):
    # Изменяем размер изображения, так как MobileNet SSD принимает входные данные определённого размера
    blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), (127.5, 127.5, 127.5, 127.5), swapRB=True)
    net.setInput(blob)
    detections = net.forward()

    return detections

# === 3. Извлечение вейвлет-признаков ===
def extract_wavelet_features(image, wavelet='db4', level=3):
    try:
        coeffs = pywt.wavedec2(image, wavelet, level=level)
        features = []
        for coeff in coeffs:
            if isinstance(coeff, tuple):
                for subband in coeff:
                    features.extend(subband.flatten())
            else:
                features.extend(coeff.flatten())
        return np.array(features)
    except Exception as e:
        print("Ошибка при извлечении признаков:", e)
        return np.array([])

def process_images_with_auto_detection(folder, net, knn_classifier=None):
    if not os.path.exists(folder):
        print(f"Папка {folder} не найдена.")
        return

    classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 
               'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'plant', 'sheep', 'sofa', 'train', 
               'tvmonitor']

    files = os.listdir(folder)
    file_index = 0  # Индекс текущего файла

    # Если классификатор не передан, создаём новый
    if knn_classifier is None:
        knn_classifier = KNeighborsClassifier(n_neighbors=3)

    features_data = []
    labels = []

    while file_index < len(files):
        file = files[file_index]
        image_path = os.path.join(folder, file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Не удалось загрузить {file}.")
            continue

        height, width = image.shape[:2]
        detections = detect_objects(image, net)

        object_found = False
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:  # Порог уверенности
                class_id = int(detections[0, 0, i, 1])
                class_name = classes[class_id]
                if class_name == 'aeroplane':  # Проверяем, был ли найден самолёт
                    object_found = True
                    box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                    (startX, startY, endX, endY) = box.astype("int")
                    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.putText(image, f'Object: {class_name}', (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # Извлечение признаков из области, содержащей объект
                    roi = image[startY:endY, startX:endX]
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    gray_roi_resized = cv2.resize(gray_roi, (64, 64))
                    features_roi = extract_wavelet_features(gray_roi_resized)
                    print(f"Признаки из выделенной области: {features_roi[:30]}...")  # Печать первых 30 признаков

                    # Собираем признаки и метки для обучения классификатора
                    features_data.append(features_roi)
                    labels.append(class_name)

        if not object_found:
            cv2.putText(image, "", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Создаем окно в полноэкранном режиме
        cv2.namedWindow("Результат", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Результат", width, height)  # Устанавливаем размер окна под размер изображения
        # Показать изображение
        cv2.imshow("Результат", image)

        key = cv2.waitKey(0)  # Ждём нажатие клавиши
        if key == 13:  # Enter
            file_index += 1  # Переход к следующему изображению
        elif key == 27:  # Escape
            break  # Закрыть программу

    # Обучаем классификатор после того, как собраны все данные
    if len(features_data) > 8:  # Например, обучаем после сбора 8 примеров
        knn_classifier.fit(features_data, labels)
        print(f"Обучение завершено. Классификатор готов!")

        # Применение классификатора для предсказания
        for i in range(len(features_data)):
            try:
                predicted_label = knn_classifier.predict([features_data[i]])
                print(f"Предсказание для объекта: {predicted_label}")
            except Exception as e:
                print(f"Ошибка при предсказании: {e}")
    else:
        print("Недостаточно данных для обучения классификатора.")

    cv2.destroyAllWindows()

# === 5. Запуск программы ===
def main():
    data_dir = 'dataset/left_folder'  # Папка с изображениями
    net = load_model()
    knn_classifier = KNeighborsClassifier(n_neighbors=3)  # Создаем классификатор
    process_images_with_auto_detection(data_dir, net, knn_classifier)  # Передаем классификатор в функцию

if __name__ == '__main__':
    main()
