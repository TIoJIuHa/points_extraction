import argparse
import glob
import cv2
import os
import shutil
import numpy as np
from pathlib import Path

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd


MAX_NUM_POINTS = 10000  # Максимальное количество ключевых точек

class SuperPointLightGlue:
    @property
    def model(self):
        """Инициализация модели SuperPoint"""
        return SuperPoint(max_num_keypoints=MAX_NUM_POINTS)
    
    @property
    def matcher(self):
        """Инициализация сопоставителя LightGlue"""
        return LightGlue(features="superpoint")
    
    def detect_and_compute(self, img_path: str):
        """Нахождение ключевых точек и дескрипторов для изображения

        Parameters
        ----------
        img1_path : str
            путь до изображения
        """
        img = load_image(img_path)
        features = self.model.extract(img)
        return features
    
    def matches(self, features1, features2):
        """Сопоставление извлеченных признаков

        Parameters
        ----------
        features1 : dict
            словарь с извлеченными признаками первого изображения
        features2 : dict
            словарь с извлеченными признаками второго изображения
        """
        matches01 = self.matcher({"image0": features1, "image1": features2})
        features1, features2, matches01 = [
            rbd(x) for x in [features1, features2, matches01]
        ]
        keypoints1, keypoints2, matches = features1["keypoints"], features2["keypoints"], matches01["matches"]
        scores1, scores2 = features1["keypoint_scores"], features2["keypoint_scores"]
        good_keypoints1, good_keypoints2, = keypoints1[matches[..., 0]], keypoints2[matches[..., 1]]
        good_scores1, good_scores2, = scores1[matches[..., 0]].reshape(-1, 1), scores2[matches[..., 1]].reshape(-1, 1)
        result1 = np.concatenate((good_keypoints1, good_scores1), axis=1)
        result2 = np.concatenate((good_keypoints2, good_scores2), axis=1)
        return result1, result2


def export_detections(dataset_path: Path, output: Path):
    """Получает ключевые точки для всех заданных изображений и сохраняет их.

    Parameters
    ----------
    dataset_path : Path
        путь до папке с изображениями
    output : Path
        путь до папке, где будут сохранены результаты (ключевые точки)
    """
    lg = SuperPointLightGlue()
    folders = os.listdir(dataset_path)  # Получение подпапок с разными последовательными кадрами


    for folder in folders:
        pts_folder = output.joinpath("pts", folder)
        pts_folder.mkdir(parents=True, exist_ok=True)

        # Получение изображений в текущей директории по порядку
        images = sorted(glob.glob(f"{dataset_path}/{folder}/*.jpg"), key=lambda name: int(Path(name).stem.split("_")[-1]))
        # Нахождение ключевых точек для i - 1 кадра
        prev_features = lg.detect_and_compute(images[0])
        print(folder, ":")

        for i in range(1, len(images)-1):
            # Нахождение ключевых точек для текущего кадра
            curr_features = lg.detect_and_compute(images[i])
            # Нахождение сопоставлений точек между предыдущим и текущим кадрами
            _, matches01 = lg.matches(prev_features, curr_features)
            # Нахождение ключевых точек для следующего кадра
            next_features = lg.detect_and_compute(images[i+1])
            # Нахождение сопоставлений точек между текущим и следующим кадрами
            matches12, _ = lg.matches(curr_features, next_features)

            pred = {}
            mask = np.isin(matches01, matches12).all(axis=1)  # Вычисляем пересечение между найденными сопоставлениями
            intersection = matches01[mask]
            print(i, intersection.shape)

            # Если больше ста точек сопоставляется и с предыдущим и со следующим кадрами,
            # то записываем для этого изображения его ключевые точки в файл .npz
            if intersection.shape[0] > 100:
                pred.update({"pts": intersection})
                filename = Path(images[i]).name
                path = pts_folder.joinpath(filename).with_suffix(".npz")
                np.savez_compressed(path, **pred)

            prev_features = curr_features


def make_resized_samples(dataset: Path, output: Path, new_width:int = 640, new_height:int = 480):
    """Преобразовывает заданные изображения и их ключевые точки в соответствии с заданным размером.

    Parameters
    ----------
    dataset : Path
        путь до папке с изображениями
    output : Path
        путь до папке, где будут сохранены результаты
    new_width : int
        новая ширина для изображений (по умолчанию 640)
    new_height : int
        новая высота для изображений (по умолчанию 480)
    """
    os.makedirs(output.joinpath("images/train"), exist_ok=True)
    os.makedirs(output.joinpath("images/val"), exist_ok=True)
    os.makedirs(output.joinpath("predictions/train"), exist_ok=True)
    os.makedirs(output.joinpath("predictions/val"), exist_ok=True)

    train_size = 0
    validation_size = 0

    initial_path = output.joinpath("pts")

    # Преобразовываем изображения для каждой подпапки датасета
    for folder in os.listdir(initial_path):
        folder_path = initial_path.joinpath(folder)
        npz_files = glob.glob(f"{folder_path}/*.npz")
        np.random.shuffle(npz_files)  # Перемешиваем файлы с ключевыми точками
        val_size = int(0.2 * len(npz_files))
        val_files = npz_files[:val_size]  # Файлы для валидации
        train_files = npz_files[val_size:]  # Файлы для тренировки

        train_size += len(train_files)
        validation_size += len(val_files)
        
        # Проходимся по файлам тренировочной выборки
        for file in train_files:
            filename = Path(file).name
            image = filename.replace(".npz", ".jpg")

            input_image = cv2.imread(dataset.joinpath(folder, image))
            original_height, original_width  = input_image.shape[:2]

            # Вычисляем коэффициенты масштабирования
            scale_x = new_width / original_width
            scale_y = new_height / original_height

            # Преобразовываем размеры исходного изображения и сохраняем в тренироваочную выборку
            resized_image = cv2.resize(input_image, (new_width, new_height))
            cv2.imwrite(output.joinpath("images/train", image), resized_image)

            pts_file = np.load(file)
            pts = pts_file["pts"]
            resized_keypoints = []

            # Изменяем координаты ключевых точек в соответствии с новыми размерами
            for kp in pts:
                x_resized = int(kp[0] * scale_x)
                y_resized = int(kp[1] * scale_y)
                score = kp[2]
                resized_keypoints.append([x_resized, y_resized, score])

            # Проверяем валидность точек и записываем эти метки в тестовую выборку
            resized_keypoints = np.array(resized_keypoints)
            valid_indices = (resized_keypoints[:, 0] >= 0) & (resized_keypoints[:, 0] < new_width) & (resized_keypoints[:, 1] >= 0) & (resized_keypoints[:, 1] < new_height)
            valid_pnts = resized_keypoints[valid_indices]
            pred = {"pts": valid_pnts}
            npz_train_path = output.joinpath("predictions/train", filename)
            np.savez_compressed(npz_train_path, **pred)

        # Проходимся по файлам валидационной выборки
        for file in val_files:
            filename = Path(file).name
            image = filename.replace(".npz", ".jpg")

            input_image = cv2.imread(dataset.joinpath(folder, image))
            original_height, original_width  = input_image.shape[:2]

            # Вычисляем коэффициенты масштабирования
            scale_x = new_width / original_width
            scale_y = new_height / original_height

            # Преобразовываем размеры исходного изображения и сохраняем в валидационную выборку
            resized_image = cv2.resize(input_image, (new_width, new_height))
            cv2.imwrite(output.joinpath("images/val", image), resized_image)

            pts_file = np.load(file)
            pts = pts_file["pts"]
            resized_keypoints = []

            # Изменяем координаты ключевых точек в соответствии с новыми размерами
            for kp in pts:
                x_resized = int(kp[0] * scale_x)
                y_resized = int(kp[1] * scale_y)
                score = kp[2]
                resized_keypoints.append([x_resized, y_resized, score])

            # Проверяем валидность точек и записываем эти метки в валидационную выборку
            resized_keypoints = np.array(resized_keypoints)
            valid_indices = (resized_keypoints[:, 0] >= 0) & (resized_keypoints[:, 0] < new_width) & (resized_keypoints[:, 1] >= 0) & (resized_keypoints[:, 1] < new_height)
            valid_pnts = resized_keypoints[valid_indices]
            pred = {"pts": valid_pnts}
            npz_train_path = output.joinpath("predictions/val", filename)
            np.savez_compressed(npz_train_path, **pred)

    shutil.rmtree(initial_path)

    print(f"Created train and validation samples (path '{output}/predictions'):")
    print(f"Train - {train_size} items")
    print(f"Validation - {validation_size} items")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Разметка датасета")
    parser.add_argument("--dataset-path", type=Path, default="points_extraction/dataset/training", help="Путь до датасета")
    parser.add_argument("--output-path", type=Path, default="points_extraction/output", help="Путь для выходных данных")

    args = parser.parse_args()
    dataset = args.dataset_path
    output = args.output_path

    export_detections(dataset, output)
    make_resized_samples(dataset, output)
