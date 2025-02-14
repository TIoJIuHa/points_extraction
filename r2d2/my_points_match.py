import numpy as np
import cv2
from PIL import Image
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.joinpath("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_keypoints(image_path1, keypoints_file1, image_path2, keypoints_file2):
    img1 = cv2.imread(image_path1)
    data1 = np.load(keypoints_file1)
    keypoints1 = data1['keypoints']
    descriptors1 = data1['descriptors']

    img2 = cv2.imread(image_path2)
    data2 = np.load(keypoints_file2)
    keypoints2 = data2['keypoints']
    descriptors2 = data2['descriptors']

    FLAN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLAN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1[:10000], descriptors2[:10000], k=2)
    # matches = sorted(matches, key=lambda x:x.distance)
    good_matches = []
    for m1, m2 in matches:
        if m1.distance < 0.6 * m2.distance:
            good_matches.append([m1])
    print(len(matches), len(good_matches))

    keypoints1_cv = [
        cv2.KeyPoint(x=float(kp[0]), y=float(kp[1]), size=1) for kp in keypoints1
    ]
    keypoints2_cv = [
        cv2.KeyPoint(x=float(kp[0]), y=float(kp[1]), size=1) for kp in keypoints2
    ]
    r2d2_matches = cv2.drawMatchesKnn(
        img1, keypoints1_cv, img2, keypoints2_cv, good_matches, None, matchColor = (0,0,255), flags=2
    )
    cv2.imwrite(OUTPUT_DIR.joinpath("r2d2-matches.png"), r2d2_matches)

    for point in keypoints1:
        x, y, s = point
        cv2.circle(img1, (int(x), int(y)), 3, (0, 0, 255), 2)
    cv2.imwrite(OUTPUT_DIR.joinpath(f"r2d2-{image_path1.name}"), img1)

    for point in keypoints2:
        x, y, s = point
        cv2.circle(img2, (int(x), int(y)), 3, (0, 0, 255), 2)
    cv2.imwrite(OUTPUT_DIR.joinpath(f"r2d2-{image_path2.name}"), img2)

image_path1 = Path('imgs/flight-human.png')
keypoints_file1 = Path('imgs/flight-human.png.r2d2')
image_path2 = Path('imgs/flight-human2.png')
keypoints_file2 = Path('imgs/flight-human2.png.r2d2')
draw_keypoints(image_path1, keypoints_file1, image_path2, keypoints_file2)