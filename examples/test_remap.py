import numpy as np
import cv2
import imageio


def init_fisheye_remap(K, params, width, height):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    mapx = np.zeros((height, width), dtype=np.float32)
    mapy = np.zeros((height, width), dtype=np.float32)
    for i in range(0, width):
        for j in range(0, height):
            x = float(i)
            y = float(j)
            x1 = (x - cx) / fx
            y1 = (y - cy) / fy
            theta = np.sqrt(x1**2 + y1**2)
            r = (
                1.0
                + params[0] * theta**2
                + params[1] * theta**4
                + params[2] * theta**6
                + params[3] * theta**8
            )
            x2 = fx * x1 * r + width // 2
            y2 = fy * y1 * r + height // 2
            mapx[j, i] = x2
            mapy[j, i] = y2
    return mapx, mapy


def main():
    K = np.array(
        [[610.93592297, 0.0, 876.0], [0.0, 610.84071973, 584.0], [0.0, 0.0, 1.0]]
    )
    params = np.array([0.03699945, 0.00660936, 0.00116909, -0.00038226])
    width, height = (1752, 1168)

    mapx, mapy = init_fisheye_remap(K, params, width, height)

    x_min = np.nonzero(mapx < 0)[1].max()
    x_max = np.nonzero(mapx > width)[1].min()
    y_min = np.nonzero(mapy < 0)[0].max()
    y_max = np.nonzero(mapy > height)[0].min()
    roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]
    K[0, 2] -= x_min
    K[1, 2] -= y_min

    image = imageio.imread("./data/zipnerf/fisheye/berlin/images_4/DSC00040.JPG")[
        ..., :3
    ]
    image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
    imageio.imwrite("./results/test_remap.png", image)
    x, y, w, h = roi_undist
    image = image[y : y + h, x : x + w]
    imageio.imwrite("./results/test_remap_crop.png", image)


if __name__ == "__main__":
    main()
