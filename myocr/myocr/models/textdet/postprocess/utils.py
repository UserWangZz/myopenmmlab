import numpy as np

from myocr.myocr.core.evaluation.utils import boundary_iou


def poly_nms(polygons, threshold):
    assert isinstance(polygons, list)

    polygons = np.array(sorted(polygons, key=lambda x: x[-1]))

    keep_poly = []
    index = [i for i in range(polygons.shape[0])]

    while len(index) > 0:
        keep_poly.append(polygons[index[-1]].tolist())
        A = polygons[index[-1]][:-1]
        index = np.delete(index, -1)

        iou_list = np.zeros((len(index), ))
        for i in range(len(index)):
            B = polygons[index[i]][:-1]

            iou_list[i] = boundary_iou(A, B, 1)
        remove_index = np.where(iou_list > threshold)
        index = np.delete(index, remove_index)

    return keep_poly