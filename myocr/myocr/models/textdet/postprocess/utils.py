import numpy as np
import cv2

from numpy.fft import ifft

from myocr.myocr.core.evaluation.utils import boundary_iou

def fill_hole(input_mask):
    h, w = input_mask.shape
    canvas = np.zeros((h + 2, w + 2), np.uint8)
    canvas[1:h + 1, 1:w + 1] = input_mask.copy()

    mask = np.zeros((h + 4, w + 4), np.uint8)

    cv2.floodFill(canvas, mask, (0, 0), 1)
    canvas = canvas[1:h + 1, 1:w + 1].astype(np.bool)

    return ~canvas | input_mask

def fourier2poly(fourier_coeff, num_reconstr_points=50):
    """ Inverse Fourier transform
        Args:
            fourier_coeff (ndarray): Fourier coefficients shaped (n, 2k+1),
                with n and k being candidates number and Fourier degree
                respectively.
            num_reconstr_points (int): Number of reconstructed polygon points.
        Returns:
            Polygons (ndarray): The reconstructed polygons shaped (n, n')
        """

    a = np.zeros((len(fourier_coeff), num_reconstr_points), dtype='complex')
    k = (len(fourier_coeff[0]) - 1) // 2

    a[:, 0:k + 1] = fourier_coeff[:, k:]
    a[:, -k:] = fourier_coeff[:, :k]

    poly_complex = ifft(a) * num_reconstr_points
    polygon = np.zeros((len(fourier_coeff), num_reconstr_points, 2))
    polygon[:, :, 0] = poly_complex.real
    polygon[:, :, 1] = poly_complex.imag
    return polygon.astype('int32').reshape((len(fourier_coeff), -1))

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