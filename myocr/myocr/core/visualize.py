import warnings

import cv2
import mycv
import numpy as np

import myocr.myocr.utils as utils


def imshow_pred_boundary(img,
                         boundaries_with_scores,
                         labels,
                         score_thr=0,
                         boundary_color='blue',
                         text_color='blue',
                         thickness=1,
                         font_scale=0.5,
                         show=True,
                         win_name='',
                         wait_time=0,
                         out_file=None,
                         show_score=False):
    """Draw boundaries and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        boundaries_with_scores (list[list[float]]): Boundaries with scores.
        labels (list[int]): Labels of boundaries.
        score_thr (float): Minimum score of boundaries to be shown.
        boundary_color (str or tuple or :obj:`Color`): Color of boundaries.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename of the output.
        show_score (bool): Whether to show text instance score.
    """
    assert isinstance(img, (str, np.ndarray))
    assert utils.is_2dlist(boundaries_with_scores)
    assert utils.is_type_list(labels, int)
    assert utils.equal_len(boundaries_with_scores, labels)
    if len(boundaries_with_scores) == 0:
        warnings.warn('0 text found in ' + out_file)
        return None

    utils.valid_boundary(boundaries_with_scores[0])
    img = mycv.imread(img)

    scores = np.array([b[-1] for b in boundaries_with_scores])
    inds = scores > score_thr
    boundaries = [boundaries_with_scores[i][:-1] for i in np.where(inds)[0]]
    scores = [scores[i] for i in np.where(inds)[0]]
    labels = [labels[i] for i in np.where(inds)[0]]

    boundary_color = mycv.color_val(boundary_color)
    text_color = mycv.color_val(text_color)
    font_scale = 0.5

    for boundary, score in zip(boundaries, scores):
        boundary_int = np.array(boundary).astype(np.int32)

        cv2.polylines(
            img, [boundary_int.reshape(-1, 1, 2)],
            True,
            color=boundary_color,
            thickness=thickness)

        if show_score:
            label_text = f'{score:.02f}'
            cv2.putText(img, label_text,
                        (boundary_int[0], boundary_int[1] - 2),
                        cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
    if show:
        mycv.imshow(img, win_name, wait_time)
    if out_file is not None:
        mycv.imwrite(img, out_file)

    return img