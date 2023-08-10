import myocr.myocr.utils as utils


def extract_boundary(result):
    """Extract boundaries and their scores from result.

    Args:
        result (dict): The detection result with the key 'boundary_result'
            of one image.

    Returns:
        boundaries_with_scores (list[list[float]]): The boundary and score
            list.
        boundaries (list[list[float]]): The boundary list.
        scores (list[float]): The boundary score list.
    """
    assert isinstance(result, dict)
    assert 'boundary_result' in result.keys()

    boundaries_with_scores = result['boundary_result']
    assert utils.is_2dlist(boundaries_with_scores)

    boundaries = [b[:-1] for b in boundaries_with_scores]
    scores = [b[-1] for b in boundaries_with_scores]

    return (boundaries_with_scores, boundaries, scores)