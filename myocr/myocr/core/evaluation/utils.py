import numpy as np
from shapely.geometry import Polygon as plg

import myocr.myocr.utils as utils


def boundary_iou(src, target, zero_division=0):
    """Calculate the IOU between two boundaries.

    Args:
       src (list): Source boundary.
       target (list): Target boundary.
       zero_division (int|float): The return value when invalid
                                    boundary exists.

    Returns:
       iou (float): The iou between two boundaries.
    """
    assert utils.valid_boundary(src, False)
    assert utils.valid_boundary(target, False)
    src_poly = points2polygon(src)
    target_poly = points2polygon(target)

    return poly_iou(src_poly, target_poly, zero_division=zero_division)


def poly_iou(poly_det, poly_gt, zero_division=0):
    """Calculate the IOU between two polygons.

    Args:
        poly_det (Polygon): A polygon predicted by detector.
        poly_gt (Polygon): A gt polygon.
        zero_division (int|float): The return value when invalid
                                    polygon exists.

    Returns:
        iou (float): The IOU between two polygons.
    """
    assert isinstance(poly_det, plg)
    assert isinstance(poly_gt, plg)
    area_inters = poly_intersection(poly_det, poly_gt)
    area_union = poly_union(poly_det, poly_gt)
    return area_inters / area_union if area_union != 0 else zero_division


def points2polygon(points):
    """Convert k points to 1 polygon.

    Args:
        points (ndarray or list): A ndarray or a list of shape (2k)
            that indicates k points.

    Returns:
        polygon (Polygon): A polygon object.
    """
    if isinstance(points, list):
        points = np.array(points)

    assert isinstance(points, np.ndarray)
    assert (points.size % 2 == 0) and (points.size >= 8)

    point_mat = points.reshape([-1, 2])
    return plg(point_mat)


def poly_union(poly_det, poly_gt, invalid_ret=None, return_poly=False):
    """Calculate the union area between two polygon.
    Args:
        poly_det (Polygon): A polygon predicted by detector.
        poly_gt (Polygon): A gt polygon.
        invalid_ret (None|float|int): The return value when the invalid polygon
            exists. If it is not specified, the function allows the computation
            to proceed with invalid polygons by cleaning the their
            self-touching or self-crossing parts.
        return_poly (bool): Whether to return the polygon of the intersection
            area.

    Returns:
        union_area (float): The union area between two polygons.
        poly_obj (Polygon|MultiPolygon, optional): The Polygon or MultiPolygon
            object of the union of the inputs. The type of object depends on
            whether they intersect or not. Set as `None` if the input is
            invalid.
    """
    assert isinstance(poly_det, plg)
    assert isinstance(poly_gt, plg)
    assert invalid_ret is None or isinstance(invalid_ret, float) or \
           isinstance(invalid_ret, int)

    if invalid_ret is None:
        poly_det = poly_make_valid(poly_det)
        poly_gt = poly_make_valid(poly_gt)

    poly_obj = None
    area = invalid_ret
    if poly_det.is_valid and poly_gt.is_valid:
        poly_obj = poly_det.union(poly_gt)
        area = poly_obj.area
    return (area, poly_obj) if return_poly else area


def poly_make_valid(poly):
    """Convert a potentially invalid polygon to a valid one by eliminating
    self-crossing or self-touching parts.

    Args:
        poly (Polygon): A polygon needed to be converted.

    Returns:
        A valid polygon.
    """
    return poly if poly.is_valid else poly.buffer(0)


def poly_intersection(poly_det, poly_gt, invalid_ret=None, return_poly=False):
    """Calculate the intersection area between two polygon.

    Args:
        poly_det (Polygon): A polygon predicted by detector.
        poly_gt (Polygon): A gt polygon.
        invalid_ret (None|float|int): The return value when the invalid polygon
            exists. If it is not specified, the function allows the computation
            to proceed with invalid polygons by cleaning the their
            self-touching or self-crossing parts.
        return_poly (bool): Whether to return the polygon of the intersection
            area.

    Returns:
        intersection_area (float): The intersection area between two polygons.
        poly_obj (Polygon, optional): The Polygon object of the intersection
            area. Set as `None` if the input is invalid.
    """
    assert isinstance(poly_det, plg)
    assert isinstance(poly_gt, plg)
    assert invalid_ret is None or isinstance(invalid_ret, float) or \
           isinstance(invalid_ret, int)

    if invalid_ret is None:
        poly_det = poly_make_valid(poly_det)
        poly_gt = poly_make_valid(poly_gt)

    poly_obj = None
    area = invalid_ret
    if poly_det.is_valid and poly_gt.is_valid:
        poly_obj = poly_det.intersection(poly_gt)
        area = poly_obj.area
    return (area, poly_obj) if return_poly else area
