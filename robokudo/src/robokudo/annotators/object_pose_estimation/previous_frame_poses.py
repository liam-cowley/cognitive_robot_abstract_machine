"""
Matching of objects to the objects posed in the previous frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import TYPE_CHECKING

from robokudo.utils.comparators import RoiComparator

if TYPE_CHECKING:
    from krrood_spatial_math.spatial_types import Pose

    from robokudo.types.cv import Rect


# %% Tracking across frames


@dataclass(eq=False)
class PosedObject:
    """
    An object whose pose was determined in a frame.

    Posed objects are equal only to themselves.
    """

    class_name: str
    """
    Name of the class of the object.
    """

    pose: Pose
    """
    Pose of the object in the camera frame.
    """

    mask_region: Rect
    """
    The smallest rectangle containing the object's mask, in depth image pixels.
    """


@dataclass
class PreviousFramePoses:
    """
    Objects posed in the previous frame that are not yet matched to an object of the
    current frame.
    """

    unmatched_objects: list[PosedObject] = field(default_factory=list)
    """
    Posed objects still available for matching.
    """

    def take_match(
        self, class_name: str, mask_region: Rect, minimum_overlap: float
    ) -> Pose | None:
        """
        Remove and return the pose of the unmatched object of a class whose mask region
        overlaps most with a mask region of the current frame.

        :param class_name: Class name of the object of the current frame.
        :param mask_region: Mask region of the object of the current frame.
        :param minimum_overlap: Intersection over union below which mask regions do not
            match.
        :return: The pose of the matching object, or ``None`` if none overlaps enough.
        """
        candidates = [
            posed_object
            for posed_object in self.unmatched_objects
            if posed_object.class_name == class_name
        ]
        if not candidates:
            return None
        region_comparator = RoiComparator(weight=1.0)
        overlaps = [
            region_comparator.compute_similarity(candidate.mask_region, mask_region)
            for candidate in candidates
        ]
        best_overlap = max(overlaps)
        if best_overlap < minimum_overlap:
            return None
        match = candidates[overlaps.index(best_overlap)]
        self.unmatched_objects.remove(match)
        return match.pose
