import math

from note import Note


def collapse(nums: list, distance: float) -> list:
    """
    Create a new list from the input list where adjacent items in the input list with similar values are collapsed to an
    average (mean) value in the new list.

    The algorithm works by creating groups of numbers where a number is added to the group if it is within the given distance
    from the first number added to the group. If it is not within the distance of the first number in the group, a new group
    is formed with this number as the first in the new group.

    Parameters
    ----------
    nums:
        The input list of numbers.
    
    distance:
        The measure of similarity with which to compare adjacent numbers in the list.
        Note that this does not necessarily mean two adjacent numbers within the given distance of each other are collapsed
        into the same average value.
    
        E.g.: Given nums = [0, 1, 3] and distance = 2.5, the result will be [0.5, 3].

    Returns
    -------
    out:
        The collapsed list of numbers.
    """
    if not nums:
        return []

    output: list = []
    group_count: int = 1
    
    group_sum = nums[0]
    group_start = nums[0]

    for current_number in nums[1:]:
        if abs(current_number - group_start) < distance:
            group_sum += current_number
            group_count += 1

        else:
            output.append(group_sum / group_count)
            group_sum = current_number
            group_count = 1
            group_start = current_number

    output.append(group_sum / group_count)

    return output
