def collapse(nums: list, distance: float) -> list:
    """
    Create a new list from the input sorted list according to the following algorithm:

    1. Define an empty collection of numbers called the current group.
    2. Place the first number, i.e.: nums[0] into the current group.
    3. While the following numbers, i.e.: nums[1], nums[2], ..., nums[i] are less than a given distance from the
       first number in the current group, add them to the current group.
    4. At this step, the current number is greater than or equal to the given distance from the first number in the current group.
       Take the mean of the numbers in the current group, add the result to the output list, remove all numbers from the current 
       group and add the current number to the current group.
    5. Repeat 3, 4 until the end of the list. Take the mean of the remaining numbers in the current group and add the result to the
       output list.

    Parameters
    ----------
    nums:
        The input list of numbers.
    
    distance:
        The distance to determine which numbers belong in the same group. Expected to be greater than 0.
    
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
