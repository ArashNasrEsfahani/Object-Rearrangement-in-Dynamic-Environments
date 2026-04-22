import itertools
import time

data_dict = {2: [130, 149, 0.753472, 0.859028, 0.773611, 0.851389], 7: [48, 15, 0.51875, 0.593056, 0.493056, 0.68125], 19: [15, 72, 0.532639, 0.688889, 0.429861, 0.694444], 22: [21, 29, 0.5, 0.590972, 0.671528, 0.572917], 33: [106, 7, 0.640278, 0.579167, 0.499306, 0.654167], 38: [187, 187, 0.413194, 0.588194, 0.493056, 0.640972], 39: [78, 78, 0.41875, 0.690972, 0.419444, 0.690972], 45: [25, 26, 0.784028, 0.711111, 0.870833, 0.761111], 8: [101, 82, 0.808333, 0.911806, 0.519444, 0.743056], 34: [76, 47, 0.545139, 0.669444, 0.626389, 0.7125]}

def min_distance_move(source, destination, circumference):
    """ Calculate minimum distance on a circle. """
    direct = abs(destination - source)
    reverse = circumference - direct
    return min(direct, reverse)

def greedy_lookahead(current_position, remaining_objects, data_dict, circumference, lookahead):
    """ Recursively find the best next move considering lookahead steps. """
    if not remaining_objects:
        return [], 0

    if lookahead == 0 or len(remaining_objects) == 1:
        next_obj_id = remaining_objects[0]
        source, destination = data_dict[next_obj_id][0:2]
        total_dist = (min_distance_move(current_position, source, circumference) +
                      min_distance_move(source, destination, circumference))
        return [next_obj_id], total_dist

    best_sequence = None
    min_distance = float('inf')

    for perm in itertools.permutations(remaining_objects[:lookahead]):
        perm_distance = 0
        current_perm_position = current_position
        valid = True
        for obj_id in perm:
            source, destination = data_dict[obj_id][0:2]
            perm_distance += min_distance_move(current_perm_position, source, circumference)
            perm_distance += min_distance_move(source, destination, circumference)
            current_perm_position = destination
            if perm_distance >= min_distance: 
                valid = False
                break

        if valid:
            remaining_after_perm = [obj for obj in remaining_objects if obj not in perm]
            next_sequence, future_distance = greedy_lookahead(current_perm_position, remaining_after_perm, data_dict, circumference, lookahead)
            total_dist = perm_distance + future_distance

            if total_dist < min_distance:
                min_distance = total_dist
                best_sequence = list(perm) + next_sequence

    return best_sequence, min_distance

circumference = 200
lookahead = 8  


start_position = 0
remaining_objects = list(data_dict.keys())

start_time = time.time()
optimal_order, min_total_distance = greedy_lookahead(start_position, remaining_objects, data_dict, circumference, lookahead)
end_time = time.time()
execution_time = end_time - start_time

print(f"Best order to pick the objects: {optimal_order}")
print(f"Minimum total distance traveled: {min_total_distance} frames")
print(f"Execution time: {execution_time:.2f} seconds")

data_dict = {obj_id: data_dict[obj_id] for obj_id in optimal_order}

print("Sorted data_dict based on optimal order:")
for obj_id, obj_data in data_dict.items():
    print(f"Object ID: {obj_id}, Data: {obj_data}")