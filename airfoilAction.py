import numpy as np
import random

def calculate_total_distance(data_points):
    total_distance = 0.0
    for i in range(len(data_points) - 1):
        total_distance += np.sqrt((data_points[i+1][0] - data_points[i][0])**2 + (data_points[i+1][1] - data_points[i][1])**2)
    return total_distance

def smooth_adjustment(upper_half_points, lower_half_points, x_target, y_change_upper, y_change_lower, sigma=1.0, distance_threshold=0.1):
    # Calculate distances from the target x-coordinate for each point in upper and lower curves
    distances_upper = [abs(p[0] - x_target) for p in upper_half_points]
    distances_lower = [abs(p[0] - x_target) for p in lower_half_points]
    
    # Calculate gaussian weights based on distance
    # Set weights to zero for points that are farther than the distance threshold
    weights_upper = [np.exp(-d**2 / (2 * sigma**2)) if d < distance_threshold else 0 for d in distances_upper]
    weights_lower = [np.exp(-d**2 / (2 * sigma**2)) if d < distance_threshold else 0 for d in distances_lower]
    # Set the weight of x_target to 0 to avoid double adjustment
    idx_target_upper = np.argmin(distances_upper)
    idx_target_lower = np.argmin(distances_lower)
    weights_upper[idx_target_upper] = weights_upper[idx_target_upper] - 1
    weights_lower[idx_target_lower] = weights_lower[idx_target_lower] - 1

    # Adjust the y-coordinates of upper and lower curves based on the weights
    adjusted_upper = np.copy(upper_half_points)
    adjusted_lower = np.copy(lower_half_points)
    
    for i, (x, y) in enumerate(upper_half_points):
        adjusted_upper[i][1] = y + y_change_upper * weights_upper[i]
    
    for i, (x, y) in enumerate(lower_half_points):
        adjusted_lower[i][1] = y + y_change_lower * weights_lower[i]

    # Handle the start and end points (average adjustment)
    avg_change = (y_change_upper + y_change_lower) / 2
    adjusted_upper[0][1] = upper_half_points[0][1] +  avg_change * weights_upper[0] + avg_change * weights_lower[0]
    adjusted_upper[-1][1] = upper_half_points[-1][1] + avg_change * weights_upper[-1] + avg_change * weights_lower[-1]
    adjusted_lower[0][1] = lower_half_points[0][1] + avg_change * weights_upper[0] + avg_change * weights_lower[0]
    adjusted_lower[-1][1] = lower_half_points[-1][1] + avg_change * weights_upper[-1] + avg_change * weights_lower[-1]

    stop_x = None
    for i, (x, y) in enumerate(adjusted_upper):
        if adjusted_upper[i][1] - adjusted_lower[i][1] < (upper_half_points[i][1] - lower_half_points[i][1]) / 2:
            stop_x = x
            break

    if stop_x is not None:
        for i, (x, y) in enumerate(adjusted_upper):
            d = abs(x - x_target)
            if d >= abs(stop_x - x_target):
                adjusted_upper[i] = upper_half_points[i]
                adjusted_lower[i] = lower_half_points[i]     


    return adjusted_upper, adjusted_lower


def adjust_point(data_points_upper, data_points_lower, x_target, y_change_upper, y_change_lower):
    warning_occurred = False  # 默认值为False
    for i, point in enumerate(data_points_upper):
        if point[0] == x_target:
            old_value_upper = data_points_upper[i][1]
            new_value_upper = old_value_upper + y_change_upper
    for i, point in enumerate(data_points_lower):
        if point[0] == x_target:
            old_value_lower = data_points_lower[i][1]
            new_value_lower = old_value_lower + y_change_lower
    if new_value_upper <= new_value_lower:
        print(f"警告：在 {x_target} 处的调整导致上半曲线的 y 值小于或等于下半曲线的 y >值。这次调整将被忽略。")
        warning_occurred = True
        return data_points_upper, data_points_lower, warning_occurred
    else:
        for i, point in enumerate(data_points_upper):
            if point[0] == x_target:
                data_points_upper[i][1] = new_value_upper
                print(f"上半曲线点 ({x_target}, {old_value_upper}) 的 y 值被调整为 {new_value_upper}")
        for i, point in enumerate(data_points_lower):
            if point[0] == x_target:
                data_points_lower[i][1] = new_value_lower
                print(f"下半曲线点 ({x_target}, {old_value_lower}) 的 y 值被调整为 {new_value_lower}")

    return data_points_upper, data_points_lower, warning_occurred  # 在函数的末尾返回


def adjust_pointIE(data_points_upper, data_points_lower, x_target, y_change):
    for i, point in enumerate(data_points_upper):
        if point[0] == x_target:
            old_value_upper = data_points_upper[i][1]
            new_value_upper = old_value_upper + y_change
            data_points_upper[i][1] = new_value_upper
            print(f"上半曲线点 ({x_target}, {old_value_upper}) 的 y 值被调整为 {new_value_upper}")
    for i, point in enumerate(data_points_lower):
        if point[0] == x_target:
            old_value_lower = data_points_lower[i][1]
            new_value_lower = old_value_lower + y_change
            data_points_lower[i][1] = new_value_lower
            print(f"下半曲线点 ({x_target}, {old_value_lower}) 的 y 值被调整为 {new_value_lower}")
    return data_points_upper, data_points_lower



def add_point(data_points, x_new):
    # 找到 x_new 左右两侧的点
    for i in range(len(data_points) - 1):
        x_left, y_left = data_points[i]
        x_right, y_right = data_points[i+1]
        if x_left <= x_new <= x_right:
            # 对 y 值进行线性插值
            y_new = y_left + (y_right - y_left) * (x_new - x_left) / (x_right - x_left)
            data_points.insert(i+1, [x_new, y_new])
            print(f"在 {x_new} 处添加了一个新的点，y 值为 {y_new}")
            break
    return data_points


def remove_point(data_points_upper, data_points_lower, x_target):
    data_points_upper = [point for point in data_points_upper if point[0] != x_target]
    data_points_lower = [point for point in data_points_lower if point[0] != x_target]
    print(f"删除了 x = {x_target} 的点")
    return data_points_upper, data_points_lower


def closest_point(a, points):
    # 找到最接近 a 的点
    closest_x = min(points, key=lambda x: abs(x[0] - a))
    return closest_x[0]  # 返回 x 坐标

def perform_action(filepath, action, action_params):
    with open(filepath, 'r') as file:
        data = file.readlines()

    upper_half_points = []
    lower_half_points = []
    current_section = 0  # 0 for header, 1 for upper half, 2 for lower half

    for i, line in enumerate(data):
        line = line.strip()

        if line == '':
            current_section += 1
            continue

        if current_section == 0:
            continue  # Ignore the header lines

        if len(line.split()) != 2:
            print(f"警告：跳过了行 '{line}', 因为它不含有正好两个元素")
            continue

        try:
            x, y = map(float, line.split())
        except ValueError:
            print(f"警告：跳过了行 '{line}', 因为它不含有能转换为浮点数的元素")
            continue

        if current_section == 1:
            upper_half_points.append([x, y])
        elif current_section == 2:
            lower_half_points.append([x, y])
        else:
            print(f"警告：忽略了行 '{line}', 因为它在未预期的位置")

    if action == 0:  # 改变点，但是不改变初端和末端
        a = action_params[0]
        x_target = closest_point(a, upper_half_points[1:-1])  # 忽略初端和末端
        y_change_upper = action_params[1]
        y_change_lower = action_params[2]
        upper_half_points, lower_half_points, warning_occurred = adjust_point(upper_half_points, lower_half_points, x_target, y_change_upper, y_change_lower)
        upper_half_points, lower_half_points = smooth_adjustment(upper_half_points, lower_half_points, x_target, y_change_upper, y_change_lower)

    elif action == 1:  # 插入点
        x_new = action_params[0]
        upper_half_points = add_point(upper_half_points, x_new)
        lower_half_points = add_point(lower_half_points, x_new)

    elif action == 2:  # 删除点
        a = action_params[0]
        x_target = closest_point(a, upper_half_points[1:-1])  # 忽略初端和末端
        upper_half_points, lower_half_points = remove_point(upper_half_points, lower_half_points, x_target)

    elif action == 3:  # 改变初端点
        y_change = action_params[0]
        x_target = upper_half_points[0][0]
        upper_half_points, lower_half_points = adjust_pointIE(upper_half_points, lower_half_points, x_target, y_change)
        upper_half_points, lower_half_points = smooth_adjustment(upper_half_points, lower_half_points, x_target, y_change, y_change)

    elif action == 4:  # 改变末端点
        y_change = action_params[0]
        x_target = upper_half_points[-1][0]
        upper_half_points, lower_half_points = adjust_pointIE(upper_half_points, lower_half_points, x_target, y_change)
        upper_half_points, lower_half_points = smooth_adjustment(upper_half_points, lower_half_points, x_target, y_change, y_change)
    upperhalf_distance = calculate_total_distance(upper_half_points)
    print(f"upper half distance = '{upperhalf_distance}'")
    if 0.9  <= upperhalf_distance <= 1.6:
        with open(filepath, 'w') as file:
            file.write('NACA 0012 AIRFOILS\n')
            file.write(f'{len(upper_half_points)}       {len(lower_half_points)}.\n')
            file.write('\n')
            for point in upper_half_points:
                file.write(' '.join(map(lambda x: '{:.16f}'.format(x), point)) + '\n')
            file.write('\n')
            for point in lower_half_points:
                file.write(' '.join(map(lambda x: '{:.16f}'.format(x), point)) + '\n')
        return True, True
    else:
        return False, True

