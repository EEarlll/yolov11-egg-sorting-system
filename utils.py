import numpy as np

def get_features(width, height):
    ratio = width / height
    delta = (ratio - 0.5) * (height / 2)
    
    a = height / 2 + delta
    b = height / 2 - delta
    
    a_corrected = max(a, b)
    b_corrected = min(a, b)
    
    semi_major_axis_ratio = b_corrected / a_corrected
    
    theta = np.pi / 4
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    half_width = width / 2
    d_right = (a_corrected * half_width) / np.sqrt((half_width**2) * (cos_theta**2) + (a_corrected**2) * (sin_theta**2))
    d_left = (b_corrected * half_width) / np.sqrt((half_width**2) * (cos_theta**2) + (b_corrected**2) * (sin_theta**2))
    
    return {
        'longer_semi_major': a_corrected,
        'shorter_semi_major': b_corrected,
        'semi_major_axis_ratio': semi_major_axis_ratio,
        'd1': d_right,
        'd2': d_left,
        'd3': d_left,
        'd4': d_right,
    }

def calculateLength(width_pixel, height_pixel, ref_x_min, ref_x_max, real_length_of_reference):
    ref_length_pixel = ref_x_max - ref_x_min
    conversion_factor = real_length_of_reference / ref_length_pixel

    real_width = width_pixel * conversion_factor
    real_height = height_pixel * conversion_factor
    
    return real_width, real_height

def calculatePerimeterArea(perimeter, area, ref_x_min, ref_x_max, real_length_of_reference):
    ref_length_pixel = ref_x_max - ref_x_min
    conversion_factor = real_length_of_reference / ref_length_pixel

    real_perimeter = perimeter * conversion_factor
    real_area = area * conversion_factor

    return real_perimeter, real_area