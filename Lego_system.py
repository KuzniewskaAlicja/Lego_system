import cv2
import numpy as np 
from os import path, listdir
from json import load, dumps
import argparse

def parsing_command_line():
    cmd_line = argparse.ArgumentParser(description= 'Projekt - Alicja Kuźniewska')
    cmd_line.add_argument('images_directory', type = str, help = 'Path to the directory with input images')
    cmd_line.add_argument('input_file', type = str, help= 'Path to the file, which contain objects description')
    cmd_line.add_argument('output_file', type= str, help= 'Path to the output json file')

    args = cmd_line.parse_args()
    imgs_dir = args.images_directory
    input_path = args.input_file
    output_path = args.output_file

    return (imgs_dir, input_path, output_path)

def read_json(path_name):
    with open(path_name,'r') as file:
        json_file = load(file)
    
    return json_file

def load_images(path_name):
    images = [cv2.imread(path.join(path_name, file_name), cv2.IMREAD_COLOR) for file_name in sorted(listdir(path_name))]
    images = [cv2.resize(image, None, fx = 0.5, fy = 0.5) for image in images]

    return images

def create_object_image(rect, box, image):
    w, h = tuple(map(int, rect[1]))
    box_pts = box.astype('float32')
    image_pts = np.float32([[0, h - 1],
                            [0, 0],
                            [w - 1, 0],
                            [w - 1, h - 1]])
    Matrix = cv2.getPerspectiveTransform(box_pts, image_pts)
    img_perspective = cv2.warpPerspective(image, Matrix, (w, h))

    return img_perspective

def grouping_objects(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    objects_images = []
    edges = cv2.Canny(gray, 111, 46)
    edges = cv2.dilate(edges, (3,3), iterations=35)
    contour, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        if cv2.contourArea(cnt) > 2000:
            min_rect = cv2.minAreaRect(cnt)
            box_contour = cv2.boxPoints(min_rect)
            box_contour = np.int0(box_contour) 
            objects_images.append(create_object_image(min_rect, box_contour, image))
            
    return objects_images

def create_red_mask(image_hsv):
    red_mask = cv2.inRange(image_hsv, np.array([0, 54, 100]), np.array([10, 255,255]))
    red_mask = cv2.add(red_mask, cv2.inRange(image_hsv, np.array([160, 50, 50]), np.array([180, 200,200])))
    red_mask = cv2.dilate(red_mask, (3,3), iterations=1)

    return red_mask

def create_blue_mask(image_hsv):
    blue_mask = cv2.inRange(image_hsv, np.array([104, 70, 70]), np.array([130, 255,255]))
    blue_mask = cv2.add(blue_mask, cv2.inRange(image_hsv, np.array([104, 24, 180]), np.array([124, 64,220])))
    blue_mask = cv2.dilate(blue_mask, (3,3), iterations=5)

    return blue_mask

def create_yellow_mask(image_hsv):
    yellow_mask = cv2.inRange(image_hsv, np.array([13, 109, 160]), np.array([35, 255,240]))
    yellow_mask = cv2.add(yellow_mask, cv2.inRange(image_hsv, np.array([14, 32, 210]), np.array([35, 87,250])))
    yellow_mask = cv2.dilate(yellow_mask, (3,3), iterations=5)

    return yellow_mask

def create_gray_mask(image_hsv):
    gray_mask = cv2.inRange(image_hsv, np.array([37, 7, 56]), np.array([110, 60, 140]))
    gray_mask = cv2.add(gray_mask, cv2.inRange(image_hsv, np.array([37, 3, 56]), np.array([110, 60, 180])))
    gray_mask = cv2.erode(gray_mask, (3,3), iterations=3)

    return gray_mask

def create_white_mask(image_hsv):
    white_mask = cv2.inRange(image_hsv, np.array([37,8,177]), np.array([62,50,255]))
    white_mask = cv2.dilate(white_mask, (3,3), iterations=12)

    return white_mask

def detecting_circles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    object_circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 34, param1=39, param2=20, minRadius=6, maxRadius=17)
    object_circles = np.uint16(np.around(object_circles))
    
    return len(object_circles[0,:])

def counting_red_objects(image):
    mask = create_red_mask(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    red_counter = contours_counter(image, mask)
    
    return red_counter

def counting_blue_objects(image):
    mask = create_blue_mask(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    blue_counter = contours_counter(image, mask)

    return blue_counter

def counting_gray_objects(image):
    mask = create_gray_mask(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    gray_counter = contours_counter(image, mask)
    
    return gray_counter

def counting_white_objects(image):
    mask = create_white_mask(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    white_counter = contours_counter(image, mask)
    
    return white_counter

def counting_yellow_objects(image):
    mask = create_yellow_mask(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    yellow_counter = contours_counter(image, mask)

    return yellow_counter

def contours_counter(image, mask):
    counter = 0
    BLOCK_HEIGHT = 43.0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 700:
            height = min(cv2.minAreaRect(cnt)[1])
            counter += height // BLOCK_HEIGHT

    return int(counter)

def processing_json_file(json_file):
    image_keys = tuple(json_file.keys())
    object_keys = tuple(json_file[image_keys[0]][0].keys())

    return image_keys, object_keys
        
def create_object_description(image, keys):
    red_counter = counting_red_objects(image)
    yelllow_counter = counting_yellow_objects(image)
    blue_counter = counting_blue_objects(image)
    gray_counter = counting_gray_objects(image)
    white_counter = counting_white_objects(image)
    counters_list = (red_counter, blue_counter, white_counter, gray_counter, yelllow_counter)
    counters_list = tuple(map(str, counters_list))
    desctiption = dict(zip(keys, counters_list))
    
    return desctiption
    
def get_match_index(object_description, json_file, image_key):
    
    for object_number, descr in enumerate(json_file[image_key]):
        if descr == object_description:
            return object_number

def create_unmatching_objects_list(all_objects_index, match_objects_index, objects):
    unmatching_object_index = [unmatch for unmatch in all_objects_index if not unmatch in match_objects_index]
    unmatching_objects = [[i, objects[i][j]] for (i,j) in unmatching_object_index]

    return unmatching_objects

def create_unmatching_file_index(json_file, json_image_keys, strong_matching):
    all_objects_file = [[image_number, object_number] for (image_number, image_key) in enumerate(json_image_keys) for object_number in range(len(json_file[image_key]))]
    matching_file_index = [match[0] for match in strong_matching]
    unmatching_file_index = [object_index for object_index in all_objects_file if not object_index in matching_file_index]

    return unmatching_file_index

def matching_elements(json_file, unmatching_file_index, object_description, image_number, number_correct_elements):
    image_keys, object_keys = processing_json_file(json_file)
    compatibility_counter = 0
    possible_matching = []
    object_index = []
    for (i,j) in unmatching_file_index:
        compatibility_counter = 0
        if i == image_number:
            description = json_file[image_keys[i]][j]
            for color in object_keys:
                if description[color] == object_description[color]:
                    compatibility_counter += 1
                if color == object_keys[len(object_keys) - 1]:
                    if compatibility_counter == number_correct_elements:
                        object_index.append([i,j])
                        possible_matching.append(description)

    if len(possible_matching) == 1 and len(object_index) == 1:
        unmatching_file_index.remove(object_index[0])
        return possible_matching[0]

def matching_remained_objects(unmatching_objects, json_input, unmatching_objects_file, number_considered_elements):
    image_keys, object_keys = processing_json_file(json_input)
    result = []
    for (image_number, obj) in unmatching_objects:
        description = create_object_description(obj.copy(), object_keys)
        match_description = matching_elements(json_input, unmatching_objects_file, description, image_number, number_considered_elements)
        if match_description:
            object_number = get_match_index(match_description, json_input, image_keys[image_number])
            if not object_number is None:
                circles_number = detecting_circles(obj.copy())
                result.append([[image_number, object_number], circles_number])

    return result

def create_output_file(image_keys, input_file, output_path, result):
    # create final dictionary
    final_output = input_file
    for ((image_number, element_number), circles_number) in result:
        final_output[image_keys[image_number]][element_number] = circles_number

    # saving json file
    save_output_file(output_path, final_output)

def save_output_file(output_path, final_result):
    with open(output_path, 'w') as f:
        output = dumps(final_result, indent= 4)
        f.write(output)

def main_program():
    
    # parse paths from command line
    imgs_path, input_path, output_path = parsing_command_line()
    
    # read json file
    json_input = read_json(path.normcase(input_path))
    image_keys, object_keys = processing_json_file(json_input)
    
    #loading images
    lists_of_images = load_images(path.normcase(imgs_path))
    
    #loading all objects from all images to list
    objects = [grouping_objects(image.copy()) for image in lists_of_images]
    
    # strong matching descriptions of object with description from file
    strong_matching = []
    match_objects_index = []
    all_objects_index= []
    for image_number in range(len(objects)):
        for object_index, obj in enumerate(objects[image_number]):
            description = create_object_description(obj.copy(), object_keys)
            all_objects_index.append([image_number, object_index])
            object_number = get_match_index(description, json_input, image_keys[image_number])

            if not object_number is None:
                match_objects_index.append([image_number, object_index])
                circles_number = detecting_circles(obj.copy())
                strong_matching.append([[image_number, object_number], circles_number])

    # unmatched object from objects list
    unmatching_objects = create_unmatching_objects_list(all_objects_index, match_objects_index, objects)
    
    # unmatched object from file
    unmatching_objects_file = create_unmatching_file_index(json_input, image_keys, strong_matching)

    # matching remained objects on considered number of compatible blocks
    four_considered = matching_remained_objects(unmatching_objects, json_input, unmatching_objects_file, 4)
    three_considered = matching_remained_objects(unmatching_objects, json_input, unmatching_objects_file, 3)
    two_considered = matching_remained_objects(unmatching_objects, json_input, unmatching_objects_file, 2)
    one_considered = matching_remained_objects(unmatching_objects, json_input, unmatching_objects_file, 1)

    result = four_considered + three_considered + two_considered + one_considered

    # result list
    result.extend(strong_matching)

    if len(unmatching_objects_file) != 0:
        for unmatch in unmatching_objects_file:
            result.append([unmatch, 20])
    
    result.sort(key = lambda x: x[0])

    # creating output file
    create_output_file(image_keys, json_input, output_path, result)
    
if __name__ == "__main__":
    main_program()
    