import json
import os
import csv
from tqdm import tqdm


def convert_to_rectangle(points):
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(y_values)
    y_max = max(y_values)
    return [x_min, y_min, x_max, y_max]


def convert_labelme_to_rectangles(labelme_file, csv_file):
    with open(labelme_file, 'r') as f:
        labelme_data = json.load(f)

    with open(csv_file, 'w', newline='') as csvfile:
        for shape in labelme_data['shapes']:
            if shape['shape_type'] == 'point':
                points = shape['points']
                rectangle = convert_to_rectangle(points)

                fieldnames = ['pic_name', 'x_min', 'y_min', 'x_max', 'y_max']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()

                writer.writerow({
                    'pic_name': os.path.basename(labelme_file),
                    'x_min': rectangle[0],
                    'y_min': rectangle[1],
                    'x_max': rectangle[2],
                    'y_max': rectangle[3]
                })


if __name__ == '__main__':
    json_dir = r''
    csv_dir = r''
    for json in tqdm(os.listdir(json_dir)):
        convert_labelme_to_rectangles(os.path.join(json_dir, json), csv_dir)
