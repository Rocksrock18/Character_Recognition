import csv as csv
import numpy as np

class DataReader():
    def __init__(self):
        pass

    # gets a 2D list of pixels, formatted as [height x width]
    def format_pixels_as_image(self, row, height, width):
        pixels = []
        for i in range(height):
            chunk = []
            for k in range(width):
                chunk.append(float(row[k*width + i]))
            pixels.append(chunk)
        return pixels

    def get_pixels(self, row):
        pixels = np.zeros(784)
        for i, pixel in enumerate(row):
            pixels[i] = float(pixel)/255
        return pixels

    # retrieves the mapping of the character id to the character it represents
    def get_mapping(self):
        mapping = {}
        file_name = "letter_num_dataset/emnist-balanced-mapping.txt"
        with open(file_name) as txt_file:
            for row_num, row in enumerate(txt_file):
                num_digits = len(str(row_num))
                mapping[row[:num_digits]] = int(row[(num_digits+1):row.index("\n")])
        return mapping

    # gets list of format [character_id, [pixels]]
    def get_train_images(self, num_items, img_height, img_width):
        images = []
        file_name = "letter_num_dataset/balanced-subset.csv"
        with open(file_name) as csv_file:
            data_list = list(csv.reader(csv_file))
            num_items = min(num_items, len(data_list))
            for i in range(num_items):
                print("Getting image " + str(i+1) + "/" + str(num_items), end="\r")
                row = data_list[i]
                tuple = (row[0], self.get_pixels(row[1::]))
                images.append(tuple)
        return images

    # gets list of format [character_id, [pixels]]
    def get_test_images(self, num_items, img_height, img_width):
        images = []
        file_name = "letter_num_dataset/mini-subset.csv"
        with open(file_name) as csv_file:
            data_list = list(csv.reader(csv_file))
            num_items = min(num_items, len(data_list))
            for i in range(num_items):
                print("Getting image " + str(i+1) + "/" + str(num_items), end="\r")
                row = data_list[i]
                tuple = (int(row[0]), self.get_pixels(row[1::]))
                images.append(tuple)
        return images
