import cv2
import pandas
import numpy as np
from Config.DatasetConfig import plate_format, image_width, image_height, blank_num
import random


def create_random_characters():

    blanks = []
    for i in range(blank_num):
        blanks.append(random.randint(1, len(plate_format)-2))

    plate_word = ""
    for char in plate_format:
        if char == "$":
            c = chr(random.randrange(65,90))
            plate_word += c

        if char == '#':
            n = f"{int(random.randint(1,9))}"
            plate_word += n


    for blank in blanks:
        plate_word = plate_word[:blank] + " " + plate_word[blank:]

    # print(plate_word)
    return plate_word

def synthesize_license():
    word = create_random_characters()

    plate = np.ones((image_height, image_width, 3))*255
    plate = cv2.putText(plate, word, (80,80), fontScale=2, fontFace=3, color=(0, 0, 0), thickness=2)
    plate = np.float32(plate)
    cv2.imwrite("plate.jpg", plate)

    return word, plate

