import cv2
import re
import traceback
import json
import sys
from preproc import *
from pathlib import Path
from configs import *
from utils import *


def make_dataset(labels, file_names, dataset_name):
    dataset_name = str(Path(dataset_folder_path, dataset_name))
    writer = tf.io.TFRecordWriter(dataset_name)
    for i, label in enumerate(labels):
        img = cv2.imread(str(Path(img_folder_path, file_names[i])))
        bw_img = preproc_img_bw(img, threshold, prompt_roi)
        char_label = re.sub("\s", "", label)
        line_boundary = line_detector(bw_img)
        idx = 0
        examples = []
        debug_buffer = []
        for x, y, w, h in line_boundary:
            line = bw_img[y:y+h, x:x+w]
            word_boundary = word_detector(line, word_spacing)
            for x, y, w, h in word_boundary:
                word = line[y:y+h, x:x+w]
                char_boundary = character_detector(word)
                for x, y, w, h in char_boundary:
                    char = word[y:y+h, x:x+w]
                    if filter_noise(char):
                        continue
                    debug_buffer.append(char)
                    img_str = char.tobytes()
                    try:
                        features = {
                            'height': tf_int64_feature(h),
                            'width': tf_int64_feature(w),
                            'char_img': tf_bytes_feature(img_str),
                            'label': tf_bytes_feature(str.encode(char_label[idx])),
                        }
                        example = tf.train.Example(
                            features=tf.train.Features(feature=features))
                    except:
                        # traceback.print_exc()
                        pass
                    examples.append(example)
                    idx += 1
        if len(examples) == len(char_label):
            for example in examples:
                writer.write(example.SerializeToString())
        else:
            print("==========")
            print("Possible errorneous label, data excluded for training")
            print(char_label)
            # for img in debug_buffer:
            #     cv2.imshow("", img)
            #     cv2.waitKey(0)


def prepare_data(annotation_path, dataset_name):
    with open(annotation_path, "r") as f:
        label_data = json.load(f)
        labels = []
        file_names = []
        for item in label_data:
            label = item["annotations"][0]["result"][0]["value"]["text"][0]
            file_name = item["file_upload"]
            labels.append(label)
            file_names.append(file_name)
        make_dataset(labels, file_names, dataset_name)


if __name__ == "__main__":
    if not len(sys.argv) == 3:
        print("Error - wrong command line arguments")
        print("Usage: python prepare_dataset.py input_data.json output_data.tfrecords")
    else:
        resp = load_presets()
        while resp == "repeat":
            resp = load_presets()
        if resp == "rejected":
            print("You must use a preset!")
            exit()
        else:
            _, _, prompt_roi, threshold, word_spacing, max_width = resp
        path_input = sys.argv[1]
        path_output = sys.argv[2]
        prepare_data(path_input, path_output)
