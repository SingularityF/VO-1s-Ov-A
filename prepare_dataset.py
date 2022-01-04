import cv2
import re
import traceback
import json
import sys
from preproc import *
from pathlib import Path


def make_dataset(labels, file_names, dataset_name):
    writer = tf.io.TFRecordWriter(dataset_name)
    for i, label in enumerate(labels):
        img = cv2.imread(str(Path(img_folder_path, file_names[i])))
        bw_img = thresholding(img)
        char_label = re.sub("\s", "", label)
        line_boundary = line_detector(bw_img)
        idx = 0
        for x, y, w, h in line_boundary:
            line = bw_img[y:y+h, x:x+w]
            word_boundary = word_detector(line)
            for x, y, w, h in word_boundary:
                word = line[y:y+h, x:x+w]
                char_boundary = character_detector(word)
                for x, y, w, h in char_boundary:
                    char = word[y:y+h, x:x+w]
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
                        writer.write(example.SerializeToString())
                    except:
                        print("Possible errorneous label:")
                        print(char_label)
                        traceback.print_exc()
                    idx += 1


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
    # Configs
    img_folder_path = "./unlabeled_data"
    if not len(sys.argv) == 3:
        print("Error - wrong command line arguments")
        print("Usage: python prepare_dataset.py input_data.json ./datasets/output_data.tfrecords")
    else:
        path_input = sys.argv[1]
        path_output = sys.argv[2]
        prepare_data(path_input, path_output)
