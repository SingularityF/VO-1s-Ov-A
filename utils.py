import pygetwindow as gw
import re
import os
import libs.gibberish_detector.gib_detect_train
from preproc import *
import numpy as np
from pathlib import Path
from configs import *
import json
import glob


def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def getWindowCo(window_name):
    game_window = gw.getWindowsWithTitle(window_name)[0]
    return (game_window.topleft, game_window.size)


def reconstruct(chars, input):
    idx = 0
    predicted_dialog = []
    for line in input:
        predicted_line = []
        for word in line:
            predicted_word = []
            for char in word:
                predicted_word.append(chars[idx])
                idx += 1
            predicted_word = "".join(predicted_word)
            predicted_line.append(predicted_word)
        predicted_line = " ".join(predicted_line)
        predicted_dialog.append(predicted_line)
    predicted_dialog = "\n".join(predicted_dialog)
    return predicted_dialog


def get_resize_factor(size):
    return 1280/size.width


def strip_speaker(text):
    try:
        speaker = re.findall("^([\S]+): ", text)[0]
    except:
        speaker = ""
    return speaker, re.sub("^[\S]+: ", "", text)


def postprocess(text):
    return re.sub("''", "\"", text)


def find_voice_actor(speaker):
    return "Justin"


def detect_gibbrish(text, gib_model):
    model_mat = gib_model['mat']
    threshold = gib_model['thresh']
    return not libs.gibberish_detector.gib_detect_train.avg_transition_prob(text, model_mat) > threshold


def punct_dominant(text):
    letter_count = sum(bool(re.search("[a-zA-Z]", char)) for char in text)
    return letter_count / len(text) < .5


def get_word_spacing(img, line_no, word_count, resize_factor, threshold, prompt_roi):
    img = preproc_img_color(img, resize_factor)
    bw_img = preproc_img_bw(img, threshold, prompt_roi)
    line_boundary = line_detector(bw_img)
    x, y, w, h = line_boundary[line_no-1]
    first_line = bw_img[y:y+h, x:x+w]
    count_contour = []
    for word_spacing in range(1, 100):
        count_contour.append(len(word_detector(first_line, word_spacing)))
    return (np.argmax([int(count == word_count) for count in count_contour]) + 2)


def save_preset(file_name, window_name, dialog_roi, prompt_roi, threshold, word_spacing, max_width):
    file_path = str(Path(presets_path, file_name+".json"))
    preset_dict = {"window_name": window_name,
                   "dialog_roi": dialog_roi,
                   "prompt_roi": prompt_roi,
                   "threshold": threshold,
                   "word_spacing": int(word_spacing),
                   "max_width": int(max_width)
                   }
    with open(file_path, "w") as f:
        json.dump(preset_dict, f)


def load_presets():
    presets = glob.glob(str(Path(presets_path, "*.json")))
    if len(presets) != 0:
        print("Presets found, choose a preset from below")
        for idx, preset in enumerate(presets):
            with open(preset, "r") as f:
                preset_dict = json.load(f)
            print(
                f"{idx}): {str(Path(preset).name)} ({preset_dict['window_name']})")
        print("x): Don't use a preset")
        resp = input().strip()
        if resp == "x":
            return "rejected"
        else:
            try:
                preset = presets[int(resp)]
                with open(preset, "r") as f:
                    preset_dict = json.load(f)
                window_name = preset_dict["window_name"]
                dialog_roi = preset_dict["dialog_roi"]
                prompt_roi = preset_dict["prompt_roi"]
                word_spacing = preset_dict["word_spacing"]
                threshold = preset_dict["threshold"]
                max_width = preset_dict["max_width"]
                return window_name, dialog_roi, prompt_roi, threshold, word_spacing, max_width
            except:
                print("Loading preset failed, try again")
                return "repeat"
    else:
        return "rejected"


def get_character_width(img, word_spacing, resize_factor, threshold, prompt_roi):
    max_width = 0
    max_img = None
    img = preproc_img_color(img, resize_factor)
    bw_img = preproc_img_bw(img, threshold, prompt_roi)
    line_boundary = line_detector(bw_img)
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
                if max(max_width, w) == w:
                    max_width = w
                    max_img = char
    return max_width, max_img
