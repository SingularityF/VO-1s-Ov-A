import pygetwindow as gw
import re
import os


def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def getWindowCo():
    window_name = "VA-11 Hall-A: Cyberpunk Bartender Action"
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


def get_dialog_co(tl, size):
    return {"top": tl.y + int(size.height*.78), "left": tl.x + int(
        size.width*.03), "width": int(size.width * .64), "height": int(size.height * .16)}


def get_resize_factor(size):
    return 1280/size.width


def strip_speaker(text):
    return re.sub("^[\S]+: ", "", text)


def postprocess(text):
    return re.sub("''", "\"", text)
