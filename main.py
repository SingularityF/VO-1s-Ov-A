print("Initializing...")
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils import *
from pathlib import Path
import numpy as np
from preproc import *
from configs import *
import pandas as pd
import pygame
import uuid
import boto3
import keyboard
import cv2
import time
import mss
import sys


def process(img, resize_factor):
    img = preproc_img_color(img, resize_factor)
    bw_img = thresholding(img)
    line_boundary = line_detector(bw_img)
    dialog_input = []
    for x, y, w, h in line_boundary:
        line = bw_img[y:y+h, x:x+w]
        word_boundary = word_detector(line)
        line_input = []
        for x, y, w, h in word_boundary:
            word = line[y:y+h, x:x+w]
            char_boundary = character_detector(word)
            word_input = []
            for x, y, w, h in char_boundary:
                char = word[y:y+h, x:x+w]
                input = char_img2arr(char)
                word_input.append(input)
            line_input.append(word_input)
        dialog_input.append(line_input)
    input_flattened = [
        char for line in dialog_input for word in line for char in word]
    predicted = model.predict(np.array(input_flattened)).tolist()
    predicted = [chr(np.argmax(char)) for char in predicted]
    predicted_dialog = reconstruct(predicted, dialog_input)
    return predicted_dialog


def pause_detection():
    global pause_flag
    pause_flag = not pause_flag
    if pause_flag:
        print("==========")
        print("Detection paused")
    else:
        print("Detection started")
        print("==========")


def tts(text):
    pygame.mixer.music.unload()
    response = polly_client.synthesize_speech(VoiceId='Ivy',
                                              OutputFormat='mp3',
                                              Text=text,
                                              Engine='neural')
    file = open('speech.mp3', 'wb')
    file.write(response['AudioStream'].read())
    file.close()
    pygame.mixer.music.load('speech.mp3')
    pygame.mixer.music.play()


def save_img():
    with mss.mss() as sct:
        try:
            tl, size = getWindowCo()
        except:
            print("Game window not found")
            return
        dialog = get_dialog_co(tl, size)
        img = np.array(sct.grab(dialog))
        img = preproc_img_color(img, get_resize_factor(size))
        mkdir_if_not_exist(img_output_path)
        cv2.imwrite(str(Path(img_output_path, str(uuid.uuid4())+".bmp")), img)
        print("Image saved")


def main():
    predicted_prev = ""
    time_prev = time.time()
    flag_change = False
    print("Detecting input...")
    with mss.mss() as sct:
        while True:
            elapsed_time = time.time() - time_prev
            if pause_flag or not elapsed_time > loop_time:
                time.sleep(sleep_time)
                continue
            time_prev = time.time()
            start_time = time.time()
            try:
                tl, size = getWindowCo()
            except:
                print("Game window not found")
                time.sleep(3)
                continue
            dialog = get_dialog_co(tl, size)
            img = np.array(sct.grab(dialog))
            try:
                predicted_dialog = process(
                    img, resize_factor=get_resize_factor(size))
                predicted_dialog = postprocess(predicted_dialog)
            except:
                continue
            if not predicted_dialog == predicted_prev:
                flag_change = True
            else:
                if flag_change:
                    print(predicted_dialog)
                    print()
                    if not debug_flag:
                        tts(strip_speaker(predicted_dialog))
                flag_change = False
            predicted_prev = predicted_dialog
            end_time = time.time()
            #print(f"Loop took {end_time - start_time} seconds")


if __name__ == "__main__":
    debug_flag = False
    pause_flag = False
    try:
        if sys.argv[1] == "debug":
            debug_flag = True
    except:
        pass
    model = tf.keras.models.load_model(model_path)
    df = pd.read_csv(credentials_file)
    key_id = df["Access key ID"][0]
    secret = df["Secret access key"][0]
    polly_client = boto3.Session(
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
        region_name=aws_region).client('polly')
    pygame.mixer.init()
    keyboard.add_hotkey('f8', save_img)
    keyboard.add_hotkey('f9', pause_detection)
    main()
