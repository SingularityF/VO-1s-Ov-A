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
import traceback
import pickle


def process(img, resize_factor):
    img = preproc_img_color(img, resize_factor)
    bw_img= preproc_img_bw(img, threshold, prompt_roi)
    # cv2.imshow("", bw_img)
    # cv2.waitKey(0)
    line_boundary = line_detector(bw_img)
    dialog_input = []
    for x, y, w, h in line_boundary:
        line = bw_img[y:y+h, x:x+w]
        word_boundary = word_detector(line, word_spacing)
        line_input = []
        for x, y, w, h in word_boundary:
            word = line[y:y+h, x:x+w]
            char_boundary = character_detector(word)
            word_input = []
            for x, y, w, h in char_boundary:
                char = word[y:y+h, x:x+w]
                if filter_noise(char):
                    continue
                input = char_img2arr(char)
                word_input.append(input)
            if len(word_input) != 0:
                line_input.append(word_input)
        if len(line_input)!= 0:
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


def tts(text, actor):
    pygame.mixer.music.unload()
    response = polly_client.synthesize_speech(VoiceId=actor,
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
            tl, size = getWindowCo(window_name)
        except:
            print("Game window not found")
            return
        dialog_co = {"top": tl.y + int(dialog_roi[1]*size.height), "left": tl.x + int(dialog_roi[0]*size.width), "width": int(dialog_roi[2]*size.width), "height": int(dialog_roi[3]*size.height)}
        img = np.array(sct.grab(dialog_co))
        img = preproc_img_color(img, get_resize_factor(size))
        mkdir_if_not_exist(img_output_path)
        cv2.imwrite(str(Path(img_output_path, str(uuid.uuid4())+".bmp")), img)
        print("Image saved")


def main():
    global flag_change, roi_set, thresh_set, word_spacing, word_spacing_set, threshold, dialog_roi, preset_saved, max_width_set, max_width, prompt_roi, mask_set
    predicted_prev = ""
    time_prev = time.time()
    print("Detecting input...")
    with mss.mss() as sct:
        while True:
            elapsed_time = time.time() - time_prev
            if pause_flag or not elapsed_time > refresh_time:
                time.sleep(loop_time)
                continue
            time_prev = time.time()
            start_time = time.time()
            try:
                tl, size = getWindowCo(window_name)
            except:
                print("Game window not found")
                time.sleep(wait_time)
                continue
            #################################
            # Set dialog region of interest #
            #################################
            if not roi_set:
                window = {"top": tl.y, "left": tl.x, "width": size.width, "height": size.height}
                img = np.array(sct.grab(window))
                dialog_roi = cv2.selectROI(img, False)
                cv2.destroyAllWindows() 
                if all(x==0 for x in dialog_roi):
                    print("Waiting for next attempt")
                    time.sleep(wait_time)
                else:
                    dialog_roi = [dialog_roi[0]/size.width, dialog_roi[1]/size.height, dialog_roi[2]/size.width, dialog_roi[3]/size.height]
                    roi_set = True
                continue
            else:
                dialog_co = {"top": tl.y + int(dialog_roi[1]*size.height), "left": tl.x + int(dialog_roi[0]*size.width), "width": int(dialog_roi[2]*size.width), "height": int(dialog_roi[3]*size.height)}
                img = np.array(sct.grab(dialog_co))      
            #################
            # Set threshold #
            #################
            if not thresh_set:
                def on_change_b(val):
                    if val < 3:
                        return
                    val = val + val % 2 - 1
                    global threshold
                    threshold["block_size"] = val
                    bw_img = preproc_img_bw(img, threshold)
                    cv2.imshow("Threshold selector", bw_img)
                def on_change_c(val):
                    global threshold
                    threshold["const"] = val
                    bw_img = preproc_img_bw(img, threshold)
                    cv2.imshow("Threshold selector", bw_img)
                cv2.imshow("Threshold selector", img)
                cv2.createTrackbar('Block Size', "Threshold selector", 3, 255, on_change_b)
                cv2.createTrackbar('Constant', "Threshold selector", 0, 255, on_change_c)
                key = cv2.waitKey(0) & 0xFF
                cv2.destroyAllWindows()
                if key == 10 or key == 13:
                    thresh_set = True
                    print(f"Threshold set: {threshold}")
                else:
                    print("Waiting for next attempt")
                    time.sleep(wait_time)
                continue
            ###################
            # Set prompt mask #
            ###################
            if not mask_set:
                prompt_roi = cv2.selectROI(img, False)
                cv2.destroyAllWindows() 
                if all(x==0 for x in prompt_roi):
                    print("Waiting for next attempt")
                    time.sleep(wait_time)
                else:
                    dialog_width = dialog_co["width"]
                    dialog_height = dialog_co["height"]
                    prompt_roi = [prompt_roi[0]/dialog_width, prompt_roi[1]/dialog_height, prompt_roi[2]/dialog_width, prompt_roi[3]/dialog_height]
                    mask_set = True
                continue

            ####################
            # Set word spacing #
            ####################
            if not word_spacing_set:
                cv2.imshow("Is this a good image for telling word spacing? (ENTER/ESC)", img)
                key = cv2.waitKey(0) & 0xFF
                cv2.destroyAllWindows()
                if key == 10 or key == 13:
                    print("Which line do you want to use for calculation? (1/2/3/...)")
                    line_no = input().strip()
                    try:
                        int(line_no)
                    except:
                        print("Try again")
                        continue
                    if line_no == '1':
                        print(f"How many words are shown on the {line_no}st line?")
                    elif line_no == '2':
                        print(f"How many words are shown on the {line_no}nd line?")
                    elif line_no == '3':
                        print(f"How many words are shown on the {line_no}rd line?")
                    else:
                        print(f"How many words are shown on the {line_no}th line?")
                    word_count = input().strip()
                    print("Calculating word spacing")
                    word_spacing = get_word_spacing(img, int(line_no), int(word_count), get_resize_factor(size), threshold, prompt_roi)
                    print(f"Word spacing set: {word_spacing}")
                    word_spacing_set = True
                else:
                    print("Waiting for next attempt")
                    time.sleep(wait_time)
                continue
            ###########################
            # Set max character width #
            ###########################
            if not max_width_set:
                cv2.imshow("Is this a good image for telling max character width? (ENTER/ESC)", img)
                key = cv2.waitKey(0) & 0xFF
                cv2.destroyAllWindows()
                if key == 10 or key == 13:
                    max_width, max_img = get_character_width(img, word_spacing, get_resize_factor(size), threshold, prompt_roi)
                    print("Is this character an exemplary single wide character? (ENTER/ESC)")
                    cv2.imshow("Is this character an exemplary single wide character? (ENTER/ESC)", max_img)
                    key = cv2.waitKey(0) & 0xFF
                    cv2.destroyAllWindows()
                    if key == 10 or key == 13:
                        print(f"Max character width set: {max_width}")
                        max_width_set = True
                    else:
                        print("Waiting for next attempt")
                        time.sleep(wait_time)
                else:
                    print("Waiting for next attempt")
                    time.sleep(wait_time)
                continue
            ###############
            # Save preset #
            ###############
            if not preset_saved:
                print("Do you want to save this preset? (Y/N)")
                resp = input().strip()
                if resp=="Y":
                    print("Give the preset a name: ")
                    resp = input().strip()
                    try:
                        save_preset(resp, window_name, dialog_roi, prompt_roi, threshold, word_spacing, max_width)
                        preset_saved = True
                    except:
                        print("Failed saving preset, try again")
                        continue
                elif resp=="N":
                    preset_saved = True
                else:
                    continue
            ##############
            # Main logic #
            ##############
            try:
                predicted_dialog = process(
                    img, get_resize_factor(size))
                predicted_dialog = postprocess(predicted_dialog)
            except:
                #traceback.print_exc()
                continue
            # Don't voice when dialog is being displayed
            if predicted_dialog != predicted_prev:
                predicted_prev = predicted_dialog
                flag_change = True
            else:
                predicted_prev = predicted_dialog
                if flag_change:
                    flag_change = False
                    print(predicted_dialog)
                    if len(predicted_dialog.strip()) < 2:
                        print("Dialog too short")
                        print()
                        continue
                    if punct_dominant(predicted_dialog):
                        print("Too much punctuation")
                        print()
                        continue
                    if detect_gibbrish(predicted_dialog.replace("\n"," "), gib_model):
                        print("Gibbrish detected")
                        print()
                        continue
                    if not debug_flag:
                        speaker, dialog = strip_speaker(predicted_dialog)
                        actor = find_voice_actor(speaker)
                        tts(dialog, actor)
                        print()
                else:
                    flag_change = False
            end_time = time.time()
            #print(f"Loop took {end_time - start_time} seconds")




if __name__ == "__main__":
    # Control variables
    debug_flag = False
    pause_flag = False
    # Main variables
    flag_change = False
    roi_set = False
    thresh_set = False
    word_spacing_set = False
    preset_saved = False
    max_width_set = False
    mask_set = False
    # Preset variables
    dialog_roi = None
    prompt_roi = None
    word_spacing = 0
    max_width = 0
    threshold = {"const": 0,
                 "block_size": 0}
    window_name = ""
    
    resp = load_presets()
    while resp == "repeat":
        resp = load_presets()
    if resp == "rejected":
        print("Enter window name (can be a substring): ")
        window_name = input().strip()
    else:
        window_name, dialog_roi, prompt_roi, threshold, word_spacing, max_width = resp
        roi_set = True
        mask_set = True
        thresh_set = True
        word_spacing_set = True
        max_width_set = True
        preset_saved = True
    try:
        if sys.argv[1] == "debug":
            debug_flag = True
    except:
        pass
    gib_model = pickle.load(open('./libs/gibberish_detector/gib_model.pki', 'rb'))
    model = tf.keras.models.load_model(model_path)
    df = pd.read_csv(credentials_file)
    key_id = df["Access key ID"][0]
    secret = df["Secret access key"][0]
    polly_client = boto3.Session(
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
        region_name=aws_region).client('polly')
    pygame.mixer.init()
    keyboard.add_hotkey(capture_key, save_img)
    keyboard.add_hotkey(pause_key, pause_detection)
    keyboard.add_hotkey(exit_key, exit)
    main()
