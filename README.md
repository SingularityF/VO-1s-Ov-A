# VO-1s-Ov-A
Pronounced as "Voice Over", in British accent🤔. This is a voice over "mod" for "VA-11 Hall-A" using OCR and TTS, achieved by a CNN model trained by hand-labelled data and Amazon Polly. AWS offers a "Free Tier" for Amazon Polly so this voice-over "mod" should be free for the most part. 

[**You can watch a video of the "mod" in action here**](https://youtu.be/2dIXSJq-nPg).

# Behind-the-scenes
I am very bad at imagining stuff (in scientific term, only half of my brain is well developed). I can barely see with my mind's eye and hear with my mind's ear. Therefore I've always hated reading nonfiction books. Growing up I found Japanese visual novels to be interesting because it's a solution to both of my problems. The game "VA-11 Hall-A" is a visual novel with a very interesting premise and setting, however it lacks voice acting. And, for me, that made the gameplay very difficult. Without any voice I just can't picture myself immersed in the world the game's trying to set. Luckily for me, making a voice over feature is just a walk in the park.

# Instuctions
1. Install python, app is tested with Python 3.8.0
2. Install dependencies `pip install -r requirements.txt`
3. Create an AWS account if not exist
4. Set up a new user in AWS IAM, grant it `PollyFullAccess`
5. Download csv credentials for the user you created and put it in this folder, name it `credentials.csv`
6. Adjust configs in `configs.py`
7. Run the app with `python main.py`
8. Press `F9` to pause and unpause the app. Exit the app with `Ctrl+c`

# Model Training
You can use the trained models under the `models` folder, the app will work straight out of the box. You can also choose to train your own model.
1. Run the app with `python main.py`
2. Press `F8` to save screenshots
3. Run label-studio with `python label-studio`
4. Upload images from the `saved_img` folder and hand label the saved screenshots
5. Create a new folder here, name it `unlabeled_data`
6. Export labeled data in json, copy it here
7. Copy images uploaded to label-studio from `???/AppData/Local/label-studio/label-studio/media/upload` to the `unlabeled_data` folder
8. Run `prepare_dataset.py`
9. Run `train.py`
