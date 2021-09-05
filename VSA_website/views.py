# tensorflow version
import tensorflow
# print('tensorflow: %s' % tensorflow.__version__)
# keras version
import keras
# print('keras: %s' % keras.__version__)
from VSA_website.models import Doubt, Video_Upload, Config
from django.shortcuts import render, HttpResponse
from datetime import datetime
import os, re, os.path
import speech_recognition as sr
import ffmpeg
from moviepy.editor import *
from django.contrib import messages
import pytz 
from django.core.files.storage import FileSystemStorage
from .forms import VideoForm
from pydub import AudioSegment
import glob2
import cv2
import csv
import time

import mutagen
from mutagen.wave import WAVE

import random

import mimetypes
mimetypes.init()

import subprocess

from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
from .text_to_emotions import text_to_emotion

from sawo import createTemplate, getContext, verifyToken
from sawo import getContext
import json

# Create your views here.

load = ''
loaded = 0
user_name = ""

overall_review = ""


def setPayload(payload):
    global load
    load = payload

def setLoaded(reset=False):
    global loaded
    if reset:
        loaded=0
    else:
        loaded+=1

createTemplate("templates/partials")
# Create your views here.

number_of_calls_to_index_page = 0
tte = 0
def index(request):    
    global number_of_calls_to_index_page
    global tte
    if(number_of_calls_to_index_page==0):
        tte = text_to_emotion()
        tte.train()
        print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH",number_of_calls_to_index_page)
    else:
        print("ELSE ELSE ELSE ELSEELSE ELSE ELSE "+str(number_of_calls_to_index_page))
    # tte.test('Hello')
    # tte.test('I love you')
    # tte.test('I graduated my class')
    # tte.test('You scare me')
    # tte.test('Do not leave me alone')
    # tte.test('Bring eggs from the market')
    number_of_calls_to_index_page+=1
    if request.method == "POST":
        video_file = request.FILES['video_file']
        video_names = Video_Upload.objects.all()
        length = len(video_names)-1
        if(length>=0):
            video_name = int(video_names[length].video_name) + 1
        else:
            video_name = 0
        # fs = FileSystemStorage()
        # fs.save(str(video_name),video_file)
        video_upload = Video_Upload(video_name = video_name,video_file=video_file)
        video_upload.save()
        messages.success(request, 'Media uploaded successfully!')
        
        video_names = Video_Upload.objects.all()
        length = len(video_names)-1
        video_name = str(video_names[length].video_file)
        video_path = 'media/'+ video_name
        
        global overall_review
        overall_review, csv_file = vid_to_audio(video_path)
        mimestart = mimetypes.guess_type(video_path)[0]

        text_return = []
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                text_return.append(row)

        if mimestart != None:
            mimestart = mimestart.split('/')[0]

            if mimestart == 'audio':
                context = {
                    'play_audio': '../media/'+ video_name,
                    'text_return': text_return,
                    'overall_review': overall_review
                }
            elif mimestart == 'video':
                vid_to_frames(video_path)
                context = {
                    'play_video': '../media/'+ video_name,
                    'text_return': text_return,
                    'overall_review': overall_review
                }
                # image_captioning('media/video_frames/')
            else:
                print("Incorrect type")
                
        
        return render(request, 'index.html', context)

    return render(request, 'index.html')
    # return HttpResponse("This is homepage")

def about(request):
    return render(request, 'about.html')
    # return HttpResponse("This is aboutpage")

def contributors(request):
    return render(request, 'contributors.html')
    # return HttpResponse("This is contributorspage")


def doubt(request):
    if request.method == "POST":
        name = request.POST.get('name')
        email = request.POST.get('email')
        subject = request.POST.get('subject')
        doubt_desc = request.POST.get('doubt_desc')

        doubt = Doubt(name=name, email=email, subject=subject, doubt_desc=doubt_desc, date=datetime.now(pytz.timezone('Asia/Kolkata')).date(), time=datetime.now(pytz.timezone('Asia/Kolkata')).time())
        doubt.save()
        messages.success(request, 'Doubt sent successfully!')
    context ={}
    context["dataset"] = Doubt.objects.all()
    return render(request, 'doubt.html', context)
    # return HttpResponse("This is doubtpage")

def login(request):
    if request.method == "POST":
        sent_emotion = []
        for i in range(10):
            sent_emotion.append([request.POST.get('sent'+str(i+1)),request.POST.get('emotion'+str(i+1)),load['identifier']])

        with open('static/contributors_dataset_complete.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            # writer.writeheader()
            writer.writerows(sent_emotion)

        print(sent_emotion)
        return HttpResponse("<h2 style='text-align:center'>Thankyou for your contribution!</h2>")
        
    setLoaded()
    setPayload(load if loaded<2 else '')
    print(os.environ.get('api_key'))
    config = {
                "auth_key": os.environ.get('api_key'),
                "identifier": "email",
                "to": "receive"
    }
    sentences = []
    
    with open('static/contributors_dataset.csv', 'r') as file:
        reader = csv.reader(file)
        for sent in reader:
            sentences.append(sent[0])
    
    num = random.randint(0,950)
    sentences = sentences[num:num+10]
    context = {"sawo":config,"load":load,"title":"Home","sentences": sentences}
    
    return render(request,"login.html", context)

def receive(request):
    if request.method == 'POST':
        global user_name
        payload = json.loads(request.body)["payload"]
        setLoaded(True)
        setPayload(payload)
        print(payload)
        
        if verifyToken(payload):
            status = 200
            user_list = []
            with open('static/user_details.csv', 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    user_list.append(row[1])

            fieldnames = ['user_id', 'identifier']
            if payload[fieldnames[1]] not in user_list:
                user_name = payload[fieldnames[1]]
                with open('static/user_details.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    # writer.writeheader()
                    writer.writerow([payload[fieldnames[0]],payload[fieldnames[1]]])
            
        else:
             404
        print(status)
        response_data = {"status":status}
        return HttpResponse(json.dumps(response_data), content_type="application/json")

def print_page(request):
    if request.method == 'POST':
        csv_filename_print = "media/dataset_videos_caption_csv/csv_file_print.csv"
        rows_text = []
        with open(csv_filename_print, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    rows_text.append(row)

        
        # images_print = "media/dataset_videos_caption_csv/"

        context = {
                        'rows_text_return': rows_text,
                        # 'rows_images_return': rows_images,
                    }

    
    return render(request, 'print_page.html', context)
    # return render(request, 'print_page.html')


# Functions for performing analysis

# Step A - Performing video to audio conversion then audio to caption then caption emotion analysis
# Step B - Performing video to image conversion (frames extraction) then video frame face detection
#          and facial emotion analysis
# Step C - Performing video to to image conversion (frames extraction) then video frame image captioning
#          (using Flickr8k_Dataset and VGG16 model from keras library for training, will train the model
#           on google colab then will use the trained model for prediction in this project) then 
#          caption emotion analysis
# Step D - Video background music extraction then music emotion analysis
# Step E - Finding the inference from Step A, Step B, Step C and Step D to display an approximate 
#          emotion of the video




#### STEP A ####

# Step A1 - Video to audio conversion
# Added ffmpeg to environment variables path and restart your system

# Step A1 - Part a) - Clearing the existing directories
def clear_directories(paths_list):
    paths = paths_list
    for path in paths:
        print(path)
        for root, dirs, files in os.walk(path):
            for file in files:
                # print(os.path.join(root, file))
                os.remove(os.path.join(root, file))

def vid_to_audio(path):
    # Initially we will remove all the files present in directories so that program
    # does NOT undergo filename conflit- 
    # audio_chunks_5s
    # audio_chunks_10s
    # audio_chunks_10s_with5s_overlapping
    # dataset_video_converted_audio
    # dataset_videos_caption_csv
    paths_list = ["media/audio_chunks_5s","media/audio_chunks_10s","media/audio_chunks_10s_with5s_overlapping","media/dataset_video_converted_audio","media/dataset_videos_caption_csv","media/video_frames"]
    clear_directories(paths_list)


    video_src = path
    # speech_mp3_src = "media/dataset_video_converted_audio/mp3_file.mp3"
    speech_wav_src = "media/dataset_video_converted_audio/wav_file.wav"

    # <--This is Method 1 of converting mp4 to mp3 and then mp3 to wav-->
    # (Not using this method since it requires ffmpeg to be present in windows environment variable
    # path which will not be feasible for a live web app since it should not be system environment 
    # dependent)
    mimestart = mimetypes.guess_type(path)[0]
    if mimestart != None:
        mimestart = mimestart.split('/')[0]

        if mimestart == 'audio':
            # convert mp3 to wav file
            subprocess.call(['ffmpeg', '-i', path,speech_wav_src])
        elif mimestart == 'video':
            # command2mp3 = "ffmpeg -i {} -vn {}".format(video_src,speech_mp3_src)
            command2wav = "ffmpeg -i {} -vn {}".format(video_src,speech_wav_src)
            # os.system(command2mp3)
            os.system(command2wav)

    return audio_chunking(speech_wav_src)
    # <--Method 1 ends-->

    # r = sr.Recognizer()
    # with sr.AudioFile(speech_wav_src) as source:
    #     audio1 = r.record(source, duration=120) 
    # #     audio2 = r.record(source, duration=120) 
    # #     audio3 = r.record(source, duration=120) 
    # #     audio4 = r.record(source, duration=120) 
    # print(r.recognize_google(audio1))
    # # print(r.recognize_google(audio2))
    # # print(r.recognize_google(audio3))
    # # print(r.recognize_google(audio4))

# Step A2 - Chunking(Dividing) audio files into sub-parts
def audio_chunking(path):
    sound = AudioSegment.from_file(path)
    chunk_5_5 = "media/audio_chunks_5s/"
    chunk_10_10 = "media/audio_chunks_10s/"
    chunk_10_10_overlap5 = "media/audio_chunks_10s_with5s_overlapping/"
    paths = [chunk_5_5, chunk_10_10, chunk_10_10_overlap5]

    # 5s - 5s chunking
    i = 0
    c=0
    # 5000ms == 5s
    while((i+5000)<len(sound)):
        chunk = sound[i:(i+5000)]
        chunk.export(chunk_5_5+"audio_chunk_5_5_"+str(c+1000)+".wav", format="wav")
        c+=1
        i+=5000

    chunk = sound[i:]
    chunk.export(chunk_5_5+"audio_chunk_5_5_"+str(c+1000)+".wav", format="wav")


    # 10s - 10s  chunking with 5s overlapping from previous chunk
    i = 0
    c=0
    # 10000ms == 10s
    while((i+10000)<len(sound)):
        chunk = sound[i:(i+10000)]
        chunk.export(chunk_10_10_overlap5+"audio_chunk_10_10_overlap5_"+str(c+1000)+".wav", format="wav")
        c+=1
        i+=5000

    chunk = sound[i:]
    chunk.export(chunk_10_10_overlap5+"audio_chunk_10_10_overlap5_"+str(c+1000)+".wav", format="wav")


    # 10s - 10s chunking
    i = 0
    c=0
    # 10000ms == 10s
    while((i+10000)<len(sound)):
        chunk = sound[i:(i+10000)]
        chunk.export(chunk_10_10+"audio_chunk_10_10_"+str(c+1000)+".wav", format="wav")
        c+=1
        i+=10000

    chunk = sound[i:]
    chunk.export(chunk_10_10+"audio_chunk_10_10_"+str(c+1000)+".wav", format="wav")

    return audio_to_caption_emotion(paths)

# Step A3 - Captioning the audio and (displaying the same - this will not be the part of final release)
# and
# Step A4 - Audio caption emotion analysis using a trained ML model and storing the results in a csv
def audio_to_caption_emotion(paths):
    # files = []
    rows = [] # For inserts rows in a csv
    rows_print = [] # For inserts rows in a csv
    tags = ["5s-5s","10s-10s","10s-5s"]
    duration_increment = [5,10,5]
    percent_div = [20,30,50] # Total = 100%
    idx = 0
    overall_emotion_dict = {
            'Joy': 0,
            'Fear': 0,
            'Anger': 0,
            'Sadness': 0,
            'Neutral': 0
        }
    count = 1
    for path in paths:
        file_list = glob2.glob(path+"*")
        # files += file_list
        # print(files)
        duration_start = 0
        for i in file_list:
            duration,caption = captioning(i)
            starttime = time_to_hms(duration_start)
            duration = duration+duration_start
            endtime = time_to_hms(duration)
            if(caption=="Speech Recognition could not understand audio"):
                rows_print.append([count,starttime+"-"+endtime,caption,"Neutral"])
                rows.append([starttime+"-"+endtime,"Neutral"])
                overall_emotion_dict["Neutral"] += percent_div[idx] 
            else:
                emo = tte.test(caption)
                rows_print.append([count,starttime+"-"+endtime,caption,emo])
                rows.append([starttime+"-"+endtime,emo])
                overall_emotion_dict[emo] += percent_div[idx] 
            duration_start+=duration_increment[idx]
            count+=1
        idx+=1
    
    # print(rows)
    csv_filename = "media/dataset_videos_caption_csv/csv_file.csv"
    csv_filename_print = "media/dataset_videos_caption_csv/csv_file_print.csv"
    
    with open(csv_filename, 'w', newline='') as csvfile: 
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerows(rows)

    with open(csv_filename_print, 'w', newline='') as csvfile: 
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerows(rows_print)
    
    print(overall_emotion_dict)
    keymax = max(zip(overall_emotion_dict.values(), overall_emotion_dict.keys()))[1]
    valuemax = max(zip(overall_emotion_dict.values(), overall_emotion_dict.keys()))[0]
    print(keymax)

    nextvaluemax = 0
    nextkeymax = ""
    for i in zip(overall_emotion_dict.values(), overall_emotion_dict.keys()):
        if(i[0]<=valuemax and i[0]>=(valuemax-20) and i[1]!=keymax):
            if(i[0]>nextvaluemax):
                nextvaluemax = i[0]
                nextkeymax = i[1]

    key = keymax
    if (nextkeymax!=""):
        key+=" & "+nextkeymax+" "

    return key, csv_filename

def captioning(path):
    try:
        AUDIO_FILE = (path)
        audio_file_name = WAVE(path)
  
        # contains all the metadata about the wavpack file
        audio_info = audio_file_name.info
        length = int(audio_info.length)

        # using the audio file as the audio source
        r = sr.Recognizer()

        with sr.AudioFile(AUDIO_FILE) as source:
            audio = r.record(source)
        return(length,r.recognize_google(audio))

    except sr.UnknownValueError:
        return(length,"Speech Recognition could not understand audio")

    except sr.RequestError as e:
        return("Could not request results from Speech Recognition service; {0}".format(e))

    except Exception as e:
        return(e)

    
def time_to_hms(length):
    hours = length // 3600  # calculate in hours
    length %= 3600
    mins = length // 60  # calculate in minutes
    length %= 60
    seconds = length  # calculate in seconds
  
    return (("%02d" % hours)+":"+("%02d" % mins)+":"+("%02d" % seconds))  # returns the duration







#### STEP B ####

# Step B1 - Capturing frames of the video 
def vid_to_frames(path):
    vidcap = cv2.VideoCapture(path)
    count = 0
    success = True
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    f_count = 1000
    
    # Frame capture after every 5s
    while success:
        success,image = vidcap.read()
        if count%(10*fps) == 0 :
            cv2.imwrite('media/video_frames/frame_%d.jpg'%f_count,image)
            f_count+=1
#              print('successfully written 5th frame')
        count+=1


# # generate a description for an image (Helper function for code of image captionong)
# def generate_desc(model, tokenizer, photo, max_length):
# 	# seed the generation process
# 	in_text = 'startseq'
# 	# iterate over the whole length of the sequence
# 	for i in range(max_length):
# 		# integer encode input sequence
# 		sequence = tokenizer.texts_to_sequences([in_text])[0]
# 		# pad input
# 		sequence = pad_sequences([sequence], maxlen=max_length)
# 		# predict next word
# 		yhat = model.predict([photo,sequence], verbose=0)
# 		# convert probability to integer
# 		yhat = argmax(yhat)
# 		# map integer to word
# 		word = word_for_id(yhat, tokenizer)
# 		# stop if we cannot map the word
# 		if word is None:
# 			break
# 		# append as input for generating the next word
# 		in_text += ' ' + word
# 		# stop if we predict the end of the sequence
# 		if word == 'endseq':
# 			break
# 	return in_text

# def word_for_id(integer, tokenizer):
# 	for word, index in tokenizer.word_index.items():
# 		if index == integer:
# 			return word
# 	return None

# def extract_features(filename):
# 	# load the model
# 	model = VGG16()
# 	# re-structure the model
# 	model.layers.pop()
# 	model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
# 	# load the photo
# 	image = load_img(filename, target_size=(224, 224))
# 	# convert the image pixels to a numpy array
# 	image = img_to_array(image)
# 	# reshape data for the model
# 	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# 	# prepare the image for the VGG model
# 	image = preprocess_input(image)
# 	# get features
# 	feature = model.predict(image, verbose=0)
# 	return feature

# def image_captioning(image_dir):
#     # load the tokenizer
#     tokenizer = load(open('static/tokenizer.pkl', 'rb'))
#     # pre-define the max sequence length (from training)
#     max_length = 34
#     # load the model
#     model = load_model('static/model_18.h5')

#     for image_path in glob2.glob(image_dir+"*"):
#         # load and prepare the photograph
#         photo = extract_features(image_path)
#         # generate description
#         description = generate_desc(model, tokenizer, photo, max_length)
#     #     print(description)

#         #Remove startseq and endseq
#         query = description
#         stopwords = ['startseq','endseq']
#         querywords = query.split()

#         resultwords  = [word for word in querywords if word.lower() not in stopwords]
#         result = ' '.join(resultwords)

#         print(str(image_path)+result)
#         # print(str(image_path))