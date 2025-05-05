# import all required libraries

import noisereduce as nr # for noise removal methods
import soundfile as sf # for reading and writing audio files
import numpy as np # for any math operation ever
import webrtcvad # for speech detection
import sounddevice as sd # for audio playback
import time #  for measuring latency of program
import librosa # for audio processing 

# the following libraries for running the neural network
import tensorflow as tf 
from keras.utils import np_utils, to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import keras
from tensorflow.keras.models import load_model
import tkinter

# the following libraries are for the GUI
import customtkinter
from tkinter import ttk, filedialog
from tkinter.filedialog import askopenfile

#  start_time = time.process_time_ns() # start timer for measuring runtime

def loadaudio(): # reads the audio file for processing
	file = filedialog.askopenfile(mode='r', filetypes=[('Audio Files', '*.wav')]) #opens a file window for browsing and selecting files
	if file: # if a file is selected
		filepath = str(file.name) # saves the file path of the selected file
	return filepath

def readaudio(): # reads the audio file for processing
	filepath = loadaudio() # returns the file path
	data, rate = sf.read(filepath) # reads the audio file
	return data, rate # outputs the wav data and sample rate

def playaudio(data, rate): # plays the audio in real time.This function is only for testing, not for use in production
	sd.play(data, rate) # Audio playback
	time.sleep(5.5) # no other command runs while the audio is playing
	sd.stop() #stops playback after the sleep time

def record(time, rate): # records an audio file and stores it as a variable
	duration = time  # no. of seconds
	myrecording = sd.rec(int(duration * rate), samplerate=rate, channels = 1, blocking = True) #records the audio in real time
	return myrecording, rate

def detectspeech(data): # WebRTC voice activity detector detects speech in an audio file and returns true if speech is detected
	vad = webrtcvad.Vad() # Creates a voice activity detector (VAD) object which has a mode attribute and speech detection method
	vad.set_mode(3) # most aggressive speech detection mode
	sample_rate = 16000 # sample rate must be same or a factor of that of data
	frame_duration = 20  # ms
	n = int(sample_rate * (frame_duration/ 1000.0) * 2) # no. of samples
	duration = (float(n) / sample_rate) / 2.0 # time duration of one frame
	frames =  list() # defines a list for the audio frames
	i = 0
	while i+n< len(data): # for iteratively adding the audio data to the list of frames
		frame = data[i:i+n] # dividing audio data into frames
		frames.append(frame) # adding the created frame to the list of frames
		i+=n
	  
	triggered = False
	for frame in frames: # run iteratively for all frames
		is_speech = vad.is_speech(frame, sample_rate) # detects speech in a frame
		if is_speech: # if speech is detected
			triggered = True # raise trigger flag
	return triggered # if even one frame has speech, returns true, else returns false

def remove_stationary_noise(data,rate):  # function for removing stationary or broadband noise
	reduced_noise = nr.reduce_noise(y = data.T, sr=rate, n_std_thresh_stationary=1.0, n_fft = 1024, stationary = True) # removes stationary noise from the signal
	#print('speech after removing stationary noise')
	return reduced_noise # returns clean audio

'''

def remove_non_stationary_noise(data,rate): # removes non-stationary noise from the signal
	reduced_noise = nr.reduce_noise(y = data.T, sr=rate, n_fft = 1024, stationary = False)
	print('speech after removing non stationary noise')
	return reduced_noise

'''

def removenoise(audio, rate): # combined function for speech processing
	#playaudio(audio, rate)
	#print('sound played')
	if detectspeech(audio): # call function for detecting speech. Process audio only when it contains speech
		#print('voice detected')
		rmstat = remove_stationary_noise(audio, rate) # call function to remove the stationary noise from audio
		#playaudio(rmstat, rate)
		#sf.write('nostat.wav', np.ravel(rmstat), rate)
		#rmnonstat = remove_non_stationary_noise(audio, rate)
		#playaudio(rmnonstat, rate)	
		#sf.write('rmnonstat.wav', np.ravel(rmnonstat), rate)
		return rmstat # returns clean speech
	else:
		print(audio.shape)
		print('No speech detected. Please upload speech only')

def prepare_data_upload(n, sampling_rate, audio_duration):
	filepath = loadaudio() # call function for returning the filepath of the selected file
	data, sr = librosa.load(path = filepath, sr = sampling_rate, res_type="kaiser_fast", duration=audio_duration, offset=0.5) # loads the data file from the file path
	data = removenoise(data,sr) # call function to remove stationary noise
	X = np.empty(shape=(1, n, 108, 1)) # create a 4 dimenional array as input to the neural network
	print('data shape', data.shape) 
	cnt = 0
	input_length = sampling_rate * audio_duration # Random offset / Padding
	if len(data) > input_length: # this if-else ladder equalizes and formats the audio files which are not the required size
		max_offset = len(data) - input_length
		offset = np.random.randint(max_offset)
		data = data[offset:(input_length+offset)]
	else:
		if input_length > len(data):
			max_offset = input_length - len(data)
			offset = np.random.randint(max_offset)
		else:
			offset = 0
		data = np.pad(data, (offset, int(input_length) - len(data) - offset), "constant")
	MFCC = librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc= n, n_fft = 4096, hop_length = 512) # MFCC extraction
	print('MFCC before expanding dimensions', MFCC.shape)
	MFCC = np.expand_dims(MFCC, axis=-1) # expand the dimensions from 2 to 3
	print('MFCC after expanding dimensions', MFCC.shape)
	X[cnt,] = MFCC # save the MFCC in the 4D input array
	print('input shape to neural network', X.shape)
	return X 

def prepare_data_record(n, sampling_rate):
	data , _ = record(2.5 , 22050) # records audio data and saves it in the required format
	data = removenoise(data,sampling_rate)  # call function to remove stationary noise
	X = np.empty(shape=(1, n, 108, 1)) # create a 4 dimenional array as input to the neural network
	print('data shape', data.shape)
	cnt = 0
	MFCC = librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc= n, n_fft = 4096, hop_length = 512) # MFCC extraction  
	print('MFCC before expanding dimensions', MFCC.shape)
	MFCC = np.expand_dims(MFCC, axis=-1) # expand the dimensions from 2 to 3
	print('MFCC after expanding dimensions', MFCC.shape)
	X[cnt,] = MFCC # save the MFCC in the 4D input array
	print('input shape to neural network', X.shape)
	return X


def detect_emotion_upload(): # function for detecting emotion of an uploaded file 
	#data, rate = readaudio()
	model = keras.models.load_model("D:/com2dcnn") # load the saved neural network model which was also trained 
	# Load audio file
	n_mfcc = 30 # no. of MFCC features
	audio_duration = 2.5 # no. of seconds 
	mfcc = prepare_data_upload(n = n_mfcc, sampling_rate = 22050, audio_duration=audio_duration) #Extract 30 MFCCs for 108 audio windows
	# Make prediction
	emotion_prediction = model.predict(mfcc) # model predicts the emotion of extracted audio features and saves as a number from 0-6
	emotion_prediction = emotion_prediction.argmax(axis=1) # 
	emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad'] # emotion labels for each number from 0-6
	predicted_label = emotion_labels[np.argmax(emotion_prediction)]
	print('Predicted emotion:', predicted_label)

def detect_emotion_record(): #function for detecting emotion of 
	model = keras.models.load_model("D:/com2dcnn") # load the saved neural network model which was also trained
	# Load audio file
	n_mfcc = 30 # no. of MFCC features
	audio_duration = 2.5 # no. of second
	mfcc = prepare_data_record(n = n_mfcc, sampling_rate = 22050) # Extract 30 MFCCs for 108 audio windows
	# Make prediction
	emotion_prediction = model.predict(mfcc) # model predicts the emotion of extracted audio features and saves as a number from 0-6
	emotion_prediction = emotion_prediction.argmax(axis=1) # 
	#emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad'] # emotion labels for each number from 0-6
	#predicted_label = emotion_labels[np.argmax(emotion_prediction)]
	print('Predicted emotion:', emotion_prediction)

	
class App(customtkinter.CTk): # define a class for thw Graphical User Interface (GUI)
    def __init__(self):
        super().__init__()
        self.geometry("800x400") # defines the size of the GUI
        self.title('Real time emotion detection') # Sets a title of the GUI
	
        self.grid_rowconfigure(100, weight=1)  # configure of rows in grid system
        self.grid_columnconfigure(100, weight=1) # configure of column in grid system

        # create a textbox explaining the purpose of the GUI
        self.textbox = customtkinter.CTkTextbox(master=self, height = 400, width=800, font = ('Times New Roman', 18), border_spacing = 50) 
        self.textbox.grid(row=0, column=0, sticky="nsew")
        self.textbox.insert("100.0", 'This application can detect your emotions in real time. \n You can choose to record yourself live or upload a pre-recorded audio file')

		


app = customtkinter.CTk()  # create CTk window like you do with the Tk window
app = App()

# Use CTkButton for recording and uploading audio files
record_button = customtkinter.CTkButton(master=app, text="Record and detect emotion", command = lambda: detect_emotion_record())
upload_button = customtkinter.CTkButton(master=app, text="Upload wav file and detect emotion", command = lambda:  detect_emotion_upload())

# Place the buttons on the GUI
record_button.place(relx=0.25, rely=0.5, anchor=tkinter.E)
upload_button.place(relx=0.75, rely=0.5, anchor=tkinter.E)

customtkinter.set_appearance_mode("dark")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green
app.mainloop()