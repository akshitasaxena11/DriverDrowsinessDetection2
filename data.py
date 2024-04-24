import csv
import csv
import numpy as np
import os.path
import operator

from keras.utils import to_categorical


class Dataset():

	def __init__(self, seq_length=26, class_limit=2, image_shape=(56, 24, 3)):
		
		self.seq_length = seq_length 
		self.class_limit = class_limit
		self.sequence_path = os.path.join('data', 'new_sequences')
		self.data = self.get_data()
		self.classes = self.get_classes()

	def get_data(self):
		
		with open(os.path.join('data', 'data_file copy.csv'), 'r') as fin:
			
			reader = csv.reader(fin)
			data = list(reader)
			return data
				
	        	
	def get_classes(self):
		
		classes = ["Alert", "Drowsy"]
		return classes

	def get_class_one_hot(self, class_str):
			
			label_encoded = self.classes.index(class_str)
        	
			label_hot = to_categorical(label_encoded, len(self.classes))
			assert len(label_hot) == len(self.classes)

			return label_hot
        	

	def get_all_sequences_in_memory(self, train_test):

        #train, test = self.split_train_test()
		#data = train if train_test == 'train' else test

		print("Loading samples into memory for --> ",train_test,self.sequence_path)

		X, y = [], []

		for videos in self.data:
					if(videos[0] == train_test):
						sequence = self.get_extracted_sequence(videos)
						if sequence is None:
							print("Can't find sequence. Did you generate them?")
							raise
						X.append(sequence)
						y.append(self.get_class_one_hot(videos[1]))
		return np.array(X), np.array(y)


	def get_extracted_sequence(self, video):
		"""Get the saved extracted features."""
		filename = video[2]
		path = os.path.join(self.sequence_path, filename + '-' + str(26) + \
	            '-' + 'features' + '.npy')
		if os.path.isfile(path):
	        	return np.load(path)
	
   
