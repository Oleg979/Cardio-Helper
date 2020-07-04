from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# загружаем данные
dataset = np.loadtxt("models/dataset.txt", delimiter=";")

X = dataset[:, 0:7]
Y = dataset[:, 7:8]

model = Sequential()
model.add(Dense(7, input_dim=7, activation='relu'))
model.add(Dense(164, activation='relu'))
model.add(Dense(164, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs=195, batch_size=4, verbose=2)

model_json = model.to_json()
json_file = open("models/model.json", "w")
json_file.write(model_json)
json_file.close()
model.save_weights("models/model.h5")