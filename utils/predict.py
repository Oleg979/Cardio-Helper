import numpy as np
import tensorflow


def predict(data):
    print("Loading network...")

    json_file = open("models/model.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = tensorflow.keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights("models/model.h5")
    print("Network loaded.")
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    processed_data = np.array([data])
    print(processed_data.shape)
    prediction = loaded_model.predict(processed_data)
    print(prediction[0][0])
    return prediction[0][0]
