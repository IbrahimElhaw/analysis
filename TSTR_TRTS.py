import numpy as np
from keras import Sequential, Input
from keras.src.callbacks import EarlyStopping
from keras.src.layers import LSTM, Dropout, Dense
from keras.src.losses import SparseCategoricalCrossentropy
from physical_characteristics import filter_data


def balance_data(data, labels):
    data = np.array(data)
    labels = np.array(labels)
    # Find indices for each gender
    unique_labels = np.unique(labels)
    male_indices = [i for i, label in enumerate(labels) if label == unique_labels[0]]
    female_indices = [i for i, label in enumerate(labels) if label == unique_labels[1]]

    min_count = min(len(male_indices), len(female_indices))

    # Shuffle indices before selecting balanced subsets
    np.random.shuffle(male_indices)
    np.random.shuffle(female_indices)

    # Select a balanced subset of indices
    balanced_indices = male_indices[:min_count] + female_indices[:min_count]
    np.random.shuffle(balanced_indices)
    balanced_data = data[balanced_indices]
    balanced_labels = labels[balanced_indices]

    return balanced_data, balanced_labels


data_o = np.load("training__data/original.npy", allow_pickle=True)
infos_o = np.load("training__data/infos_o.npy", allow_pickle=True)

number = 0
Y_FEATURE = 3
# 0: person, 1: character, 2: glyph, 3: finger, 4: gender, 5: hand, 6: age
EPOCHES = 25
NumberOfDense = 1
NumberOfLSTM = 3
NeuronOfDenseCells = 20
NeuronOfLSTMCells = 128

data_o, _, _, _, infos_o = filter_data(data_o, infos_o, number=number)
print("length of data = ", len(data_o))
np.random.seed(0)

infos_o = infos_o[:, Y_FEATURE]

data_o, infos_o = balance_data(data_o, infos_o)
print(infos_o[:10])

unique_labels = np.unique(infos_o)
infos_o = np.where(infos_o == unique_labels[0], 1, 0)

infos_o = infos_o.astype(int)
print(infos_o[:10])
# split the data to train and validation
split_index = int(0.8 * len(data_o))
train_data_o = data_o[:split_index]
val_data_o = data_o[split_index:]
train_infos_o = infos_o[:split_index]
val_info_o = infos_o[split_index:]

# extracting the number of classes (should be 2 anyway)
num_classes = len(np.unique(infos_o))
print("num of classes:", num_classes)

# building model
model = Sequential()
model.add(Input(shape=np.shape(data_o[0])))
for i in range(NumberOfLSTM):
    model.add(LSTM(NeuronOfLSTMCells, return_sequences=True))
    model.add(Dropout(0.2))
model.add(LSTM(NeuronOfLSTMCells // 2))
model.add(Dropout(0.1))
for i in range(NumberOfDense):
    model.add(Dense(NeuronOfDenseCells, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
early_stopping = EarlyStopping(
    monitor='val_accuracy',  # Metric to monitor
    patience=30,  # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restore the weights of the model from the epoch with the best validation accuracy
)

# debugging code
print(np.shape(train_data_o))
print(np.shape(val_data_o))
print(np.shape(train_infos_o))
print(np.shape(val_info_o))

history = model.fit(
    train_data_o,
    train_infos_o,
    epochs=EPOCHES,
    batch_size=64,
    validation_data=(val_data_o, val_info_o),
    callbacks=[early_stopping]
)

# model.save("TSTR with represented predict number.keras")
