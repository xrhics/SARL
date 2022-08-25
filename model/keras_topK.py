import numpy as np
import pandas as pd
# for reproducibility
np.random.seed(1337)
from tqdm import tqdm
from keras.models import Model
from keras.layers import Dense, Activation, Conv1D, MaxPooling1D,GlobalAveragePooling1D,Dropout,Reshape,Input
from keras.optimizers import Adam
from sklearn.utils import shuffle
import keras.backend as K
from keras.callbacks import LearningRateScheduler
# def my_loss(y_true, y_pred):
#     return 0.5*K.mean(K.square(y_pred - y_true), axis=-1)
TFIDF=False
def load_data():
    print("loading...")
    dataset=[]
    cities = ["Milano", "Amsterdam", "Barcelona", "Lisboa"]
    # cities = ["NYC"]
    lengths = [200, 300, 400, 500]
    # lengths = [500]
    min_count = 5
    k = 10
    for city in tqdm(cities):
        for length in lengths:
            if TFIDF:
                x_file = "E:/LAND USE/data/" + city + "/" + city + "_" + str(length) + "cell_poi_tf-idf#" + str(min_count) + "_new.csv"
                y_file = "E:/LAND USE/data/" + city + "/" + city + "_" + str(length) + "cell_poi_tf-idf_sub#" + str(min_count) + "$k=" + str(k) + "_new.csv"

            else:
                x_file="E:/LAND USE/data/" + city + "/" + city + "_" + str(length) + "cell_point_tree#" + str(min_count) + "_new.csv"
                y_file="E:/LAND USE/data/" + city + "/" + city + "_" + str(length) + "cell_tree_sub#"+str(min_count)+"$k="+str(k)+"_new.csv"
            x_data=pd.read_csv(x_file)
            y_data=pd.read_csv(y_file,header=None)
            for i in range(len(x_data)):
                x_data_row=x_data.loc[i].tolist()[1:]
                y_data_row=y_data.loc[i].tolist()[1:]
                dataset.append([x_data_row,y_data_row])
    return dataset

def get_test(dataset):
    shuffle(dataset)
    test_set=[]
    train_set = dataset
    return train_set,test_set

def scheduler(epoch):
    # 每隔10个epoch，学习率减小为原来的1/2
    if epoch % 10 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.8)
        print("lr changed to {}".format(lr * 0.8))
    return K.get_value(model.optimizer.lr)

dataset=load_data()
train_set,test_set=get_test(dataset)
train_x,train_y=zip(*train_set)
# test_x,test_y=zip(*test_set)
train_x=np.array(train_x).astype(np.float32)
train_y=np.array(train_y).astype(np.float32)
# test_x=np.array(test_x).astype(np.float32)
# test_y=np.array(test_y).astype(np.float32)

input = Input(shape=(len(train_x[0]), ))

    # encoding layer
layer1 = Dense(64)(input)
layer2 = Dense(128)(layer1)
layer3 = Dense(256)(layer2)
layer4 = Dense(512)(layer3)
# layer4 = Dense(128, activation='relu')(layer3)
# layer5 = Dense(64, activation='relu')(layer4)
# layer6=Dropout(0.1)(layer3)
# output = Dense(len(train_x[0]), activation='relu')(layer3)
output = Dense(len(train_x[0]), activation='relu')(layer4)
model=Model(inputs=input, outputs=output)
# Another way to define your optimizer
adam = Adam(lr=8e-4)
# We add metrics to get more results you want to see
model.compile(
    optimizer=adam,
    loss='mse',
)
# model.compile(
#     optimizer=adam,
#     loss=my_loss,
# )
print('Training------------')
# Another way to train the model
reduce_lr = LearningRateScheduler(scheduler)
# reduce_lr = ReduceLROnPlateau(monitor='loss', patience=2, mode='auto',factor=0.5,min_lr=1e-8)
model.fit(train_x, train_y, epochs=100, batch_size=32, shuffle=True,callbacks=[reduce_lr])
if TFIDF:
    model.save('tf_idf_topK.h5')
else:
    model.save('topK.h5')
print('\nTesting------------')
# Evaluate the model with the metrics we defined earlier
# pre_y=model.predict(test_x)
# loss= model.evaluate(test_x, test_y)
pre_y1=model.predict(train_x)
loss1= model.evaluate(train_x,train_y)
print('test loss:', loss1)
# print(test_x[0])
# print(test_y[0])
# print(pre_y[0])