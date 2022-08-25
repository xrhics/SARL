import numpy as np
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Dropout,Conv2D,MaxPool2D,Flatten,Dot,concatenate,Permute,Reshape
from keras.optimizers import Adam,SGD
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,cohen_kappa_score,hamming_loss,jaccard_score,hinge_loss,confusion_matrix
from datetime import datetime
from absl import flags
from absl import app
import math
import pandas as pd
from mydata import cell_data
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau
L1=2  ##预测
L2=1      ##autoencoder
# L2=0.05      ##autoencoder
L4=1    ##sub 生成对抗
# LR1=1e-5
# LR2=5e-5
# LR3=5e-5
# LR4=5e-5
# LR5=5e-5
LR1=7e-4
LR2=7e-4
LR4=7e-4
TF_IDF=True
ENCODING_DIM_LAYER0 = 64
ENCODING_DIM_LAYER1 = 128
ENCODING_DIM_LAYER2 = 256
ENCODING_DIM_LAYER3 = 512
# ENCODING_DIM_LAYER0 = 32
# ENCODING_DIM_LAYER1 = 64
# ENCODING_DIM_LAYER2 = 128
# ENCODING_DIM_LAYER3 = 256
ENCODING_DIM_OUTPUT = 50
def my_loss1(y_true, y_pred):
    return L1*K.mean(K.categorical_crossentropy(y_true,y_pred), axis=-1)
def my_loss2(y_true, y_pred):
    return L2*K.mean(K.square(y_pred - y_true), axis=-1)
def my_loss4(y_true, y_pred):
    return L4*K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
FLAGS = flags.FLAGS
# General
flags.DEFINE_bool("adversarial", True, "Use Adversarial Autoencoder or regular Autoencoder")
flags.DEFINE_bool("train", True, "Train")
# Train
flags.DEFINE_integer("epochs",30, "Number of training epochs")
flags.DEFINE_integer("batchsize",4, "Training batchsize")
# np.random.seed(1337)
# np.random.seed(1337)
def get_train_test(dataset,cell_id_list):
    train_dataset = []
    test_dataset = []
    test_id=[]
    for i in range(len(dataset)):
        num=math.floor(0.8*len(dataset))
        if i<num:
            train_dataset.append(dataset[i])
        else:
            test_id.append(cell_id_list[i])
            test_dataset.append(dataset[i])
    return train_dataset,test_dataset,test_id
def load_decoder(input_dim):
    encode_z = Input(shape=(ENCODING_DIM_OUTPUT,))
    # decoding layer
    decode_layer1 = Dense(ENCODING_DIM_LAYER0)(encode_z)
    decode_layer2 = Dense(ENCODING_DIM_LAYER1)(decode_layer1)
    decode_layer3 = Dense(ENCODING_DIM_LAYER2)(decode_layer2)
    decode_layer4 = Dense(ENCODING_DIM_LAYER3)(decode_layer3)
    decode_output = Dense(input_dim, activation='relu')(decode_layer4)
    model=Model(inputs=encode_z,outputs=decode_output)
    return model

def load_encoder(autoencoder_input):
    layer0 = Dense(ENCODING_DIM_LAYER3)(autoencoder_input)
    layer1 = Dense(ENCODING_DIM_LAYER2)(layer0)
    layer2 = Dense(ENCODING_DIM_LAYER1)(layer1)
    layer3 = Dense(ENCODING_DIM_LAYER0)(layer2)
    encode = Dense(ENCODING_DIM_OUTPUT)(layer3)
    encoder = Model(inputs=autoencoder_input, outputs=encode)
    return encoder

def load_CNN(grid_input):
    layer0 = Conv2D(filters=128, kernel_size=(6,6),strides=(2,2), activation='relu',padding="same")(grid_input)
    layer1 = Conv2D(filters=64, kernel_size=(5,5),strides=(2,2), activation='relu',padding="same")(layer0)
    layer2 = Conv2D(filters=32, kernel_size=(4,4),strides=(2,2), activation='relu',padding="same")(layer1)
    layer3=Flatten()(layer2)
    layer4 =Dense(ENCODING_DIM_OUTPUT)(layer3)
    layer7=Dropout(0.1)(layer4)
    CNN = Model(inputs=grid_input,outputs=layer7)
    return CNN




def evaluate(test_y, y_test_pred):
    acc=accuracy_score(test_y, y_test_pred)
    wp=precision_score(test_y, y_test_pred,average='weighted')
    mp = precision_score(test_y, y_test_pred, average='macro')
    mr=recall_score(test_y, y_test_pred, average='macro')
    wf = f1_score(test_y, y_test_pred, average='weighted')
    mf1 = f1_score(test_y, y_test_pred, average='macro')
    k=cohen_kappa_score(test_y, y_test_pred)
    hamming_dis=hamming_loss(test_y, y_test_pred)
    j=jaccard_score(test_y, y_test_pred,average='weighted')
    con_matrix=str(confusion_matrix(test_y, y_test_pred))
    print([ acc, wp,wf,mp,mr,mf1,k,hamming_dis,j,con_matrix])
    return [ acc, wp,wf,mp,mr,mf1,k,hamming_dis,j,con_matrix]

def create_model(input_dim, class_num, grid_length):
    autoencoder_input = Input(shape=(input_dim,))
    generator_input = Input(shape=(input_dim,))
    grid_input = Input(shape=(grid_length,grid_length,10,))
    CNN=load_CNN(grid_input)
    encoder = load_encoder(autoencoder_input)
    layer=Reshape((ENCODING_DIM_OUTPUT,1))(encoder(autoencoder_input))
    layer0=Reshape((ENCODING_DIM_OUTPUT,1))(CNN(grid_input))
    layer1=concatenate([encoder(autoencoder_input),CNN(grid_input)])
    layer2=Dot(axes=-1)([layer0,layer])
    layer3=Flatten()(layer2)
    layer4=Dense(64)(layer1)
    layer5 = Dense(64)(layer3)
    layer6=concatenate([layer4,layer5])
    layer7=Dense(64)(layer6)
    layer8 = Dense(32)(layer7)
    layer9 = Dense(class_num, activation='softmax')(layer8)
    predict_y=Model(inputs=[autoencoder_input,grid_input],outputs=layer9)
    predict_y.compile(optimizer=SGD(lr=LR1), loss=my_loss1)
    decoder = load_decoder(input_dim)
    output_vector=Model(inputs=[autoencoder_input,grid_input],outputs=layer6)
    autoencoder = Model(inputs=autoencoder_input,outputs=decoder(encoder(autoencoder_input)))
    autoencoder.compile(optimizer=SGD(lr=LR2), loss=my_loss2)
    if FLAGS.adversarial:
        discriminator_sub_input = Input(shape=(input_dim,))
        discriminator_sub1 = Dense(ENCODING_DIM_LAYER2)(discriminator_sub_input)
        discriminator_sub2 = Dense(ENCODING_DIM_LAYER2)(discriminator_sub1)
        discriminator_sub3 = Dense(ENCODING_DIM_LAYER2)(discriminator_sub2)
        discriminator_sub4 = Dense(1, activation='sigmoid')(discriminator_sub3)
        discriminator_sub = Model(inputs=discriminator_sub_input, outputs=discriminator_sub4)
        discriminator_sub.compile(optimizer=SGD(lr=LR4), loss=my_loss4)

    if TF_IDF:
        substructure = load_model('tf_idf_topK.h5')
    else:
        substructure = load_model('topK.h5')
    substructure.trainable = False
    substructure.compile(optimizer=SGD(lr=LR2), loss=my_loss2)
    if FLAGS.adversarial:
        generator_sub = Model(generator_input, discriminator_sub(substructure(autoencoder(generator_input))))
        generator_sub.compile(optimizer=SGD(lr=LR4), loss=my_loss4)
    if FLAGS.adversarial:
        return predict_y,autoencoder,discriminator_sub,generator_sub,output_vector
    else:
        return predict_y,autoencoder, None, None



def train(city, length, min_count, batch_size, n_epochs):
    autoencoder_data = cell_data(city, length, min_count)
    if TF_IDF:
        autoencoder_data.file1="E:/LAND USE/data/" + city + "/" + city + "_" + str(length) + "cell_poi_tf-idf#" + str(min_count) + "_new.csv"
    autoencoder_data.data,autoencoder_data.cell_id_list=autoencoder_data.get_data()
    dataset = autoencoder_data.data
    cell_id_list=autoencoder_data.cell_id_list
    train_dataset, test_dataset,test_id = get_train_test(dataset,cell_id_list)
    x_l, x_grid_l, y_l, y_onehot_l = zip(*train_dataset)
    x_u, x_grid_u, y_u, y_onehot_u = zip(*test_dataset)
    x_train_l = np.array(x_l)
    x_train_l = x_train_l.astype('float32')
    x_train_grid_l = np.array(x_grid_l)
    y_train_l = np.array(y_onehot_l)
    y_train_l = y_train_l.astype('float32')
    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=2, mode='auto', factor=0.5, min_lr=1e-9)
    x_train_u = np.array(x_u)
    x_train_u = x_train_u.astype('float32')
    x_train_grid_u = np.array(x_grid_u)
    predict_y, autoencoder, discriminator_sub,  generator_sub,output_vector= create_model(input_dim=autoencoder_data.num, class_num=autoencoder_data.class_num, grid_length=int(length/10))
    past = datetime.now()
    for epoch in np.arange(1, n_epochs + 1):
        autoencoder_losses = []
        predict_y_losses=[]
        if FLAGS.adversarial:
            discriminator_losses_sub = []
            generator_losses_sub = []
        if TF_IDF:
            substructure = load_model('tf_idf_topK.h5')
        else:
            substructure = load_model('topK.h5')
        substructure.trainable = False
        x_input = np.concatenate((x_train_u, x_train_l))
        autoencoder_history = autoencoder.fit(x=x_input, y=x_input, epochs=1, batch_size=batch_size,shuffle=True, callbacks=[reduce_lr], verbose=0)
        if FLAGS.adversarial:
            all_num=x_train_l.shape[0]+x_train_u.shape[0]
            ###sub 生成对抗
            fake_sub = substructure.predict(autoencoder.predict(x_input))
            real_sub = substructure.predict(x_input)
            ##判别器
            sub_discriminator_input=np.concatenate((real_sub,fake_sub))
            sub_discriminator_labels=np.concatenate((np.ones((all_num, 1)),np.zeros((all_num, 1))))
            discriminator_sub_history = discriminator_sub.fit(x=sub_discriminator_input, y=sub_discriminator_labels, epochs=1,batch_size=batch_size, shuffle=True, verbose=0)
            ##生成器
            generator_sub_history = generator_sub.fit(x=x_input, y=np.ones((all_num, 1)), epochs=1,batch_size=batch_size, shuffle=True,validation_split=0.0, verbose=0)

        predict_y_loss = predict_y.fit(x=[x_train_l,x_train_grid_l], y=y_train_l, epochs=1, batch_size=batch_size,shuffle=True,callbacks=[reduce_lr], verbose=0)
        autoencoder_losses.append(autoencoder_history.history["loss"])
        predict_y_losses.append(predict_y_loss.history["loss"])
        if FLAGS.adversarial:
            generator_losses_sub.append(generator_sub_history.history["loss"])
            discriminator_losses_sub.append(discriminator_sub_history.history["loss"])
        now = datetime.now()
        print("\nEpoch {}/{} - {:.1f}s".format(epoch, n_epochs, (now - past).total_seconds()))
        print("Autoencoder Loss: {}".format(np.sum(autoencoder_losses)))
        print("Predict_y Loss. {}".format(np.mean(predict_y_losses)))
        if FLAGS.adversarial:
            print("Discriminator_sub Loss: {}".format(np.mean(discriminator_losses_sub)))
            print("Generator_sub Loss: {}".format(np.mean(generator_losses_sub)))
            print(np.sum(discriminator_losses_sub)+np.sum(generator_losses_sub)+np.sum(autoencoder_losses)+np.sum(predict_y_losses))
        else:
            print(np.sum(autoencoder_losses) + np.sum(predict_y_losses))
        past = now
        # y_pre = predict_y.predict([x_train_u, x_train_grid_u])
        # y_pre1 = predict_y.predict([x_train_l, x_train_grid_l])
        # loss_result=np.mean(my_loss1(y_train_u,y_pre))
        # label_pre = [np.argmax(item) for item in y_pre]
        # result = evaluate(y_u, label_pre)
        # label_pre1 = [np.argmax(item) for item in y_pre1]
        # result1 = evaluate(y_l, label_pre1)

    if TF_IDF:
        predict_y.save(city+'_encoder_sub_tf_idf_classfier.h5')
        autoencoder.save(city+'_autoencoder_sub_tf_idf_classfier.h5')
    else:
        predict_y.save(city+'_encoder_sub_classfier.h5')
        autoencoder.save(city+'_autoencoder_sub_classfier.h5')
    y_pre=predict_y.predict([x_train_u,x_train_grid_u])
    y_pre1 = predict_y.predict([x_train_l,x_train_grid_l])
    # loss_result=np.mean(my_loss1(y_train_u,y_pre))
    label_pre=[np.argmax(item) for item in y_pre]
    result=evaluate(y_u, label_pre)
    label_pre1 = [np.argmax(item) for item in y_pre1]
    result1 = evaluate(y_l, label_pre1)
    train_embedding=output_vector.predict([x_train_l,x_train_grid_l])
    test_embedding=output_vector.predict([x_train_u,x_train_grid_u])
    file1="E:/LAND USE/data/" + city + "/" + city + "_" + str(length) + "train_embedding.csv"
    file2 = "E:/LAND USE/data/" + city + "/" + city + "_" + str(length) + "test_embedding.csv"
    save_test_embedding=[[y_u[i]]+list(test_embedding[i]) for i in range(len(y_u))]
    save_train_embedding = [[y_l[i]] + list(train_embedding[i]) for i in range(len(y_l))]
    pd.DataFrame(save_train_embedding).to_csv(file1,index=None,header=None)
    pd.DataFrame(save_test_embedding).to_csv(file2, index=None,header=None)
    print(result)
    print(result1)
    pd.DataFrame([test_id,[i+1 for i in y_u],[j+1 for j in label_pre]]).T.to_csv("E:/LAND USE/data/" + city + "/" + city + "_" + str(length) + "result.csv",index=False,header=["id","truth","pre"])
    return result


def main(argv):
    cities=["Milano", "Amsterdam", "Barcelona", "Lisboa"]
    # cities = ["Milano"]
    results= []
    # lengths=[200,300,400,500]
    lengths = [200]
    min_count = 5
    times=10
    global desc
    if FLAGS.adversarial:
        desc = "test"
    else:
        desc = "regular"
    if FLAGS.train:
        for i in range(times):
            for length in lengths:
                for city in cities:
                    results.append([city,length]+train(city, length, min_count, batch_size=FLAGS.batchsize, n_epochs=FLAGS.epochs))
        pd.DataFrame(results).to_csv("E:/jupyer notebook/landuse/weighted-results"+str(L1)+"-"+str(L2)+"-"+str(L4)+"-"+str(LR1)+".csv",header=["city","length","acc", "wp","wf1", "mp", "mr","macro-f1","kappa","hamming_dis","jaccrd","con_matrix"])
if __name__ == "__main__":
    app.run(main)
