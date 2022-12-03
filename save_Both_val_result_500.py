from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization as BN
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    GlobalAveragePooling2D,
)
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.layers import Layer
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
import numpy as np
import tensorflow as tf
import csv
import pandas as pd



gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_memory_growth(gpus[1], True)
    except RuntimeError as e:
        # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
        print(e)
    

class DataReader():
    def __init__(self):
        self.x_train_filepath = 'New_Data\\New_Both_Dataset\\shuffle_both_x_train_dataset.csv'
        self.y_train_filepath = 'New_Data\\New_Both_Dataset\\shuffle_both_y_train_dataset.csv'
        self.x_val_filepath = 'New_Data\\New_Both_Dataset\\no_shuffle_both_x_val_dataset.csv'
        self.y_val_filepath = 'New_Data\\New_Both_Dataset\\no_shuffle_both_y_val_dataset.csv'
        self.data_filepath = 'New_Data\\New_Both_Data\\validataon_pp_to_ja(DATA2).csv'

    def get_dataset(self):
        x_train_df=pd.read_csv(self.x_train_filepath, header=None)
        y_train_df=pd.read_csv(self.y_train_filepath, header=None)
        x_val_df=pd.read_csv(self.x_val_filepath, header=None)
        y_val_df=pd.read_csv(self.y_val_filepath, header=None)

        x_train = x_train_df.iloc[:,:].to_numpy()
        y_train = y_train_df.iloc[:,:].to_numpy()
        x_val = x_val_df.iloc[:,:].to_numpy()
        y_val = y_val_df.iloc[:,:].to_numpy()

        print("processing data...")
        self.make_y_data(y_train)
        self.make_y_data(y_val)
        self.make_x_data(x_train)
        self.make_x_data(x_val)
        

    # y(관절각도) 데이터 가공하는 함수
    def make_y_data(self,y_data):
        y_data_list = []
        for i in range(len(y_data[0])):
            tmp =  y_data[0][i]
            y_data_list.append(tmp)
        if len(y_data_list)>30000:
            self.y_train = np.array(y_data_list)
        else:
            self.y_val = np.array(y_data_list)
    
    # x(족저압) 데이터 가공하는 함수
    def make_x_data(self,x_data):
        x_data_list =[]
        for i in range(len(x_data)):
            data = x_data[i]
            converted_data = []

            #불필요한 값 제거("", [] 등)
            for line in data:
                tmp_line = line[1:-1].split(",")
                for i in range(len(tmp_line)):
                    element = tmp_line[i]
                    element = element.replace("\"", "")
                    element = element.replace("\'", "")
                    tmp_line[i] = element
                
                tmp_data = np.array(list(map(float,tmp_line)))
                converted_data.append(tmp_data)

            converted_data = np.array(converted_data)
            x_data_list.append(converted_data)

        if len(x_data_list)>30000: #109440
            self.x_train =np.array(x_data_list)
        else:                      #7200
           self.x_val =np.array(x_data_list)
    
    def get_rawsum(self):
        data_df=pd.read_csv(self.data_filepath, header=None)
        self.L_rawsum = data_df.iloc[1:,2].to_numpy()
        self.R_rawsum = data_df.iloc[1:,3].to_numpy()


def set_optimizer():
    scheduler = ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=855,
        decay_rate=0.982,
        staircase=False,
    )

    optimizer = Adam(learning_rate=scheduler)
    return optimizer

class Resnet2DConfig:
    activation = "swish"  # ReLU | swish
    out_activ = "None"  # None | ReLU | swish
    model_size = 50
    resnet_50 = [3, 4, 6, 3]
    resnet_101 = [3, 4, 23, 3]
    output_size = 1
    use_features = True


class Resnet2D(Model):
    def __init__(self):
        super(Resnet2D, self).__init__()
        self.conf = Resnet2DConfig
        if self.conf.model_size == 50:
            self.block_list = self.conf.resnet_50
        if self.conf.model_size == 101:
            self.block_list = self.conf.resnet_101

        self.conv1 = Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding="same")
        self.bn1 = BN()
        self.pool1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding="same")

        self.bottleneck_block1 = build_bottleneck_block_2d(
            filter_num=64, blocks=self.block_list[0]
        )
        self.bottleneck_block2 = build_bottleneck_block_2d(
            filter_num=128, blocks=self.block_list[1], stride=2
        )
        self.bottleneck_block3 = build_bottleneck_block_2d(
            filter_num=256, blocks=self.block_list[2], stride=2
        )
        self.bottleneck_block4 = build_bottleneck_block_2d(
            filter_num=512, blocks=self.block_list[3], stride=2
        )

        self.avgpool = GlobalAveragePooling2D()

        self.output_dense = Dense(2048)

    def call(self, inputs, training=False):
        image = inputs

        x = self.conv1(image)
        x = self.bn1(x, training)
        if self.conf.activation == "ReLU":
            x = tf.nn.relu(x)
        elif self.conf.activation == "swish":
            x = tf.nn.swish(x)
        x = self.pool1(x)
        x = self.bottleneck_block1(x, training=training)
        x = self.bottleneck_block2(x, training=training)
        x = self.bottleneck_block3(x, training=training)
        x = self.bottleneck_block4(x, training=training)
        x = self.avgpool(x)

        output = self.output_dense(x)

        return output


class ResnetBottleneckBlock2d(Layer):
    def __init__(self, filter_num, stride=1):
        super(ResnetBottleneckBlock2d, self).__init__()
        self.conv1 = Conv2D(
            filters=filter_num, kernel_size=(1, 1), strides=1, padding="same"
        )
        self.bn1 = BN()
        self.conv2 = Conv2D(
            filters=filter_num, kernel_size=(3, 3), strides=stride, padding="same"
        )
        self.bn2 = BN()
        self.conv3 = Conv2D(
            filters=filter_num * 4, kernel_size=(1, 1), strides=1, padding="same"
        )
        self.bn3 = BN()

        self.residual = Sequential()
        self.residual.add(
            Conv2D(filters=filter_num * 4, kernel_size=(1, 1), strides=stride)
        )
        self.residual.add(BN())

    def call(self, inputs, training=False):
        residual = self.residual(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.conv3(x)
        x = self.bn3(x, training=training)

        if Resnet2DConfig.activation == "ReLU":
            output = tf.nn.relu(tf.keras.layers.add([residual, x]))
        elif Resnet2DConfig.activation == "swish":
            output = tf.nn.swish(tf.keras.layers.add([residual, x]))
        return output


def build_bottleneck_block_2d(filter_num, blocks, stride=1):
    res_block = Sequential()
    res_block.add(ResnetBottleneckBlock2d(filter_num=filter_num, stride=stride))

    for _ in range(blocks):
        res_block.add(ResnetBottleneckBlock2d(filter_num=filter_num, stride=1))

    return res_block

def save_result(y_val, y_pred):
    save_val_file  = "Result\\both_val_result.csv"
    data = np.stack([y_val, y_pred, reader.L_rawsum, reader.R_rawsum], -1)
    df = pd.DataFrame(data, columns=["Joint_Angle", "Predict_Angle", "L_rawsum", "R_rawsum"])
    df.to_csv(save_val_file)


if __name__ == '__main__':

    # get saved data
    print("loading data...")
    reader  = DataReader()
    reader.get_rawsum()
    reader.get_dataset()

    x_train, x_val = reader.x_train, reader.x_val 
    y_train, y_val = reader.y_train, reader.y_val 

    x_train = x_train.reshape(-1, 59,42,1)
    x_val = x_val.reshape(-1,59,42,1)
    y_train = y_train.squeeze()
    y_val = y_val.squeeze()

    print(x_train.shape)      #(109440, 59, 42, 1)
    print(x_val.shape)        #(7200, 59, 42, 1)
    print(y_train.shape)      #(109440,)
    print(y_val.shape)        #(7200,)

    
    # build and train a model
    model = Resnet2D()
    model_none = Resnet2D()

    optim = set_optimizer()
    model.compile(optimizer=optim, loss='mse',metrics=['mae'])
    hist_decay = model.fit(x_train, y_train, epochs=500, shuffle=True, batch_size = 256)
    
    y_pred = model.predict(x_val)
    y_pred = y_pred.squeeze()
    y_pred_list=[]
    for i in range(len(y_pred)):
        y_pred_list.append(y_pred[i][0])

    save_result(y_val, y_pred_list)

    # save_val_file  = "Result\\both_val_result_csv.csv"//
    # f =  open(save_val_file, 'w', encoding='utf-8', newline="")
    # wr = csv.writer(f)

    # print('len(y_test): ',len(y_val))             #7200
    # print('len(y_pred_list):', len(y_pred_list))      #7200

    # cnt = 0
    # wr.writerow(["Joint_Angle", "Predict_Angle", "L_rawsum", "R_rawsum"])
    # for i in range(len(y_val)):
    #     wr.writerow([y_val[i], y_pred_list[i], reader.L_rawsum[i], reader.R_rawsum[i]])        
    #     print(i,"::: ", y_val[i],", ", y_pred_list[i],", ", reader.L_rawsum[i],", ", reader.R_rawsum[i])
    #     cnt+=1

    print("========= Result =========")
    print("MAE: ",mean_absolute_error(y_true=y_val, y_pred=y_pred_list))
    print("MAE2: ", np.mean(np.abs(y_val - y_pred_list)))
    print("std: ", np.std(np.absolute(np.subtract(y_val, y_pred_list))))
    print("=========================")
    print("ME: ", np.mean(np.subtract(y_val, y_pred_list)))
    print("std: ", np.std(np.subtract(y_val, y_pred_list)))
    print("=========================")
    print("mean relative error: ", np.mean(np.divide(np.absolute(np.subtract(y_val, y_pred_list)), y_val))*100)
    print("=========================")        
    print("r2 score: ", r2_score(y_true=y_val, y_pred=y_pred_list))
    print("=========================")

    # print(cnt)    #7200

    