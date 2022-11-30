import numpy as np
import tensorflow as tf
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
        self.x_train_filepath = 'no shuffle_side_both_x_train_dataset.csv'
        self.y_train_filepath = 'no shuffle_side_both_y_train_dataset.csv'
        self.x_test_filepath = 'no shuffle_side_both_x_test_dataset.csv'
        self.y_test_filepath = 'no shuffle_side_both_y_test_dataset.csv'

    def get_dataset(self):
        x_train_df=pd.read_csv(self.x_train_filepath, header=None)
        y_train_df=pd.read_csv(self.y_train_filepath, header=None)
        x_test_df=pd.read_csv(self.x_test_filepath, header=None)
        y_test_df=pd.read_csv(self.y_test_filepath, header=None)

        x_train = x_train_df.iloc[:,:].to_numpy()
        y_train = y_train_df.iloc[:,:].to_numpy()
        x_test = x_test_df.iloc[:,:].to_numpy()
        y_test = y_test_df.iloc[:,:].to_numpy()

        print("processing data...")
        self.make_y_data(y_train)
        self.make_y_data(y_test)
        self.make_x_data(x_train)
        self.make_x_data(x_test)
        

    # y(관절각도) 데이터 가공하는 함수
    def make_y_data(self,y_data):
        y_data_list = []
        for i in range(len(y_data[0])):
            tmp =  y_data[0][i]
            y_data_list.append(tmp)
        if len(y_data_list)>30000:
            self.y_train = np.array(y_data_list)
        else:
            self.y_test = np.array(y_data_list)
    


    # x(족저압) 데이터 가공하는 함수
    def make_x_data(self,x_data):
    
        x_data_list =[]
        for i in range(len(x_data)):
            data = x_data[i]
            converted_data = []

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

        print('x_list: ',len(x_data_list))
        if len(x_data_list)>30000: #11520
            self.x_train =np.array(x_data_list)
        else:
           self.x_test =np.array(x_data_list)


if __name__ == '__main__':

    # get saved data
    print("loading data...")
    reader  = DataReader()
    reader.get_dataset()

    x_train, x_test = reader.x_train, reader.x_test 
    y_train, y_test = reader.y_train, reader.y_test 

    print('type(x_test): ', type(x_test))
    print('type(y_test): ', type(y_test))

    index = np.arange(x_test.shape[0])
    np.random.shuffle(index)

    x_test = x_test[index]
    y_test = y_test[index]

    print('셔플 후 type(x_test): ', type(x_test))
    print('셔플 후 type(y_test): ', type(y_test))

    # 예시
    x = np.arange(10)
    y = np.arange(10)

    print('type(x): ', type(x))
    print('type(y): ', type(y))

    i= np.arange(x.shape[0])
    np.random.shuffle(i)
    x = x[i]
    print ("x: ", x)
    y= y[i]
    print ("y: ", y)
    print('셔플 후 type(x): ', type(x))
    print('셔플 후 type(y): ', type(y))



