
import itertools
from pickle import TRUE

import pandas as pd
from datetime import datetime
import numpy as np
import csv

import numpy as np
import scipy.interpolate as ip
from scipy.interpolate import splrep, splev, interpolate
import matplotlib.pyplot as plt



# input_filepath_Left = 'D:\\1_정아현\\2022_대한전기학회_하계학술대회\\f-scan laft data\\L_data4.csv'        #족저압 input데이터  파일경로
# left_input_df = pd.read_csv(input_filepath_Left,encoding='cp949')                                                       #input 데이터 csv파일 불러와서 데이터프레임 저장    

# input_filepath_Right = 'D:\\1_정아현\\2022_대한전기학회_하계학술대회\\f-scan right data\\R_data4.csv'         #족저압 input데이터  파일경로
# right_input_df = pd.read_csv(input_filepath_Right,encoding='cp949')                                                       #input 데이터 csv파일 불러와서 데이터프레임 저장 

# angle_filepath = 'D:\\1_정아현\\2022_대한전기학회_하계학술대회\\azure_data\\angle4.csv'
# angle_df = pd.read_csv(angle_filepath,encoding='cp949') 
def get_filepath(num):
    input_filepath_Left = 'D:\\1_정아현\\2022_대한전기학회_하계학술대회\\f-scan laft data\\L_data'+ str(num) +'.csv'        #족저압 input데이터  파일경로
    left_input_df = pd.read_csv(input_filepath_Left,encoding='cp949')                                                       #input 데이터 csv파일 불러와서 데이터프레임 저장    

    input_filepath_Right = 'D:\\1_정아현\\2022_대한전기학회_하계학술대회\\f-scan right data\\R_data'+ str(num) +'.csv'         #족저압 input데이터  파일경로
    right_input_df = pd.read_csv(input_filepath_Right,encoding='cp949')                                                       #input 데이터 csv파일 불러와서 데이터프레임 저장 

    angle_filepath = 'D:\\1_정아현\\2022_대한전기학회_하계학술대회\\azure_data\\angle'+ str(num) +'.csv' 
    angle_df = pd.read_csv(angle_filepath,encoding='cp949') 
    
    save_filepath = 'New_Both_Data\\pp_to_ja'+ str(num) +'_1.csv'
    save_df =  open(save_filepath, 'w', encoding='utf-8', newline="")

    data_check_filepath= 'New_Both_Data\\data_check'+ str(num) +'_1.csv'
    data_check_df =  open(data_check_filepath, 'w', encoding='utf-8', newline="")

    save_angle_filepath = 'New_Both_Data\\interpolation_angle'+ str(num) +'_1.csv'
    save_angle_df =  open(save_angle_filepath, 'w', encoding='utf-8', newline="")

    return left_input_df, right_input_df, angle_df, save_df, data_check_df, save_angle_df


i=0
#족저압 클래스
class PlantarPressure:

    def __init__(self, frame_start, left_input_df, right_input_df) :
        self.frame_start = frame_start
        self.left_input_df = left_input_df
        self.right_input_df = right_input_df

    #족저압 time 데이터 불러오는 함수
    def get_input_time(self):
        time_df = self.left_input_df.loc[self.frame_start-1:self.frame_start-1]         #데이터의 컬럼 행 데이터프레임 가져오기
        time_values =time_df.filter(['time']).values                      #time 열의 데이터프레임만 가져오기
        
        time_str = str(time_values)                                      # 데이터 프레임을 문자열로 변환
        time_str = time_str[4:-3]
    
        try:
            self.time = datetime.strptime(time_str, '%H:%M:%S.%f')       # 시간 형식 지정
        except ValueError:
            self.time = datetime.strptime(time_str, '%H:%M:%S')
            
        # self.time =  self.time.time()                                                # 날짜부분 제거하고 시간만 time에 저장
        self.time  = self.time.replace(hour = self.time.hour + 12)                # hour를 24시 표시로 바꾸기
        return self.time

        
    #족저압 input 데이터 불러오는 함수
    def get_input_Left_data(self,):

        start_row = self.frame_start        #데이터 시작 행 62+62
        end_row = self.frame_start+59       #데이터 끝 행   62+62
        
        self.leftdata = self.left_input_df.loc[start_row: end_row]    
        self.leftdata = self.leftdata.values.tolist()         #데이터프레임을 2차원 리스트로 저장

        for i in range(len(self.leftdata)-1):
            for j in range(len(self.leftdata[i])):
                if self.leftdata[i][j] == 'B' or self.leftdata[i][j] == '0':
                    self.leftdata[i][j] = float(0.0)

        return self.leftdata

    def get_input_right_data(self,):
        start_row = self.frame_start        #데이터 시작 행 62+62
        end_row = self.frame_start+59       #데이터 끝 행   62+62
        
        self.rightdata = self.right_input_df.loc[start_row: end_row]    
        self.rightdata = self.rightdata.values.tolist()         #데이터프레임을 2차원 리스트로 저장

        for i in range(len(self.rightdata)-1):
            for j in range(len(self.rightdata[i])):
                if self.rightdata[i][j] == 'B' or self.rightdata[i][j] == '0':
                    self.rightdata[i][j] = float(0.0)

        return self.rightdata    
    
   
    def get_input_rawsum(self):
        global i
        
        self.left_input_df = self.left_input_df.loc[self.frame_start-1:self.frame_start-1]         #데이터의 컬럼 행 데이터프레임 가져오기
        self.right_input_df = self.right_input_df.loc[self.frame_start-1:self.frame_start-1]         

        self.left_rawsum = self.left_input_df.filter(['raw sum']).values                      #raw sum 열의 데이터프레임만 가져오기
        self.right_rawsum = self.right_input_df.filter(['raw sum']).values                      

        self.left_rawsum = list(itertools.chain(*self.left_rawsum)) 
        self.right_rawsum = list(itertools.chain(*self.right_rawsum)) 

        self.left_rawsum =  float(self.left_rawsum[0])
        self.right_rawsum =   float(self.right_rawsum[0])

        # print( i, "번째", "left raw sum: ", self.left_rawsum, "   right_rawsum ", self.right_rawsum)
        
        i+=1
        return self.left_rawsum, self.right_rawsum

    




    
# 관절 각도 클래스
class JointAngle:
    def __init__(self, angle_df) :
        self.angle_df = angle_df

    #관절 각도 time 데이터 불러오는 함수
    def get_angle_time(self):          
        self.time_values = self.angle_df.iloc[:,[0]].values.tolist()
        
        for index in range(len(self.time_values)-1):
            self.time_str  = str(self.time_values[index])                                            # 데이터 프레임을 문자열로 변환
            self.time_str = self.time_str[13:-2]

            try:
                self.time_format = datetime.strptime(self.time_str, '%H:%M:%S.%f')       # 시간 형식 지정
            except ValueError:
                self.time_format = datetime.strptime(self.time_str, '%H:%M:%S:')
            
            # self.time_format =  self.time_format.time()                                       # 날짜부분 제거하고 시간만 time에 저장
            self.time_values[index] = self.time_format
        
        self.time = self.time_values
        return self.time


    #관절 각도 output 데이터 불러오는 함수
    def get_angle_data(self):   
        self.data = self.angle_df.iloc[:,[1]].values.tolist()  # 원하는 행(index, 0부터 시작)에 맞는 데이터프레임을 2차원 리스트로 저장
        self.data = list(itertools.chain(*self.data))                             # 2차원 리스트 -> 1차원 리스트
        
        # print( "angle", self.data ) 
        return self.data
    


# 족저압과 관절각도 csv 저장 클래스
class SaveCSV:
    def __init__(self, save_df, data_check_df) :
        self.save_df =save_df
        self.data_check_df = data_check_df

    # csv 생성 및 컬럼 작성
    def make_csv(self):
        header = ['index', 'joint_angle', 'L_rawsum','R_rawsum', 'foot', 'plantar_pressure']
        self.wr = csv.writer(self.save_df)
        self.wr.writerow(header) 

        header2 = ['L_rawsum','R_rawsum', 'joint_angle']
        self.wr2 = csv.writer(self.data_check_df)
        self.wr2.writerow(header2) 

    # file close
    def close_csv(self):
        self.save_df.close()


    # 데이터 받아와서 csv에 저장
    def save_csv(self, start_index, ja_data, pp_left_data, pp_right_data, pp_left_rawsum, pp_right_rawsum):
    
        data_check_sheet(pp_left_rawsum, pp_right_rawsum, ja_data, self.data_check_df)  

        self.write_data = [start_index, ja_data, pp_left_rawsum, pp_right_rawsum]
        self.write_data.append('Both')
        left_one_pp = []
        right_one_pp = [] 

        for i in range(59):
            left_one_pp.append(pp_left_data[i])

        for i in range(59):
            right_one_pp.append(pp_right_data[i])      
        
        merge = (list(map(list.__add__, left_one_pp, right_one_pp)))

        for i in range(59):
            self.write_data.append(merge[i])

        self.wr.writerow(self.write_data)


#rawsum과 관절각도 함께 저장하여 데이터 확인할때 사용하는 함수   
def data_check_sheet(pp_left_rawsum, pp_right_rawsum, ja_data, data_check_f):
    wr = csv.writer(data_check_f)  
    data = [pp_left_rawsum,pp_right_rawsum, ja_data]
    wr.writerow(data)


# 보간한 관절각도 저장 함수   
def angle_sheet(all_angle, save_angle_df):
    wr = csv.writer(save_angle_df)
    header = ['joint_angle']
    wr.writerow(header)
    for i in range(len(all_angle)):
        data = [all_angle[i]]
        wr.writerow(data)



        

# 족저압과 관절각도 값 시간 맞게 매칭 시켜서 저장하는 함수
def save_pp_to_ja( pp_list, ja, csv, save_angle_df):
    min_time_index = 0
    before_jaindex = 0
    before_ppindex=0
    interpolation_angle_list = []

    for index in range(len(pp_list)-1):
        # 해당 인덱스 번호의 족저압, 시간 데이터 가져오기
        pp_left_data = pp_list[index].leftdata           # 족저압 2차원 배열 하나 (족저압 하나)
        pp_right_data = pp_list[index].rightdata
        pp_time = pp_list[index].time
        pp_left_rawsum = pp_list[index].left_rawsum
        pp_right_rawsum =pp_list[index].right_rawsum


        # 관절각도 시간 데이터 가져오기
        ja_time_list = ja.time
        
        #  관절각도 시간과 족저압 시간 비교하여 가장 비슷한 시간의 인덱스 가져오기
        #  인덱스로 관절각도 값 가져옴
        min = np.absolute(pp_time-ja_time_list[0])

        
        for ja_index in range(len(ja_time_list)-1):
            time_diff = np.absolute(pp_time-ja_time_list[ja_index])

            if min > time_diff:
                min =  time_diff
                min_time_index = ja_index

            else:
                continue
                
        ja_index  = min_time_index 
        
        if index == 0:
            before_jaindex = ja_index
  


        # 족저압 time과  가장 비슷한 time의  관절각도 데이터    
        ja_data = ja.data[ja_index]
        
        #관절 각도가 변할 때 같은 값의 개수만큼 보간한 리스트 가져오기
        if before_jaindex != ja_index :
            last_ppindex= index

            interpolation_angle = interpolation(ja.data[before_jaindex],ja.data[ja_index], before_ppindex, last_ppindex)
            for i in range(len(interpolation_angle)-1):
               interpolation_angle_list.append(interpolation_angle[i])

            before_ppindex =  index
            before_jaindex = ja_index


        
        csv.save_csv(index, ja_data, pp_left_data, pp_right_data, pp_left_rawsum , pp_right_rawsum)
    print("end len is", len(interpolation_angle_list))
    angle_sheet(interpolation_angle_list, save_angle_df)
    print("save end")



def interpolation(angle_start, angle_last, before_ppindex, last_ppindex):    
    # 0~10까지 15개로 나누어 점을 찍음.
    num = last_ppindex-before_ppindex+1

    x=[0,1]
    x0 = np.array(x)

    y = [angle_start, angle_last]
    y0 = np.array(y)
 
    f1 = interpolate.interp1d(x0,y0)
    x_new = np.linspace(0,1,num+1)
    

    plt.plot(x0,y0,'o',x_new,f1(x_new),'-')
    plt.legend(['data','linear'], loc='best')
    
    interpolation_angle = []

        
    for i in range(num):   
        interpolation_angle.append(f1(x_new[i]))
    
    if last_ppindex > 11994 :
        interpolation_angle.append(angle_last)

    return interpolation_angle


# 분할 인덱스 가져오는 함수
def get_index(index_df):

    start_index_list =  index_df.iloc[:,[0]].values.tolist() 
    start_index_list = list(itertools.chain(*start_index_list))                             # 2차원 리스트 -> 1차원 리스트

    end_index_list =  index_df.iloc[:,[1]].values.tolist() 
    end_index_list = list(itertools.chain(*end_index_list))                             # 2차원 리스트 -> 1차원 리스트

    return start_index_list, end_index_list



if __name__ == "__main__":

    for num in range(1, 3):

        frame_start = 1          #족저압 프레임 시작 인덱스
        index = 0                 #족저압 객체배열에 사용될 인덱스
        pp_list = []                #족저압 객체 저장 리스트  

        left_input_df, right_input_df, angle_df, save_df, data_check_df, save_angle_df = get_filepath(num)   
        
        
        #족저압 객체 생성 및 리스트로 저장
        while(True):
            pp=PlantarPressure(frame_start, left_input_df, right_input_df)
            pp.leftdata = pp.get_input_Left_data()
            pp.rightdata = pp.get_input_right_data()
            
            
            
            if len(pp.leftdata) <=1 or len(pp.rightdata) <=1:
                break

            pp.left_rawsum, pp.right_rawsum = pp.get_input_rawsum()
            pp.time = pp.get_input_time()
            pp_list.append(pp)
            
            frame_start+=62

        # SaveCSV 객체 생성 및 csv 생성
        saveCSV = SaveCSV(save_df, data_check_df)
        saveCSV.make_csv()

        #angle객체 생성 및 변수 초기화
        ja=JointAngle(angle_df)
        ja.time = ja.get_angle_time()
        ja.data = ja.get_angle_data()


        # 족저압과 관절각도 데이터 매칭 후 csv에 저장
        save_pp_to_ja(pp_list, ja, saveCSV, save_angle_df)
        print("**file", num, " save end**")

    