[ 폴더 및 파일 설명 ]
├─ Both_Data(side) 폴더 : 양 발 족저압 이미지를 옆으로 이어붙인 데이터를 사용
 |    ├─ data_check.csv : 족저압 rawsum과 관절각도 저장 파일, 그래프로 데이터를 확인하기 위해 사용
 |    ├─ interpolation_angle.csv : 보간한 관절각도 저장 파일
 |    └─ pp_to_ja.csv : 족저압이미지데이터와 관절각도를 매치시켜 저장한 파일, 데이터셋 나눌 때 사용.
 | 
├─ DPP_Data폴더 : Swing phase에서 반대쪽 발의 족저압 데이터를 사용 
 |     ├─ data_check.csv
 |     ├─ interpolation_angle.csv
 |     └─ pp_to_ja.csv 
 |
├─ Dataset 폴더 : y_test, y_pred, x_test를 저장한 폴더
 |     ├─ no shuffle_side_both_···_dataset.csv : Both_Data를 사용하여 셔플 안하고 나눈 데이터셋
 |
├─ Result 폴더 : y_test, y_pred, x_test를 저장한 폴더
 |     ├─ no_shuffle_both_result : no shuffle_side_both_···_dataset.csv를 사용한 결과 
 | 
├─ make_Both_data.py : Both_Data(side) 데이터 생성
├─ make_DPP_data.py : DPP_Data 데이터 생성
├─ save_dataset.py : Dataset 데이터 생성
├─ Resnet50_save_result.py : 모델 생성, 학습 및 Result 데이터 저장
├─ test_shuffle.py : x_test. y_test를 같은 순서로 섞는 코드


[ 데이터 셋 생성하기 ]
00. make_Both_data.py 16~22번째 줄의 원본데이터 파일경로 확인하기
01. make_Both_data.py 파일 실행 
     -> 양 발 족저압 이미지를 옆으로 이어붙인 데이터와 관절 데이터를 함께 저장한 파일 생성됨 (예) Both_Data(side)\pp_to_ja1.csv
02. save_dataset.py 파일 실행 
     -> 위에서 저장한 데이터를 통합하여 셔플 안한 상태로 8:2 split한 데이터셋 저장한 파일 생성됨 (예) Dataset\no shuffle_side_both_x_test_dataset.csv

[ 모델 학습 및 결과 저장하기 ]
01. Resnet50_save_result.py 실행
     -> y_test, y_pred, x_test 가 저장된 파일 생성됨 (예) Result\no_shuffle_both_result.csv
