import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# 0. 파일 경로 및 이름 설정
DATA_FILE = "synthetic_orange_data_2000_kde.csv"
MODEL_FILE = "orange_sweetness_model.h5"

# 1. 훈련된 모델 불러오기
try:
    model = tf.keras.models.load_model(MODEL_FILE)
    print(f"--- '{MODEL_FILE}' 모델 불러오기 성공 ---")
    model.summary()
except (IOError, ImportError) as e:
    print(f"오류: 모델 파일 '{MODEL_FILE}'을 불러올 수 없습니다.")
    print("먼저 'train_dnn_model.py'를 실행하여 모델을 생성했는지 확인하세요.")
    exit()

# 2. 데이터 스케일러(Scaler) 준비
# 중요: 예측할 새로운 데이터도 훈련 데이터와 '동일한' 기준으로 스케일링해야 합니다.
# 이를 위해 훈련 시 사용했던 원본 데이터로 Scaler를 다시 학습시킵니다.
try:
    df_train = pd.read_csv(DATA_FILE)
    features = ['껍질두께(1~3)', '내부중량', '배꼽크기(1~3)', '질감(1~3)']
    X_train = df_train[features]
    
    scaler = StandardScaler()
    scaler.fit(X_train) # 훈련 데이터로 스케일러 학습
    print("\n--- 데이터 스케일러 준비 완료 ---")
except FileNotFoundError:
    print(f"오류: 스케일러 학습에 필요한 '{DATA_FILE}'을 찾을 수 없습니다.")
    exit()


# 3. 예측할 새로운 데이터 생성 (예시)
# 3개의 새로운 오렌지 데이터가 있다고 가정합니다.
new_oranges_data = {
    '껍질두께(1~3)': [1, 3, 2],
    '내부중량': [85.5, 45.2, 60.0],
    '배꼽크기(1~3)': [2, 1, 3],
    '질감(1~3)': [1, 3, 2]
}
new_oranges_df = pd.DataFrame(new_oranges_data)

print("\n--- 예측할 새로운 데이터 ---")
print(new_oranges_df)

# 4. 새로운 데이터 전처리 및 예측
# 훈련 데이터로 학습된 스케일러를 사용하여 새로운 데이터를 변환합니다.
new_oranges_scaled = scaler.transform(new_oranges_df)

# 모델을 사용하여 예측
predictions = model.predict(new_oranges_scaled)

# 5. 예측 결과 출력
print("\n--- 당도 예측 결과 ---")
# 예측 결과는 2D 배열이므로 flatten()을 사용하여 1D로 만듭니다.
for i, prediction in enumerate(predictions.flatten()):
    print(f"오렌지 {i+1} 예측 당도: {prediction:.2f}")

print("\n(참고: 이 예측은 'train_dnn_model.py' 실행 시 생성된 모델을 기반으로 합니다.)")
