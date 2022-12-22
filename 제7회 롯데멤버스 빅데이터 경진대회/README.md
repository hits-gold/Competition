main.ipynb : 모델 및 알고리즘 구현 파일



EDA.ipynb : EDA 파일



src : 모델 및 알고리즘 모듈화 파일이 저장된 폴더

-  ESG_data_calculator.ipynb : ESG_cal class -> ESG score를 계산해주는 모듈
- support.ipynb : support class -> 연관규칙분석을 통한 연관상품추천해주는 모듈
- collab_filtering_module : collab_filtering class -> 잠재요인 협업필터링 추천시스템을 통한 상품 추천해주는 모듈
- kskwClass : src 폴더 내 ipynb 형식으로 저장된 class들을 .py 확장자로 한꺼번에 저장한 파일

​    


model&code : 모델 학습 및 custom dataset 생성하는 파일이 저장된 폴더

- Collab_filtering.ipynb : 잠재요인 협업필터링 SVD모델 학습 코드파일
- ESG_Score_data.ipynb : 제공된 데이터셋으로 고객별 ESG score를 계산해 저장하는 코드파일
- TimeSeriesAnalysis.ipynb : 시계열 예측 모델 학습 코드파일