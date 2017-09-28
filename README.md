# hyundai-destpred

현대자동차 근미래/최종 목적지 예측 과제

## 분석환경

- Ubuntu 16.04
- Python 3.6
- Tensorflow 1.3.0

## 데이터 위치/형식

- 모든 데이터는 `data` 폴더 내에 위치 (없으면 각자 만들 것)
- raw data는 어떠한 형식(csv, txt, tsv, ...)이든 상관 없음
  - 단, column parsing에 문제없도록 delimiter만 코드에 반영해줄 것.
  - 참고로, raw data의 컬럼 순서는 다음과 같음
    - `['car_id', 'start_dt', 'seq_id', 'x', 'y', 'link_id']`
- 한 번 전처리된 데이터는 pickle 형식으로 저장되어, 이후에는 전처리할 필요 없이 재사용됨

## 진행 계획

- 9월
  - [x] data_loader
  - [x] preprocessor
  - [x] baseline models: feed forward network
  - [x] initial models: recurrent neural network
  - [x] metadata embedding

- 10월
  - [ ] destination inference: one-shot prediction or averaging cluster centroids
  - [ ] structure engineering
  - [ ] hyperparameter tuning

- ...TBD