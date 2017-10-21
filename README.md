# hyundai-destpred

현대자동차 근미래/최종 목적지 예측 과제

## 분석환경

- Ubuntu 16.04
- Python 3.6
- Tensorflow 1.3.0
- dependencies
  - `pip install tqdm colorlog`

## 데이터 위치/형식

- 모든 데이터는 `data` 폴더 내에 위치 (없으면 각자 만들 것)
- raw data는 어떠한 형식(csv, txt, tsv, ...)이든 상관 없음
  - 단, column parsing에 문제없도록 delimiter만 코드에 반영해줄 것.
  - 참고로, raw data의 컬럼 순서는 다음과 같음
    - `['car_id', 'start_dt', 'seq_id', 'x', 'y', 'link_id']`
- 한 번 전처리된 데이터는 pickle 형식으로 저장되어, 이후에는 전처리할 필요 없이 재사용됨

## 모듈 설명

- `estimator.py`: 메인 모듈. 미리 정해진 인자와 함께 실행
- `models.py`: 모델 그래프를 build하는 함수를 담고 있음
- `data_preprocessor.py`: 데이터 전처리 클래스를 담고 있음
- `log.py`, `utils.py`: 기타 기능 함수들 (파일이름, 출력, 시각화 등)

## API 사용예시

`python estimator.py dnn --preprocess --gpu_no=0 --train`

Usage statement: 

```
usage: estimator.py [-h] [--preprocess [PREPROCESS]] [--gpu_no GPU_NO]
                    [--learning_rate LEARNING_RATE] [--steps STEPS]
                    [--log_freq LOG_FREQ] [--train [TRAIN]]
                    model_type

positional arguments:
  model_type            dnn/rnn

optional arguments:
  -h, --help            show this help message and exit
  --preprocess [PREPROCESS]
                        Preprocess data or not
  --gpu_no GPU_NO       gpu device number
  --learning_rate LEARNING_RATE
                        initial learning rate
  --steps STEPS         step size
  --log_freq LOG_FREQ   log frequency
  --train [TRAIN]       train or just eval
```

## 진행 계획

- 9월
  - [x] data_loader
  - [x] preprocessor
  - [x] baseline models: feed forward network
  - [x] initial models: recurrent neural network
  - [x] metadata embedding

- 10월
  - [x] applying dataset api and estimator class
  - [ ] destination inference: one-shot prediction or averaging cluster centroids
  - [ ] structure engineering
  - [ ] hyperparameter tuning

- ...TBD