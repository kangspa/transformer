# Transformer 모델 구축 및 간단 테스트

> 기간 : 25.06.30 ~ 25.07.04
> 상세 내용 : [블로그 참고](https://velog.io/@kangspa/Transformer-모델-직접-구축-및-테스트)
> 사용한 데이터 : [Kaggle WMT 2014 English-German](https://www.kaggle.com/datasets/mohamedlotfy50/wmt-2014-english-german)
> 참고한 교재 : [파이토치 트랜스포머를 활용한 자연어 처리와 컴퓨터비전 심층학습](https://github.com/wikibook/pytorchtrf)

최대한 논문 내용 토대로, `pytorch`를 활용해서 직접 모델 구축 테스트를 진행한 프로젝트이다.

완성 코드에 대한 상세 설명은 [블로그](https://velog.io/@kangspa/Transformer-모델-직접-구축-및-테스트)에 작성해두었고, 여기에는 파일별 간단한 설명을 작성해둔다.

## custom_train_validation.ipynb

완성한 transformer 모델을 토대로 간단히 학습 및 검증을 진행한 파일이다. (상세 내용 블로그에 작성)

## custom_transformer1.py

최대한 외부 자료 도움 없이 구축 시도한 transformer 모델 파일이다.
문제가 있다면 `padding_mask`에 대한 처리를 하는걸 생각을 안 했고, 학습 시 `Target` 데이터도 입력한다는 것을 생각하지 못했다.
또한 해당 모델로 학습 시도 시, shape이 어떻게 나오는지 제대로 생각하지 못하고 모델의 형태만 논문과 유사하게 구축하여 오류도 발생했었다.
관련해서 교재 및 여러 코드 등 참고하며 수정 진행하였다.

## custom_transformer2.py

앞선 `custom_transformer1.py` 내용을 수정 및 보완한 Transformer 모델 파일이다.
생성형 AI(`Gemini-CLI`)의 도움으로 각 단계별 연산 형태가 어떻게 되는지 도움받아가며 shape을 맞춰나가고, 교재 등을 참고하며 누락된 부분 등을 추가해나갔다.
상세 내용은 블로그에 작성해두었다.

## dataset_generate.py

처음에 `seq_len`을 `vocab_size`(현재 코드 기준 `input_dim`)과 동일하게 맞춰야 하는 것으로 착각해서, `wmt14 de-en` 데이터로 학습을 못해서 직접 무작위 데이터셋을 만들어보려고 작성한 코드이다.
간단한 아이디어는 무작위 문자열을 서브 워드(토큰)이라 가정하고 생성한 후, 매칭시켜서 서로 다른 문자열 간에 규칙을 만들어준다. 이후 인덱싱 사전을 만들고, 무작위 문장들을 생성 후 역변환까지 하여 서로 규칙성을 갖는 데이터셋을 만든다.
최종적으로 `origin_sentences.txt`, `origin_to_index.json`, `target_sentences.txt`, `target_to_index.json` 파일을 생성해주고, 학습 시 해당 파일들을 불러와서 사용한다.

다만 `custom_transformer1.py` 모델 진행 시 발생했던 문제라서, 모델 자체에도 문제가 있다는 것을 나중에 알고 많이 수정하고나니 `wmt14 de-en` 데이터셋으로도 학습은 가능하여 사용하지 않게 되었다.

## pytorch_train_validation.ipynb

[파이토치 트랜스포머를 활용한 자연어 처리와 컴퓨터비전 심층학습](https://github.com/wikibook/pytorchtrf) 교재를 참고하여 `pytorch`에서 제공하는 모델을 활용해 테스트 진행한 파일이다.
직접 구축한 `custom_transformer2.py` 모델과 결과 비교를 위해 진행했는데, 입력하는 데이터 형태나 모델 방식 등 많은 부분이 좀 다르다보니 예상 외로 해당 결과가 커스텀 모델보다 낮게 나왔다.

## wmt14_preprocessing.py

`wmt14 de-en` 데이터를 어떻게 전처리 했는지 간단히 작성해둔 코드이다.

---

# 보완해야하는 부분

1. `pytorch_train_validation.ipynb`에서 결과가 더 낮게 나온 것을 보아, 모델 구축 부분에서 개선해야할 부분 등이 있을텐데, 하드웨어 문제로 진행을 안 했었다.
  추후 `pytorch`를 활용해서 `transformer` 모델을 구축할 일이 있다면, 상세하게 확인하며 메소드들을 해당 코드에서 좀 더 업데이트해야할 것이다.
2. 모델이 문장을 생성할 때 하나의 토큰씩 생성하며 문장을 완성할 때까지 계속해서 모델을 돌려야하여, 생성에 많은 시간이 소요된다.
  상용화된 모델들을 생각해보면 분명 predict 메소드가 훨씬 효율적으로 작성되어 있을 것으로 생각되어, 관련 내용들을 찾아보고 활용해야할 것으로 생각된다.