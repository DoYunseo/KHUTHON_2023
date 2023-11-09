from keybert import KeyBERT
from kiwipiepy import Kiwi
from transformers import BertModel

doc = """ 임진왜란은 1592년(선조 25)부터 1598년까지 2차에 걸쳐서 우리나라에 침입한 일본과의 싸움이다. 

엄청난 시련을 겪으면서도 끈질긴 저항으로 이겨내고 각성과 자기성찰을 바탕으로 민족의 운명을 새로 개척해나간 계기가 된 전쟁이다.
"""

# kiwi를 이용하여 명사로 바꾸는 함수
def noun_extractor(text):
  results = []
  kiwi = Kiwi()
  result = kiwi.analyze(text)
  for token, pos, _, _ in result[0][0]:
      if len(token) != 1 and pos.startswith('N') or pos.startswith('SL'):
          results.append(token)
  text = ' '.join(results)
  return text

#키워드 추출 모델 (문서랑 키워드 개수 파라미터로 입력)
def keybert(text,n):
  model = BertModel.from_pretrained('skt/kobert-base-v1')
  kw_model = KeyBERT(model)
  keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words=None, top_n=n)
  #키워드만 리스트로 만들기
  keyword_list = [keywords[i][0] for i in range(len(keywords))]
  return keyword_list

# 명사로 변환된 결과 출력
text = noun_extractor(doc)
keybert(text, 5)