## Graph Neural Network 

**DeepWalk**

2014년 KDD에 발표된 논문으로, `Node Embedding`이라는 개념을 언급하는 논문입니다. 자연어처리를 하시는 분들이라면 `Embedding`이라는 개념이 쉽게 와닿을 수 있습니다. 
`Embedding`은 `One-hot encoding`, `Look-up Table`과 같이 아무런 의미가 없는 `numeric data`를 `embedding` 차원으로 `mapping`함으로써 그 수치가 의미를 가지며, 
데이터의 `feature`를 보다 잘 이해할 수 있다는 개념으로 받아들일 수 있습니다. 내적이 가능하거나 하는 등의 추가적인 이점이 존재합니다. Word2Vec과 동일하게 Skip-gram과 CBoW를 
사용하여 단어를 `Node`로 문장을 `Walk`로 매칭한 후 학습합니다. 이를 통해 `Embedding`된 `Node`혹은 `Walk`를 시각화할 수도 있습니다. 

![image]()
