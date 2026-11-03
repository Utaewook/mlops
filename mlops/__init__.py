"""MLOps 입문 실습 과제 패키지.

세 유형(Time Series, Vision, LLM)의 과제가 같은 인터페이스(`mlops.tasks.base.Task`)를 따른다.
데이터는 시간 순서가 있는 슬라이스로 나눠 두고,
학습·평가·내보내기는 `mlops.runner`가 공통으로 돌린다.

    python -m mlops prepare timeseries
    python -m mlops train timeseries --param learning_rate=0.03
"""
