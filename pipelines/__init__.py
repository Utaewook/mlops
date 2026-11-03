"""Dagster 코드 위치. 6편 실습에서 자산(asset), 리소스, 스케줄, 센서를 정의해 아래에 넣는다.

dagster-code 컨테이너가 저장소 전체를 /workspace에 읽기 전용으로 붙이므로,
3~5편 실습 코드를 여기서 import해서 쓸 수 있다.
"""

import dagster as dg

defs = dg.Definitions()
