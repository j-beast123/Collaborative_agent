# CollabLLM (adapted, GRPO)

현재 공통 조건과 실행 순서는 [상위 실행 문서](../README.md), [공통 설정](../common/settings.json)을 따른다. 이전 train500 참조에서 사용자가 지정한 train120으로 전환했으며, 학습 중 test 평가는 하지 않고 완료한 최종 checkpoint만 한 번 평가한다.

기존 현재-policy tree의 동일 terminal conversations를 미래 보상 추정에 재사용한다. 공식 CollabLLM의 전체 학습 재현이 아니라 공통 GRPO에서의 보상 비교 adaptation이다. 추가 미래 sampling은 없다.

```text
S = 기존 Doctor FINAL outcome (0 또는 1)
L = 전체 visible Agent 발화를 공백으로 이어 cl100k_base로 센 토큰 수 / 1000
I = physician-assistant interactivity (0~1)
r_leaf = S + I - lambda_length * L
R_node = 기존 유효 descendant leaf의 동일 가중 평균
A_node = R_node - 같은 원래 sibling group의 reward 평균
```

`outcome is None`인 leaf는 모든 성분에서 제외한다. 유효 descendant가 없으면 node reward=None이다. Success-only (`I` weight=0, lambda=0)는 기존 outcome-only와 정확히 같으므로 독립적인 개선으로 보고하지 않는다.

[공식 reward](https://github.com/Wuyxin/collabllm/blob/7378aab4efb1168557e72699cc51aebb5b8bc656/collabllm/reward.py), [efficiency](https://github.com/Wuyxin/collabllm/blob/7378aab4efb1168557e72699cc51aebb5b8bc656/examples/metrics/efficiency.py), [interactivity](https://github.com/Wuyxin/collabllm/blob/7378aab4efb1168557e72699cc51aebb5b8bc656/examples/metrics/interactivity.py)를 근거로 적용했다. SHA256 등은 [adaptation_spec.json](adaptation_spec.json)에 저장한다.

길이는 prefix를 포함한 전체 terminal의 공개 Agent 발화만 센다. Doctor, private assessment, system prompt는 제외한다. `tiktoken==0.12.0`, `cl100k_base` vocabulary/pattern/special token 정의 hash를 실행 시 검사한다. Private/visible 모두 기존 GRPO action 및 실제 rollout 비용에는 포함된다.

[interactivity.txt](interactivity.txt)는 원래 engagement·clarity·needs 이해와 0/0.5/1 anchor를 유지한다. Physician 역할과 선택지 구분 예시로 바꿨고, 충분한 문제가 주어진 경우 추가 질문 없이도 높은 점수를 받을 수 있도록 했다. 질문 수·반복·단순 동의나 반박을 별도 보너스로 주지 않는다. Visible discussion만 입력하고 gold/outcome/FINAL/private/hidden status를 제외한다. 잘못된 JSON·범위 오류·요청 실패는 0점이나 leaf 제외로 바꾸지 않는다.

고정 judge·실제 입력·prompt·params로 성공 채점을 cache한다. 같은 visible 입력의 I는 ancestor와 중복 leaf에서 재사용하지만 leaf별 평균 횟수는 줄이지 않는다. 재시도와 실패 비용은 별도 audit에 남긴다.

Pilot은 원래 train 순서의 5/20/50개 prefix, 동일 초기 policy, optimizer update 없음이다. 5개는 동작 검증용이며 freeze할 수 없다. 최소 20개에서 case별 terminal 평균과 sibling 편차의 안정성을 검토하고 필요하면 50개까지 확대한다. 사전 진단 기준은 [명세](adaptation_spec.json)의 pilot_plan이며 통계적 대표성을 보장하지 않는다. Lambda는 자동 선택하지 않고 성분 분포, sibling 편차, 선택 후 reward/advantage 분포와 이유를 함께 고정한다. 실제 lambda는 아직 미정이다.

첫 정규 배치는 검증된 pilot 첫 5개 tree를 재사용한다. 나머지 확대 pilot tree는 policy update 뒤 replay하지 않는다. 따라서 보상 추정을 위한 추가 미래 sampling은 0이지만, 확대 pilot의 15/45개 초기-policy tree 비용은 별도로 발생한다. 모든 경로가 4턴이고 입력이 다르면 judge 최초 채점은 case당 최대 256회이며 실패·retry는 추가 요청을 만든다.

원래 runner를 독립 namespace로 불러와 parent reward hook만 연결했다. 교체된 target이 원래 record digest/checkpoint/GRPO updater로 전달된다. 학습 완료 후 원래 evaluator를 최종 checkpoint에만 연결한다. Code·judge·가중치·data·초기 model identity가 바뀌면 재개를 차단한다.

[데이터 audit](data_audit.json), [설정 비교](comparison_audit.json), [CPU 연결 검증](validation_report.json)은 실제 검증 범위를 구분해 기록한다. 현재 live pilot/학습/test 평가 완료를 뜻하지 않는다.
