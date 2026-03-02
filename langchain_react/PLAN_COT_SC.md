# ReAct + CoT-SC 구현 계획

## 1. 현재 상태

### 프롬프트 파일 (`prompts_naive.json`)에 이미 있는 것

| 키 | 방식 | 설명 |
|---|---|---|
| `webthink_simple6` | **ReAct** | Thought + Action + Observation (현재 사용 중) |
| `cotqa_simple6` | **CoT** | Thought + Answer (추론만, 행동 없음) |
| `webqa_simple6` | **Direct QA** | Question → Answer (추론도 행동도 없음) |
| `webact_simple6` | **Act-only** | Action + Observation (추론 없음) |

### 구현되어 있지 않은 것

- **CoT-SC** (Chain-of-Thought Self-Consistency): CoT를 여러 번 샘플링하고 다수결 투표
- **ReAct + CoT-SC**: 논문의 핵심 결합 전략

---

## 2. 배경: 논문의 방법론

### CoT-SC란?

Wang et al. (2022) "Self-Consistency Improves Chain of Thought Reasoning"에서 제안.

```
같은 질문에 대해 temperature > 0으로 N번 CoT를 실행
→ N개의 답을 수집
→ 다수결 투표(majority vote)로 최종 답 결정
```

핵심: greedy decoding(temperature=0)은 하나의 추론 경로만 탐색하지만,
sampling(temperature>0)은 다양한 경로를 탐색하여 더 robust한 답을 얻는다.

### ReAct + CoT-SC 결합 전략 (논문 Section 4)

논문에서는 두 가지 결합 방식을 제시:

#### 방식 A: ReAct → CoT-SC (Fallback)

```
1. ReAct 실행
2. ReAct가 정상 종료 (finish[answer]) → ReAct 답 사용
3. ReAct가 실패 (max_steps 초과, 반복 루프 등) → CoT-SC 답으로 대체
```

#### 방식 B: CoT-SC → ReAct (Confidence 기반)

```
1. CoT-SC 실행 (N번 샘플링 + 다수결)
2. 다수결 일치율이 높으면 (≥ threshold) → CoT-SC 답 사용
3. 다수결 일치율이 낮으면 (< threshold) → ReAct 실행하여 답 사용
```

논문 결과에서는 **방식 A + B를 상황에 맞게** 사용하여 최고 성능 달성.

---

## 3. 구현 범위

### 구현할 것

1. **CoT-SC**: CoT 프롬프트 N번 샘플링 + 다수결 투표
2. **ReAct + CoT-SC (방식 A)**: ReAct 실패 시 CoT-SC fallback
3. **CoT-SC → ReAct (방식 B)**: CoT-SC 신뢰도 낮을 때 ReAct 실행

### 파일 계획

```
langchain_react/
├── tools.py              # [기존] 그대로
├── eval_utils.py         # [수정] majority_vote 함수 추가
├── hotpotqa_cot_sc.ipynb # [신규] CoT-SC + ReAct+CoT-SC 결합 노트북
└── ...
```

---

## 4. 구현 상세

### 4-1. `eval_utils.py` 수정 - majority_vote 추가

```python
from collections import Counter

def majority_vote(answers: list[str]) -> tuple[str, float]:
    """다수결 투표. (최다 답, 일치율) 반환."""
    normalized = [normalize_answer(a) for a in answers]
    counter = Counter(normalized)
    most_common, count = counter.most_common(1)[0]
    confidence = count / len(normalized)
    # 원본 형태의 답을 반환 (normalize 전)
    for a, n in zip(answers, normalized):
        if n == most_common:
            return a, confidence
    return answers[0], confidence
```

### 4-2. `hotpotqa_cot_sc.ipynb` - 메인 노트북

#### 셀 구성

**Part 1: CoT-SC 구현**

```python
# CoT 프롬프트 로딩 (cotqa_simple6 사용)
cot_examples = prompt_dict["cotqa_simple6"]
cot_instruction = """질문에 대해 단계별로 추론하여 답을 제시하시오.
다음은 몇 가지 예시이다.
"""
cot_prompt = cot_instruction + cot_examples

def cot_sc(question: str, n_samples: int = 5, temperature: float = 0.7) -> tuple[str, float]:
    """CoT Self-Consistency: N번 샘플링 + 다수결 투표."""
    answers = []
    for _ in range(n_samples):
        response = llm.invoke(
            cot_prompt + f"Question: {question}\n",
            temperature=temperature,
            stop=["\n\n"],
        )
        # "Answer: ..." 파싱
        answer = parse_cot_answer(response.content)
        answers.append(answer)

    final_answer, confidence = majority_vote(answers)
    return final_answer, confidence
```

**Part 2: ReAct + CoT-SC 방식 A (Fallback)**

```python
def react_with_cot_sc_fallback(question: str, n_samples: int = 5) -> tuple[str, dict]:
    """ReAct 실행 → 실패 시 CoT-SC로 대체."""
    # 1) ReAct 실행
    answer, info = webthink(question)

    # 2) ReAct 성공 여부 판단
    react_failed = (
        not info["answer"]                    # 답 없음
        or info["steps"] >= 7                  # max steps 도달
        or "Invalid action" in info["trajectory"]  # 잘못된 액션
    )

    if react_failed:
        # 3) CoT-SC fallback
        answer, confidence = cot_sc(question, n_samples=n_samples)
        info["method"] = "cot_sc_fallback"
        info["cot_confidence"] = confidence
    else:
        info["method"] = "react"

    return answer, info
```

**Part 3: CoT-SC → ReAct 방식 B (Confidence 기반)**

```python
def cot_sc_with_react_fallback(
    question: str,
    n_samples: int = 5,
    confidence_threshold: float = 0.5,
) -> tuple[str, dict]:
    """CoT-SC 실행 → 신뢰도 낮으면 ReAct로 대체."""
    # 1) CoT-SC 실행
    cot_answer, confidence = cot_sc(question, n_samples=n_samples)

    if confidence >= confidence_threshold:
        # 다수결 일치율이 높으면 CoT-SC 답 사용
        return cot_answer, {"method": "cot_sc", "confidence": confidence}
    else:
        # 신뢰도 낮으면 ReAct 실행
        answer, info = webthink(question)
        info["method"] = "react_after_low_confidence"
        info["cot_confidence"] = confidence
        info["cot_answer"] = cot_answer
        return answer, info
```

**Part 4: 비교 평가**

```python
# 4가지 방식을 동일 500개 샘플로 비교
methods = {
    "ReAct only":       lambda q: webthink(q),
    "CoT-SC only":      lambda q: cot_sc(q),
    "ReAct→CoT-SC":     lambda q: react_with_cot_sc_fallback(q),
    "CoT-SC→ReAct":     lambda q: cot_sc_with_react_fallback(q),
}
# 각 방식의 EM, F1 비교 테이블 출력
```

---

## 5. 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `n_samples` | 5 | CoT-SC 샘플링 횟수. 논문에서는 5~10 사용 |
| `temperature` | 0.7 | 샘플링 다양성. 0이면 항상 같은 답, 높을수록 다양 |
| `confidence_threshold` | 0.5 | 방식 B에서 CoT-SC 신뢰 기준. 5개 중 3개 이상 일치 시 사용 |

---

## 6. 구현 순서

| 단계 | 작업 | 파일 |
|------|------|------|
| 1 | `eval_utils.py`에 `majority_vote()` 함수 추가 | `eval_utils.py` |
| 2 | `hotpotqa_cot_sc.ipynb` 작성 - CoT-SC, 방식 A, 방식 B, 비교 평가 | `hotpotqa_cot_sc.ipynb` |

---

## 7. 비용/시간 고려

CoT-SC는 같은 질문을 N번 호출하므로 API 비용이 N배 증가한다.

| 방식 | 질문당 LLM 호출 수 | 500개 기준 |
|------|-------------------|-----------|
| ReAct only | 3~7회 (step 수만큼) | 1,500~3,500회 |
| CoT-SC only (n=5) | 5회 | 2,500회 |
| ReAct → CoT-SC | 3~12회 (실패 시 +5) | 1,500~6,000회 |
| CoT-SC → ReAct | 5~12회 (저신뢰 시 +3~7) | 2,500~6,000회 |

소수 샘플(10~20개)로 먼저 테스트 후 전체 평가를 권장한다.
