# create_agent + CoT / CoT-SC 결합 가능성 검토

## 1. 현재 구현 현황

| 파일 | ReAct 방식 | CoT-SC |
|------|-----------|--------|
| `hotpotqa_langchain.ipynb` | `create_agent` (tool calling) | 없음 |
| `hotpotqa_langgraph.ipynb` | `StateGraph` (텍스트 파싱) | 없음 |
| `hotpotqa_cot_sc.ipynb` | `StateGraph` (텍스트 파싱) | **있음** |

현재 CoT-SC 결합 전략은 LangGraph 기반 ReAct에만 구현되어 있다.
`create_agent` 기반 ReAct에는 CoT-SC가 결합되어 있지 않다.

---

## 2. 결론: 결합 가능

**CoT-SC는 `create_agent`와 결합할 수 있다.**

이유: CoT-SC와 ReAct의 결합은 에이전트 *내부* 구조를 변경하는 것이 아니라, 에이전트 *외부*에서 두 방식을 조합하는 것이기 때문이다.

```
┌─────────────────────────────────────────────┐
│          결합 전략 (외부 래퍼)                │
│                                             │
│   ┌──────────────┐   ┌──────────────┐       │
│   │  CoT-SC      │   │  ReAct       │       │
│   │  (순수 LLM)  │   │  (에이전트)   │       │
│   └──────────────┘   └──────────────┘       │
│                           ↑                 │
│                    create_agent 또는         │
│                    StateGraph 무관           │
└─────────────────────────────────────────────┘
```

- **CoT-SC**는 도구를 사용하지 않는 순수 LLM 호출이므로, 에이전트 프레임워크와 무관하게 동작한다.
- **결합 전략** (방식 A: ReAct→CoT-SC, 방식 B: CoT-SC→ReAct)은 ReAct와 CoT-SC를 순차적으로 호출하는 래퍼 함수일 뿐이다.
- 따라서 ReAct 부분이 `create_agent`이든 `StateGraph`이든, 결합 전략의 로직은 동일하게 적용된다.

---

## 3. 구현 방법

### 3-1. CoT-SC 부분 (변경 없음)

CoT-SC는 에이전트와 무관하므로, 기존 `hotpotqa_cot_sc.ipynb`의 `cot_sc()` 함수를 그대로 사용한다.

```python
def cot_sc(question, n_samples=5):
    """CoT Self-Consistency: N번 샘플링 + 다수결."""
    answers = []
    for _ in range(n_samples):
        response = llm_sampling.invoke(COT_PROMPT + f"Question: {question}\n", stop=["\n\n"])
        answer = parse_cot_answer(response.content)
        answers.append(answer)
    final_answer, confidence = majority_vote(answers)
    return final_answer, confidence, answers
```

### 3-2. ReAct 부분 (`create_agent` 사용)

`hotpotqa_langchain.ipynb`의 `run_hotpotqa()` 함수를 래핑하여 실패 감지 로직을 추가한다.

```python
from langchain.agents import create_agent

agent = create_agent(
    model=model,
    tools=[search, lookup, finish],
    system_prompt=system_prompt,
)

def react_create_agent(question: str) -> tuple[str, dict]:
    """create_agent 기반 ReAct 실행."""
    wiki_state.reset()
    result = agent.invoke({
        "messages": [{"role": "user", "content": f"Question: {question}"}]
    })
    messages = result["messages"]
    answer = extract_answer(messages)

    # 실패 판정을 위한 정보 수집
    finish_called = any(
        tc["name"] == "finish"
        for msg in messages
        if hasattr(msg, "tool_calls") and msg.tool_calls
        for tc in msg.tool_calls
    )
    n_tool_calls = sum(
        len(msg.tool_calls)
        for msg in messages
        if hasattr(msg, "tool_calls") and msg.tool_calls
    )

    info = {
        "answer": answer,
        "finish_called": finish_called,
        "n_tool_calls": n_tool_calls,
        "done": finish_called and bool(answer),
    }
    return answer, info
```

### 3-3. 결합 전략

#### 방식 A: ReAct(`create_agent`) → CoT-SC Fallback

```python
def react_then_cot_sc(question, n_samples=5):
    """create_agent ReAct 실행 → 실패 시 CoT-SC fallback."""
    answer, info = react_create_agent(question)

    react_failed = not info["done"] or not answer

    if react_failed:
        cot_answer, confidence, all_answers = cot_sc(question, n_samples)
        info["method"] = "cot_sc_fallback"
        info["cot_confidence"] = confidence
        return cot_answer, info
    else:
        info["method"] = "react"
        return answer, info
```

#### 방식 B: CoT-SC → ReAct(`create_agent`) Fallback

```python
def cot_sc_then_react(question, n_samples=5, threshold=0.5):
    """CoT-SC 실행 → 신뢰도 낮으면 create_agent ReAct 실행."""
    cot_answer, confidence, all_answers = cot_sc(question, n_samples)

    if confidence >= threshold:
        return cot_answer, {"method": "cot_sc", "confidence": confidence}
    else:
        answer, info = react_create_agent(question)
        info["method"] = "react_after_low_confidence"
        info["cot_confidence"] = confidence
        return answer, info
```

---

## 4. `create_agent` vs `StateGraph`: 차이점

결합 전략 자체는 동일하게 적용되지만, ReAct 실패 감지 방식에 차이가 있다.

| 항목 | `create_agent` | `StateGraph` |
|------|---------------|-------------|
| 실패 감지 | 메시지 히스토리 사후 분석 | state 변수로 실시간 추적 |
| step 수 제한 | `recursion_limit` 파라미터 | `max_steps` state 변수 |
| finish 호출 확인 | `tool_calls` 에서 `finish` 이름 검색 | 텍스트에서 `finish[...]` 파싱 |
| trajectory 접근 | 메시지 리스트를 순회하여 재구성 | state에 trajectory 문자열 직접 보관 |
| 중간 개입 | 불가 (자동 루프) | 가능 (노드 단위 제어) |

### 핵심 차이: 실패 감지의 정밀도

**`StateGraph`**는 매 스텝마다 상태를 추적하므로 실패를 정밀하게 판정할 수 있다:

```python
react_failed = (
    not info["done"]                          # finish 안 함
    or not answer                              # 답 없음
    or "Invalid action" in info["trajectory"]  # 잘못된 액션
)
```

**`create_agent`**는 실행 완료 후 메시지를 분석하므로, 간접적으로 판정한다:

```python
react_failed = (
    not finish_called    # finish tool 호출 여부
    or not answer        # 답 추출 실패
)
```

실제로는 이 정도 차이가 결합 전략의 성능에 큰 영향을 주지 않는다.
`create_agent`가 정상적으로 `finish` tool을 호출했는지만 확인하면 충분하다.

---

## 5. `create_agent`에서 CoT를 에이전트 *내부*에 넣을 수 있는가?

위의 결합 전략은 CoT-SC와 ReAct를 **외부에서** 조합하는 방식이다.
그렇다면 에이전트 **내부**에 CoT 추론을 강화할 수 있는가?

### 가능: 시스템 프롬프트를 통한 CoT 유도

`create_agent`의 `system_prompt`에 CoT 스타일 추론을 유도하는 지시를 넣을 수 있다.
사실 현재 시스템 프롬프트의 "Thought는 현재 상황에 대해 추론하는 단계" 지시가 이미 CoT의 역할을 한다.

```python
system_prompt = """...
각 도구를 호출하기 전에 반드시 단계적으로 추론(chain-of-thought)하시오.
추론에서는 다음을 포함하시오:
1. 현재까지 알게 된 정보 정리
2. 아직 모르는 정보 식별
3. 다음 행동의 근거 설명
...
"""
```

이것은 ReAct 자체가 이미 CoT의 요소를 포함하고 있기 때문에 자연스럽다.
(논문에서도 ReAct = Reasoning + Acting, 즉 CoT + Action으로 설명한다)

### 불가능: `create_agent` 내부에서 Self-Consistency

Self-Consistency(SC)는 **같은 질문에 대해 여러 번 독립 샘플링**하는 것이다.
`create_agent`는 단일 실행 경로를 가지므로, 내부에서 SC를 수행할 수 없다.
SC는 반드시 외부 래퍼에서 구현해야 한다.

---

## 6. 요약

| 질문 | 답 |
|------|-----|
| `create_agent` + CoT-SC 결합 가능? | **가능** (외부 래퍼로 구현) |
| `create_agent` + 방식 A (ReAct→CoT-SC) 가능? | **가능** (finish 호출 여부로 실패 감지) |
| `create_agent` + 방식 B (CoT-SC→ReAct) 가능? | **가능** (CoT-SC 결과에 따라 에이전트 호출) |
| `create_agent` 내부에 CoT 강화 가능? | **가능** (시스템 프롬프트로 유도, 이미 적용됨) |
| `create_agent` 내부에 SC 가능? | **불가** (SC는 외부 다중 실행이 필수) |
| StateGraph 대비 단점? | 실패 감지가 간접적, 중간 개입 불가 |
| StateGraph 대비 장점? | 코드가 간결, 도구 호출이 자동화됨 |

결론적으로, **`create_agent`를 사용해도 CoT-SC 결합 전략은 동일하게 구현 가능**하다.
현재 `hotpotqa_cot_sc.ipynb`의 결합 전략 코드에서 ReAct 부분만 `create_agent` 버전으로 교체하면 된다.
