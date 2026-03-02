# ReAct 프로젝트 LangChain / LangGraph 마이그레이션 계획

## 1. 현재 구조 분석

현재 프로젝트는 **수동으로 ReAct 루프를 구현**하고 있다:

```
[프롬프트 조립] → [LLM 호출 (Thought+Action 생성)] → [Action 파싱] → [WikiEnv.step() 실행] → [Observation 수집] → 반복
```

### 핵심 컴포넌트
| 컴포넌트 | 현재 파일 | 역할 |
|---|---|---|
| LLM 호출 | `hotpotqa.ipynb` (`llm()` 함수) | OpenAI API 직접 호출 |
| 환경 (Tool) | `wikienv.py` | Wikipedia 검색/조회 환경 |
| 래퍼 | `wrappers.py` | HotPotQA/FEVER 데이터 로딩, 평가, 로깅 |
| 프롬프트 | `prompts/prompts_naive.json` | Few-shot 예시 |
| ReAct 루프 | `hotpotqa.ipynb` (`webthink()`) | Thought→Action→Observation 수동 반복 |

### 현재 코드의 한계
- Action 파싱을 문자열 split으로 수동 처리
- 에러 핸들링이 단순 (badcall 카운트만)
- 프롬프트에 전체 히스토리를 문자열로 연결 (토큰 관리 없음)
- 환경이 gym.Env에 종속

---

## 2. 마이그레이션 대상

두 가지 버전을 각각 별도 노트북으로 구현한다:

### A. `hotpotqa_langchain.ipynb` - LangChain `create_agent` 버전
### B. `hotpotqa_langgraph.ipynb` - LangGraph 커스텀 그래프 버전

---

## 3. 공통 작업

### 3-1. Tool 정의 (`tools.py`)

현재 `wikienv.py`의 3가지 액션을 LangChain Tool로 변환한다:

```python
from langchain.tools import tool
import requests
from bs4 import BeautifulSoup

# Wikipedia 페이지 상태를 관리하는 전역 객체
class WikiState:
    def __init__(self):
        self.page = None
        self.lookup_keyword = None
        self.lookup_list = []
        self.lookup_cnt = 0

    def reset(self):
        self.page = None
        self.lookup_keyword = None
        self.lookup_list = []
        self.lookup_cnt = 0

wiki_state = WikiState()

@tool
def search(entity: str) -> str:
    """Wikipedia에서 entity를 검색하고 첫 번째 문단을 반환한다.
    존재하지 않으면 유사한 entity 목록을 반환한다."""
    # wikienv.py의 search_step 로직 이식
    entity_ = entity.replace(" ", "+")
    search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
    response_text = requests.get(
        search_url,
        headers={"User-Agent": "MyResearchBot/1.0 (your@email.com)"}
    ).text
    soup = BeautifulSoup(response_text, "html.parser")
    result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})

    if result_divs:
        result_titles = [div.get_text().strip() for div in result_divs]
        wiki_state.page = None
        return f"Could not find {entity}. Similar: {result_titles[:5]}."
    else:
        page_parts = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
        page_text = ""
        for p in page_parts:
            if len(p.split(" ")) > 2:
                page_text += p + "\n"
        wiki_state.page = page_text
        wiki_state.lookup_keyword = None
        wiki_state.lookup_list = []
        wiki_state.lookup_cnt = 0
        # 첫 5문장 반환
        paragraphs = [p.strip() for p in page_text.split("\n") if p.strip()]
        sentences = []
        for p in paragraphs:
            sentences += p.split('. ')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        return ' '.join(sentences[:5])

@tool
def lookup(keyword: str) -> str:
    """현재 Wikipedia 페이지에서 keyword가 포함된 다음 문장을 반환한다."""
    if wiki_state.page is None:
        return "No page loaded. Please search first."
    if wiki_state.lookup_keyword != keyword:
        wiki_state.lookup_keyword = keyword
        paragraphs = [p.strip() for p in wiki_state.page.split("\n") if p.strip()]
        sentences = []
        for p in paragraphs:
            sentences += p.split('. ')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        wiki_state.lookup_list = [s for s in sentences if keyword.lower() in s.lower()]
        wiki_state.lookup_cnt = 0
    if wiki_state.lookup_cnt >= len(wiki_state.lookup_list):
        return "No more results."
    result = f"(Result {wiki_state.lookup_cnt + 1} / {len(wiki_state.lookup_list)}) " + wiki_state.lookup_list[wiki_state.lookup_cnt]
    wiki_state.lookup_cnt += 1
    return result

@tool
def finish(answer: str) -> str:
    """최종 답을 반환하고 과제를 종료한다."""
    return f"FINAL_ANSWER: {answer}"
```

### 3-2. 평가 유틸 (`eval_utils.py`)

`wrappers.py`에서 평가 로직만 분리한다:

```python
# normalize_answer, f1_score 함수를 그대로 가져옴
# HotPotQA 데이터 로딩 함수 추가
```

### 3-3. 의존성 추가 (`requirements.txt`)

```
langchain>=0.3
langchain-openai>=0.3
langgraph>=0.3
```

---

## 4. Plan A: LangChain `create_agent` 버전

### 파일: `hotpotqa_langchain.ipynb`

`create_agent`를 사용하여 가장 간결하게 구현하는 버전이다.

### 구현 흐름

```
[create_agent(model, tools, system_prompt)] → [agent.invoke(question)] → 자동 ReAct 루프 → 결과
```

### 핵심 코드 구조

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from tools import search, lookup, finish, wiki_state

# 1) 모델 설정
model = ChatOpenAI(model="gpt-4.1-mini", temperature=0, max_tokens=1000)

# 2) 시스템 프롬프트 (한국어 instruction)
system_prompt = """Thought, Action, Observation 단계를 번갈아가며 수행하여 질문 응답 과제를 해결하시오.

사용 가능한 도구:
- search: Wikipedia에서 entity를 검색
- lookup: 현재 문서에서 keyword가 포함된 문장을 조회
- finish: 최종 답을 제출

반드시 finish 도구를 사용하여 최종 답을 제출하시오."""

# 3) 에이전트 생성
agent = create_agent(
    model=model,
    tools=[search, lookup, finish],
    system_prompt=system_prompt,
)

# 4) 실행
def run_hotpotqa(question: str) -> str:
    wiki_state.reset()
    result = agent.invoke({
        "messages": [{"role": "user", "content": f"Question: {question}"}]
    })
    # 마지막 메시지에서 답 추출
    return extract_answer(result["messages"])

# 5) 평가 루프 (500개 샘플)
for idx in idxs[:500]:
    question, gt_answer = data[idx]
    pred = run_hotpotqa(question)
    em = (normalize_answer(pred) == normalize_answer(gt_answer))
    ...
```

### 장점
- 코드가 매우 간결 (ReAct 루프를 수동으로 구현할 필요 없음)
- `create_agent`가 tool calling, 에러 핸들링을 자동 처리
- 스트리밍으로 중간 과정 확인 가능

### 한계
- ReAct 루프의 세부 동작 커스터마이징이 제한적
- 최대 반복 횟수 등 세밀한 제어가 어려움

---

## 5. Plan B: LangGraph 커스텀 그래프 버전

### 파일: `hotpotqa_langgraph.ipynb`

LangGraph로 ReAct 루프를 **명시적 그래프**로 구현하는 버전이다. 현재 `webthink()` 함수의 로직을 그래프 노드로 변환한다.

### 그래프 구조

```
        ┌──────────────┐
        │   START      │
        │ (질문 입력)   │
        └──────┬───────┘
               │
               v
        ┌──────────────┐
        │   reasoning  │ ◄──────────────┐
        │ (LLM 추론)   │                │
        └──────┬───────┘                │
               │                        │
               v                        │
        ┌──────────────┐                │
        │   route      │                │
        │ (분기 판단)   │                │
        └──┬───┬───┬───┘                │
           │   │   │                    │
     search│   │   │finish              │
           │   │lookup                  │
           v   v   v                    │
        ┌──────────────┐                │
        │  tool_exec   │                │
        │ (도구 실행)   │ ───────────────┘
        └──────┬───────┘    (finish가 아닌 경우)
               │
               v (finish인 경우)
        ┌──────────────┐
        │     END      │
        │ (답 반환)     │
        └──────────────┘
```

### 핵심 코드 구조

```python
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
import operator

# 1) 상태 정의
class AgentState(TypedDict):
    question: str
    messages: Annotated[list, operator.add]  # 전체 대화 히스토리
    current_step: int
    max_steps: int
    answer: str | None
    done: bool

# 2) 노드 함수들
def reasoning_node(state: AgentState) -> dict:
    """LLM을 호출하여 Thought + Action을 생성"""
    model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    step = state["current_step"]
    # 프롬프트 조립 (few-shot 예시 + 히스토리)
    prompt = build_prompt(state)
    response = model.invoke(prompt)
    thought, action = parse_thought_action(response.content, step)
    return {
        "messages": [
            {"role": "assistant", "content": f"Thought {step}: {thought}"},
            {"role": "assistant", "content": f"Action {step}: {action}"},
        ],
    }

def tool_exec_node(state: AgentState) -> dict:
    """파싱된 Action을 실행하고 Observation을 반환"""
    last_action = get_last_action(state["messages"])
    if last_action.startswith("search["):
        entity = last_action[7:-1]
        obs = search.invoke({"entity": entity})
    elif last_action.startswith("lookup["):
        keyword = last_action[7:-1]
        obs = lookup.invoke({"keyword": keyword})
    elif last_action.startswith("finish["):
        answer = last_action[7:-1]
        return {"answer": answer, "done": True, "messages": [...]}
    step = state["current_step"]
    return {
        "messages": [{"role": "tool", "content": f"Observation {step}: {obs}"}],
        "current_step": step + 1,
    }

def should_continue(state: AgentState) -> str:
    """그래프 분기: 계속할지 종료할지 결정"""
    if state.get("done") or state["current_step"] > state["max_steps"]:
        return "end"
    return "continue"

# 3) 그래프 조립
graph = StateGraph(AgentState)
graph.add_node("reasoning", reasoning_node)
graph.add_node("tool_exec", tool_exec_node)

graph.add_edge(START, "reasoning")
graph.add_edge("reasoning", "tool_exec")
graph.add_conditional_edges(
    "tool_exec",
    should_continue,
    {"continue": "reasoning", "end": END},
)

app = graph.compile()

# 4) 실행
result = app.invoke({
    "question": "Which magazine was started first?",
    "messages": [],
    "current_step": 1,
    "max_steps": 7,
    "answer": None,
    "done": False,
})
```

### 장점
- ReAct 루프의 각 단계를 **노드로 명시적 제어** 가능
- 최대 반복 횟수, 분기 조건 등 세밀한 커스터마이징
- 중간 상태 추적/디버깅 용이
- 현재 `webthink()` 함수의 로직을 가장 충실하게 재현

### 한계
- 코드량이 `create_agent` 대비 많음
- 그래프 구조를 직접 설계해야 함

---

## 6. 파일 구조 (최종)

```
ReAct-260302/
├── tools.py                      # [신규] LangChain Tool 정의 (search, lookup, finish)
├── eval_utils.py                 # [신규] 평가 유틸 (normalize_answer, f1_score, 데이터 로딩)
├── hotpotqa_langchain.ipynb      # [신규] Plan A: create_agent 버전
├── hotpotqa_langgraph.ipynb      # [신규] Plan B: LangGraph 그래프 버전
├── hotpotqa.ipynb                # [기존] 원본 유지
├── wikienv.py                    # [기존] 원본 유지 (참조용)
├── wrappers.py                   # [기존] 원본 유지 (참조용)
├── requirements.txt              # [수정] langchain, langgraph 의존성 추가
└── ...
```

---

## 7. 구현 순서

| 단계 | 작업 | 파일 |
|------|------|------|
| 1 | `tools.py` 작성 - Wikipedia Tool 3종 정의 | `tools.py` |
| 2 | `eval_utils.py` 작성 - 평가/데이터 유틸 분리 | `eval_utils.py` |
| 3 | `requirements.txt` 업데이트 | `requirements.txt` |
| 4 | `hotpotqa_langchain.ipynb` 작성 - create_agent 버전 | `hotpotqa_langchain.ipynb` |
| 5 | `hotpotqa_langgraph.ipynb` 작성 - LangGraph 그래프 버전 | `hotpotqa_langgraph.ipynb` |
| 6 | 각 노트북에서 소수 샘플로 동작 테스트 | 노트북 내 |

---

## 8. 두 버전 비교 요약

| 항목 | LangChain (`create_agent`) | LangGraph (커스텀 그래프) |
|------|---------------------------|--------------------------|
| 코드량 | 적음 (약 50줄) | 많음 (약 150줄) |
| 커스터마이징 | 제한적 | 자유로움 |
| ReAct 루프 | 자동 (내부 처리) | 명시적 (노드/엣지) |
| 최대 반복 제어 | middleware 필요 | `max_steps` 직접 제어 |
| 디버깅 | 스트리밍으로 확인 | 노드별 상태 추적 |
| 학습 가치 | LangChain 에이전트 패턴 이해 | 그래프 기반 워크플로우 이해 |
| 원본 대비 | 구조가 크게 다름 | 원본 `webthink()` 로직과 유사 |
