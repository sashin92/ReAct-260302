import time
import requests
from bs4 import BeautifulSoup
from langchain.tools import tool


def clean_str(p):
    return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


class WikiState:
    """Wikipedia 페이지 상태를 관리하는 클래스."""

    def __init__(self):
        self.reset()
        self.search_time = 0
        self.num_searches = 0

    def reset(self):
        self.page = None
        self.lookup_keyword = None
        self.lookup_list = []
        self.lookup_cnt = 0

    def reset_stats(self):
        self.search_time = 0
        self.num_searches = 0


wiki_state = WikiState()


@tool
def search(entity: str) -> str:
    """Wikipedia에서 entity를 검색하고 첫 번째 문단을 반환한다. 존재하지 않으면 유사한 entity 목록을 반환한다."""
    entity_ = entity.replace(" ", "+")
    search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
    old_time = time.time()
    # TODO: 이메일 설정 필요
    response_text = requests.get(
        search_url,
        headers={
            "User-Agent": "MyResearchBot/1.0 (example@gmail.com)",
            "Accept-Language": "en-US,en;q=0.9",
        },
    ).text
    wiki_state.search_time += time.time() - old_time
    wiki_state.num_searches += 1

    soup = BeautifulSoup(response_text, "html.parser")
    result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})

    if result_divs:
        result_titles = [clean_str(div.get_text().strip()) for div in result_divs]
        wiki_state.page = None
        return f"Could not find {entity}. Similar: {result_titles[:5]}."
    else:
        page_parts = [
            p.get_text().strip()
            for p in soup.find_all("p") + soup.find_all("ul")
        ]
        if any("may refer to:" in p for p in page_parts):
            return search.invoke({"entity": f"[{entity}]"})

        page_text = ""
        for p in page_parts:
            if len(p.split(" ")) > 2:
                page_text += clean_str(p)
                if not p.endswith("\n"):
                    page_text += "\n"

        wiki_state.page = page_text
        wiki_state.lookup_keyword = None
        wiki_state.lookup_list = []
        wiki_state.lookup_cnt = 0

        paragraphs = [p.strip() for p in page_text.split("\n") if p.strip()]
        sentences = []
        for p in paragraphs:
            sentences += p.split(". ")
        sentences = [s.strip() + "." for s in sentences if s.strip()]
        return " ".join(sentences[:5])


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
            sentences += p.split(". ")
        sentences = [s.strip() + "." for s in sentences if s.strip()]
        wiki_state.lookup_list = [
            s for s in sentences if keyword.lower() in s.lower()
        ]
        wiki_state.lookup_cnt = 0

    if wiki_state.lookup_cnt >= len(wiki_state.lookup_list):
        return "No more results."

    result = (
        f"(Result {wiki_state.lookup_cnt + 1} / {len(wiki_state.lookup_list)}) "
        + wiki_state.lookup_list[wiki_state.lookup_cnt]
    )
    wiki_state.lookup_cnt += 1
    return result


@tool
def finish(answer: str) -> str:
    """최종 답을 반환하고 과제를 종료한다. 답을 확정했을 때 반드시 이 도구를 호출하시오."""
    return f"Episode finished, answer: {answer}"
