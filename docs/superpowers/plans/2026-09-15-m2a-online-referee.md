# M2a 온라인 심판 구현 계획 — 대회 채점기와 같은 판정을 한 프레임씩

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 주행 한 판을 20 Hz 로 받으며 대회 채점기(`score_fma.py`)와 같은 감점 판정을 내는 온라인 심판을 만든다. 일부러 규칙을 어기는 대본 운전자와 선생님 주행으로 "심판 판정 = 채점기 판정"을 검증한다.

**Architecture:** 정답은 규칙 스택의 `eval/score_fma.py` 다. `vtd_rl/referee/oracle.py` 는 `score_fma.main()` 과 같은 조립으로 한 판 전체를 사후 채점하고, 감점 호출(`sheet.hit`)을 모두 기록한다. `vtd_rl/referee/judges/` 의 판정기들은 같은 행을 한 프레임씩 받아 같은 감점 호출을 낸다. 검증은 두 호출 목록의 `(구간, 항목, 등급)` 다중집합이 같은지 본다. 판정 문턱과 도우미는 `score_fma`·`check_lanes` 에서 import 한다.

**Tech Stack:** Python 3.10 표준 라이브러리, pytest. Gymnasium·numpy 는 M2b 에서 들어온다.

**Spec:** `docs/superpowers/specs/2026-09-15-vtd-rl-urban-driving-design.md` (§5 보상, §8.1 심판 대조, §11 M2)

## Global Constraints

- 커밋 메시지에 Claude 표기(`Co-Authored-By`, `Claude-Session`, "Generated with")를 **넣지 않는다**.
- 규칙 스택 서브모듈(`third_party/rule_stack`)은 **수정하지 않는다**. import 하고 읽기만 한다.
- 공개 레포에 주최측 자료(`*.xodr`, 시나리오 XML, 강의·답변), 주행 CSV, `runs/` 를 커밋하지 않는다. 지도는 서브모듈 안의 `map/HL_FMA_VTD_LivingLab.xodr` 를 **읽기만** 한다.
- 테스트는 `env -u PYTHONPATH .venv/bin/pytest` 로 돌리고 **종료 코드 0** 으로 판정한다. "N passed" 문자열로 판정하지 않는다.
- 시뮬 주기 20 Hz(`dt = 0.05`), 구간 수는 대회와 같은 **5** 다.
- **판정 문턱은 `score_fma`·`check_lanes` 모듈 속성으로 import 한다.** `score_fma` 함수 안에 지역 상수나 숫자로만 있는 값(`PED_STOP_WIN=6.0`, `LC_WIN=2.5`, `LANE_W=3.0`, `WIN=8.0`, 신호 상태 번호, 사람 키 1.2 m, 횡단보도 반경 3.0 m, 출발 제외 8 m 등)만 옮겨 적는다. 옮겨 적은 자리에는 원본 함수 이름을 주석으로 단다.
- **일치의 정의:** 한 판에서 심판이 낸 `(sec, item, level)` 다중집합과 채점기가 `sheet.hit` 에 넘긴 `(sec, item, level)` 다중집합이 같아야 한다. 항목 15 는 `{sec: [t, ...]}` 사전이 같아야 한다. `why` 문자열과 호출 순서는 비교하지 않는다.
- **심판은 미래 행을 보지 않는다.** `step(row)` 는 그때까지 받은 행만 쓴다. 채점기가 뒤 행을 보는 항목은 그만큼 기다렸다가 판정을 낸다. ⑧ 은 정차 구간이 끝날 때, ⑩ 은 1 초 뒤, ③ 은 2.5 초 뒤, ⑬ 은 최대 8 초 뒤에 낸다. 판 끝의 `finish()` 가 대기 중인 판정을 모두 낸다.
- **판정 시점 하나는 일부러 다르게 둔다(⑦⑨).** 채점기는 판 끝에 "그 (신호, 상태)에서 한 번이라도 정지선 앞에 섰나"를 본다. 그래서 같은 (신호, 상태)로 **다시 다가가 서면** 앞서 신호를 무시하고 지나간 통과도 면책된다. 심판은 선을 넘는 프레임에 확정한다(보상이 늦게 오지 않게). 검증 판과 선생님 코스에는 이런 재접근이 없다. 재접근이 생겨 일치 검증이 깨지면 이 규칙 때문인지 먼저 확인한다.
- 대본 판의 수치(속도, 정지 위치, 오프셋, 액터 위치, 자르는 구간)는 기대 항목이 채점기에서 일어나도록 조정해도 된다. **채점기·심판·월드 문턱은 바꾸지 않는다.**

## 스펙과 다른 점(M2 범위)

- **지도 캐시를 만들지 않는다.** 스펙은 xodr 파싱이 몇 분 걸린다고 봤다. 실측(2026-09-15)은 `Map(xodr)` 0.34 초, 행당 차로 판정 약 40 µs 였다. 프로세스마다 한 번 읽는다.
- **M2 를 계획 두 개로 나눈다.** M2a(이 문서)는 온라인 심판과 일치 검증이다. M2b 는 Gymnasium 환경(관측, 행동, 보상, 학생 CSV)이다. 보상이 심판 출력을 쓰므로 심판을 먼저 만든다.
- **항목 15(리스폰)는 오프라인 세계에서 일어나지 않는다.** 도로 이탈이 판을 끝내기 때문이다. 판정기는 만들되 합성 행으로만 검증한다.

## 참고 자료(구현자 필독)

- `third_party/rule_stack/eval/score_fma.py` — 정답 채점기. `Sections`, `Sheet`, `load`, `spans`, `item_*`, `main()` 조립.
- `third_party/rule_stack/eval/check_lanes.py` — `classify(rows, mp)` 는 행마다 독립으로 차로 판정 키를 붙인다(제자리 수정). `lane_change_events(rows)` 는 직전 행과만 비교해 사건을 찾는다. 사건 뒤 행을 훑는 부분은 설명용 `lane_change_to_lane` 만 채운다.
- 채점기 함정:
  - CSV `sig` 는 **자차 지시등**(0/1/2)이고, 신호등 상태는 `tl`(`tl_state`)이다.
  - ⑦⑨ 는 **앞범퍼**(`FRONT=3.808`) 기준 거리를 쓴다. ⑧ 은 뒷축 기준 거리를 쓴다. ⑫ 는 `FRONT / 2` 를 "차체 중앙"으로 쓴다. `EGO_CX` 로 고치지 않는다.
  - ⑧ 면책은 `reason` **또는** `cap_by` 가 안전 사유이고 `objs` 에 근거 객체가 있는 프레임이 구간의 과반일 때다.
  - ③⑥ 은 **구간마다 몇 번째 위반인지**에 따라 경미/중대가 갈린다.
  - ⑮ 는 `Sheet` 를 쓰지 않고 따로 센다.
  - `classify` 는 행을 제자리에서 바꾼다. 채점기와 심판에 같은 행을 넣을 때는 각각 복사본을 준다.
  - `RunLogger` 는 `t` 를 소수 둘째 자리로 적는다. 행의 `t` 는 그 반올림값이다.

## 코스 데이터(2026-09-15 측정, 대본 판 위치의 근거)

- 코스 H 0~250 m: 제한 50 km/h. 경로가 차로 중앙이고 왼쪽 중앙선까지 `l≈1.5 m`, 오른쪽 `r≈4.3~4.5 m`. 교차로(`j=1`)는 106~135 m. 신호 151 의 정지선 DB 점은 경로 s=96.3 m(횡 +1.51). 신호 횡단보도 중심은 s=104.5(횡 +3.14)와 s=130.7(횡 +2.99).
- 코스 G 640~780 m: 제한 50 km/h 에서 683 m 부터 보호구역 30 km/h. 교차로는 788 m 에서 시작한다.
- 코스 G 2575~2841 m: 교차로 없는 266 m, 제한 50 km/h. 왼쪽에 같은 방향 차로가 있다(`l≈4.4~4.6`, `r≈1.5~1.9`).

## 파일 구조

| 파일 | 책임 |
|---|---|
| `vtd_rl/rule_stack.py` (수정) | `score_fma`·`check_lanes`·`control` 재노출, 지도 로더 `load_map()` |
| `vtd_rl/world/board.py` (수정) | `Board.ego_lanes`(계획기 차로변경 표식) 보존과 자르기, 자를 때 신호 운용 지정 |
| `vtd_rl/world/signals.py` (수정) | 시험용 신호 운용 `always_red`·`always_flash` |
| `vtd_rl/referee/__init__.py` | 패키지 |
| `vtd_rl/referee/rows.py` | `RowRecorder`: (t, State, Command) → RunLogger CSV 한 줄 → `score_fma.load` 와 같은 행 사전 |
| `vtd_rl/referee/oracle.py` | `score_episode`: `score_fma.main()` 과 같은 조립으로 사후 채점, 감점 호출 기록 |
| `vtd_rl/drivers/__init__.py` · `vtd_rl/drivers/scripted.py` | 대본 운전자(속도·횡오프셋·지시등 일정)와 선생님 운전자 |
| `vtd_rl/rollout.py` (수정) | 운전자 종류와 무관한 `run_episode` + 행 기록 |
| `vtd_rl/referee/scenarios.py` | 일치 검증용 대본 판 목록 |
| `vtd_rl/referee/sections.py` | `FastSections`: `score_fma.Sections` 와 같은 답을 격자 색인으로 빠르게 |
| `vtd_rl/referee/core.py` | `Hit`, `SpanTracker`, `Context`, `Referee` |
| `vtd_rl/referee/parity.py` | `compare`: 같은 행을 채점기와 심판에 넣고 비교 |
| `vtd_rl/referee/judges/speed.py` | ① ② |
| `vtd_rl/referee/judges/traffic_light.py` | ⑦ ⑧ ⑨ |
| `vtd_rl/referee/judges/contact.py` | ⑪ ⑭ ⑫ |
| `vtd_rl/referee/judges/pedestrian.py` | ⑩ |
| `vtd_rl/referee/judges/lane_geometry.py` | ③ ④ ⑤ ⑥ |
| `vtd_rl/referee/judges/turn_signal.py` | ⑬ ⑮ |
| `tests/referee/…` | 단위 테스트, 일치 테스트 |
| `scripts/referee_parity_report.py` · `docs/reports/m2a-referee-parity.md` | 완료 증거 |

---

### Task 1: 규칙 스택 재노출, 판 차로변경 표식, 시험용 신호, 행 기록기

**Files:**
- Modify: `vtd_rl/rule_stack.py`, `vtd_rl/world/board.py`, `vtd_rl/world/signals.py`
- Create: `vtd_rl/referee/__init__.py`, `vtd_rl/referee/rows.py`
- Test: `tests/test_rule_stack.py`(추가), `tests/test_board.py`(추가), `tests/test_signals.py`(추가), `tests/referee/__init__.py`, `tests/referee/test_rows.py`

**Interfaces:**
- Produces:
  - `rs.score_fma`, `rs.check_lanes`, `rs.control` (모듈), `rs.XODR: str`, `rs.load_map() -> build_lane_plan.Map` (`lru_cache`, 같은 객체를 돌려준다)
  - `Board(name, scenario, lane_plan, signals="always_green", ego_lanes=[])` — `ego_lanes` 는 경로 JSON 의 `ego_lanes`(없으면 `[]`), 비어 있지 않은데 길이가 경로점 수와 다르면 `ValueError`
  - `slice_board(board, s_from, s_to, name, signals=None) -> Board` — `signals=None` 이면 원판의 운용을 따른다. `ego_lanes` 도 같은 범위로 자른다
  - `vtd_rl.world.signals.SIGNAL_MODES = ("always_green", "cycle", "always_red", "always_flash")`, `vtd_rl.world.signals.TL_FLASH = 6`. `board.py` 는 `SIGNAL_MODES` 를 여기서 import 한다
  - `RowRecorder(csv_path: str | None = None)`: `.record(t, state, cmd) -> dict`, `.close()`. 행 사전의 키와 형은 `score_fma.load` 와 같다: `t,x,y,h,v,tl,tid,reason,sig,clr,d_ego,objs,cap_by`
  - `parse_row(line: str) -> dict`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_rule_stack.py` 끝에 추가:
```python
def test_채점기_모듈과_지도():
    assert rs.score_fma.MINOR == 3 and rs.score_fma.MAJOR == 6
    assert callable(rs.check_lanes.classify) and callable(rs.check_lanes.lane_change_events)
    assert callable(rs.control.PurePursuit) and callable(rs.control.LongPI)
    mp = rs.load_map()
    assert rs.load_map() is mp
    assert mp.roads
```

`tests/test_board.py` 끝에 추가(파일의 기존 `H` 를 쓴다):
```python
def test_차로변경_표식_보존과_자르기():
    b = load_board(H)
    assert len(b.ego_lanes) == len(b.route.pts)
    s = slice_board(b, 0.0, 250.0, "H_0_250")
    assert len(s.ego_lanes) == len(s.route.pts)
    assert s.ego_lanes == b.ego_lanes[:len(s.ego_lanes)]
    assert s.signals == b.signals
    red = slice_board(b, 0.0, 250.0, "H_red", signals="always_red")
    assert red.signals == "always_red"
    with pytest.raises(ValueError):
        slice_board(b, 0.0, 250.0, "H_bad", signals="purple")
```

`tests/test_signals.py` 끝에 추가:
```python
def test_시험용_신호_운용():
    sig = [RouteSignal(10.0, 3)]
    red = programs_for("always_red", sig, random.Random(0))
    flash = programs_for("always_flash", sig, random.Random(0))
    assert red[3].state(123.0) == rs.TL_RED
    assert flash[3].state(0.0) == 6
```

`tests/referee/__init__.py` — 빈 파일.

`tests/referee/test_rows.py`
```python
import math

from vtd_rl import rule_stack as rs
from vtd_rl.referee.rows import RowRecorder


def state_and_cmd():
    s = rs.State(x=10.0, y=-2.0, heading=0.3, speed=4.25, tl_id=151, tl_state=rs.TL_GREEN,
                 objects=[rs.Obj(7, 20.0, -1.0, 0.0, 0.1, 1.3, 0.6, 0.7, 1.8)])
    c = rs.Command(steer=0.01, accel=0.5, turn=rs.TS_LEFT, reason="SCRIPT", cap_by="SCRIPT",
                   d_ego=0.42)
    return s, c


def test_행_사전은_채점기_load_와_같다(tmp_path):
    s, c = state_and_cmd()
    path = tmp_path / "run.csv"
    rec = RowRecorder(str(path))
    rows = [rec.record(1.25, s, c), rec.record(1.30, s, c)]
    rec.close()
    assert rs.score_fma.load(str(path)) == rows
    row = rows[0]
    assert row["sig"] == rs.TS_LEFT and row["tid"] == 151 and math.isclose(row["d_ego"], 0.42)
    assert row["objs"].count(":") >= 9 and row["clr"] < 99.9


def test_파일_없이도_행을_만든다():
    s, c = state_and_cmd()
    row = RowRecorder().record(0.0, s, c)
    assert set(row) == {"t", "x", "y", "h", "v", "tl", "tid", "reason", "sig", "clr",
                        "d_ego", "objs", "cap_by"}
```

- [ ] **Step 2: 실패 확인**

Run: `env -u PYTHONPATH .venv/bin/pytest tests/test_rule_stack.py tests/test_board.py tests/test_signals.py tests/referee/test_rows.py -q`
Expected: FAIL — `AttributeError: module 'vtd_rl.rule_stack' has no attribute 'score_fma'` 등

- [ ] **Step 3: 구현**

`vtd_rl/rule_stack.py` — 기존 import 줄들 아래에 추가하고, `__all__` 에 `"score_fma", "check_lanes", "control", "XODR", "load_map"` 를 더한다:
```python
import score_fma  # noqa: E402  (eval/ — check_lanes·run_logger 를 최상위 이름으로 import 한다)
import check_lanes  # noqa: E402
import control  # noqa: E402

XODR = os.path.join(ROOT, "map", "HL_FMA_VTD_LivingLab.xodr")


@functools.lru_cache(maxsize=1)
def load_map():
    """LivingLab 지도. 실측 0.34 초(2026-09-15)라 캐시 파일 없이 프로세스당 한 번 읽는다."""
    from build_lane_plan import Map
    return Map(XODR)
```

`vtd_rl/world/signals.py`:
- `NO_SIGNAL` 줄 아래에 추가:
```python
TL_FLASH = 6             # 적색점멸 — rule_stack 은 이 이름을 재노출하지 않는다(vtd_io.TL_FLASH)
SIGNAL_MODES = ("always_green", "cycle", "always_red", "always_flash")   # always_red·always_flash 는 시험용
_FIXED = {"always_green": rs.TL_GREEN, "always_red": rs.TL_RED, "always_flash": TL_FLASH}
```
- `SignalProgram.state` 의 첫 분기(`if self.mode == "always_green": return rs.TL_GREEN`)를 바꾼다:
```python
        if self.mode in _FIXED:
            return _FIXED[self.mode]
```
- `programs_for` 의 첫 분기(`if mode == "always_green": ...`)를 바꾼다:
```python
    if mode in _FIXED:
        return {s.tl_id: SignalProgram(mode) for s in signals}
```

`vtd_rl/world/board.py`:
- `SIGNAL_MODES = ("always_green", "cycle")` 줄을 `from vtd_rl.world.signals import SIGNAL_MODES` 로 바꾼다.
- `Board` 에 필드를 넣고 검사를 더한다:
```python
@dataclass
class Board:
    name: str
    scenario: object
    lane_plan: list
    signals: str = "always_green"
    ego_lanes: list = field(default_factory=list)   # 경로 JSON ego_lanes — [i][4] 가 계획기 차로변경 표식
    route: RouteIndex = field(init=False)

    def __post_init__(self):
        ...(기존 신호·길이 검사 그대로)
        if self.ego_lanes and len(self.ego_lanes) != len(pts):
            raise ValueError(f"{self.name}: 경로점 {len(pts)} != ego_lanes {len(self.ego_lanes)}")
        self.route = RouteIndex(pts)
```
- `load_board`:
```python
def load_board(entry: dict, signals: str = "always_green") -> Board:
    sc = rs.Scenario.load(rs.path(entry["route"]))
    lane = _load_json(entry["lane"])["pts"]
    ego_lanes = _load_json(entry["route"]).get("ego_lanes") or []
    return Board(entry["name"], sc, lane, signals, ego_lanes)
```
- `slice_board` 서명을 `def slice_board(board: Board, s_from: float, s_to: float, name: str, signals: str | None = None) -> Board:` 로 바꾸고 반환을 바꾼다:
```python
    return Board(name, sc, board.lane_plan[i0:i1 + 1], signals or board.signals,
                 board.ego_lanes[i0:i1 + 1])
```

`vtd_rl/referee/__init__.py`
```python
"""온라인 심판 — 대회 채점기(score_fma)와 같은 판정을 한 프레임씩."""
```

`vtd_rl/referee/rows.py`
```python
"""주행 한 프레임 -> 채점기 행.

규칙 스택 RunLogger 가 쓰는 CSV 한 줄을 그대로 만들고, score_fma.load 와 같은 변환으로 사전을 만든다.
심판과 사후 채점기가 글자 하나 다르지 않은 입력을 받게 하려는 것이다.
"""
import csv
import io

from vtd_rl import rule_stack as rs

_FIELDS = rs.HEADER.strip().split(",")


def parse_row(line: str) -> dict:
    r = next(csv.DictReader(io.StringIO(line), fieldnames=_FIELDS))

    def g(k, d=None):                       # score_fma.load 의 g 와 같은 규칙
        v = r.get(k)
        return d if v in (None, "") else v

    return dict(
        t=float(r["t"]), x=float(r["x"]), y=float(r["y"]),
        h=float(r["heading"]), v=float(r["v"]),
        tl=int(r["tl_state"]), tid=int(r["tl_id"]),
        reason=r.get("reason", ""),
        sig=int(g("sig", 0)),
        clr=float(g("clr", 99.9)),
        d_ego=float(g("d_ego", 0.0)),
        objs=r.get("objs", ""),
        cap_by=r.get("cap_by", ""))


class RowRecorder:
    def __init__(self, csv_path=None):
        self._buf = io.StringIO()
        self._logger = rs.RunLogger(None)
        self._logger.f = self._buf           # RunLogger.log 는 self.f 에 한 줄 쓰고 flush 한다
        self._file = open(csv_path, "w", encoding="utf-8") if csv_path else None
        if self._file:
            self._file.write(rs.HEADER)

    def record(self, t, state, cmd) -> dict:
        self._buf.seek(0)
        self._buf.truncate()
        self._logger.log(t, state, cmd)
        line = self._buf.getvalue()
        if self._file:
            self._file.write(line)
        return parse_row(line)

    def close(self):
        if self._file:
            self._file.close()
            self._file = None
```

- [ ] **Step 4: 통과 확인**

Run: `env -u PYTHONPATH .venv/bin/pytest -q; echo rc=$?`
Expected: `rc=0`

- [ ] **Step 5: 커밋**

```bash
git add vtd_rl/rule_stack.py vtd_rl/world/board.py vtd_rl/world/signals.py vtd_rl/referee tests
git commit -m "심판 준비 — 채점기 모듈 재노출·지도 로더, 판 차로변경 표식, 시험용 적색·점멸 신호, 행 기록기"
```

---

### Task 2: 사후 채점기(oracle) — `score_fma.main()` 과 같은 조립

**Files:**
- Create: `vtd_rl/referee/oracle.py`
- Test: `tests/referee/test_oracle.py`

**Interfaces:**
- Consumes: `rs.score_fma`, `rs.check_lanes`, `rs.load_map`, `rs.path`, `Board.route.pts`, `Board.lane_plan`, `Board.ego_lanes`
- Produces:
  - `MatchInputs(route: list, lims: list, lc_flag: list, tl_stops: dict, cws: list)` 와 `match_inputs(board) -> MatchInputs`
  - `make_lim_at(secs, lims)`, `make_lc_at(secs, lc_flag)` — `score_fma.main()` 안의 `lim_at`·`lc_at` 과 같은 함수를 만든다
  - `OracleResult(hits: list[tuple[int, int, str]], state: list[dict], respawns: dict[int, list[float]], notes: list[str])` — `hits` 는 `sheet.hit` 호출 순서의 `(sec, item, level)`
  - `score_episode(rows, board, sections: int = 5, use_map: bool = True) -> OracleResult` — `rows`(목록 또는 튜플)의 깊은 복사본을 쓴다

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/referee/test_oracle.py`
```python
import copy

from vtd_rl import rule_stack as rs
from vtd_rl.referee.oracle import score_episode
from vtd_rl.world.board import load_board, slice_board

H = {"name": "course_H", "route": "routes/HL_FMA_NEW_H.json", "lane": "routes/HL_FMA_NEW_H_lane.json"}


def h_slice():
    return slice_board(load_board(H), 0.0, 250.0, "H_0_250")


def rows_along(board, v, n, dt=0.05):
    """경로를 따라 속도 v 로 가는 합성 행."""
    out, s = [], 0.0
    for k in range(n):
        x, y, h = board.route.point_at(s)
        out.append(dict(t=round(k * dt, 2), x=x, y=y, h=h, v=v, tl=0, tid=0, reason="", sig=0,
                        clr=99.9, d_ego=0.0, objs="", cap_by=""))
        s += v * dt
    return out


def test_과속은_항목1_중대():
    b = h_slice()
    rows = rows_along(b, v=80 / 3.6, n=200)
    before = copy.deepcopy(rows)
    res = score_episode(rows, b, sections=5, use_map=False)
    assert rows == before
    assert {i for _, i, _ in res.hits} == {1}
    assert {lv for _, _, lv in res.hits} == {"major"}
    assert res.state[0] == {1: "major"}
    assert res.respawns == {}


def test_규정속도는_감점_없음():
    b = h_slice()
    res = score_episode(rows_along(b, v=40 / 3.6, n=300), b, sections=5, use_map=False)
    assert res.hits == []


def test_지도는_복사한_행에_붙이고_없으면_3456_을_건너뛴다(monkeypatch):
    b = h_slice()
    rows = rows_along(b, v=40 / 3.6, n=50)
    seen, called = [], []
    monkeypatch.setattr(rs.check_lanes, "classify", lambda rr, mp: seen.append((rr, mp)))
    monkeypatch.setattr(rs.score_fma, "item_lane_geometry", lambda *a: called.append(a))
    score_episode(rows, b, use_map=True)
    assert len(seen) == 1 and seen[0][0] is not rows and seen[0][0] == rows
    assert seen[0][1] is rs.load_map()
    assert len(called) == 1
    seen.clear()
    called.clear()
    score_episode(rows, b, use_map=False)
    assert seen == [] and called == []


def test_리스폰은_따로_센다():
    b = h_slice()
    rows = rows_along(b, v=10.0, n=100)
    for r in rows[50:]:
        r["x"] += 20.0
    res = score_episode(rows, b, sections=5, use_map=False)
    assert sum(len(v) for v in res.respawns.values()) == 1
    assert all(i != 15 for _, i, _ in res.hits)
```

- [ ] **Step 2: 실패 확인**

Run: `env -u PYTHONPATH .venv/bin/pytest tests/referee/test_oracle.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'vtd_rl.referee.oracle'`

- [ ] **Step 3: 구현**

`vtd_rl/referee/oracle.py`
```python
"""사후 채점기 — score_fma.main() 과 같은 조립으로 한 판을 채점하고 감점 호출을 기록한다.

심판(온라인)이 맞는지 가르는 정답이다. main() 과 같은 데이터를 쓴다:
경로 = 판의 ego_route, 제한속도 = 차로계획 lim, 차로변경 표식 = ego_lanes[i][4],
정지선 = routes/tl_map_livinglab.json(문자열 키 그대로), 횡단보도 = routes/crosswalks.json 중 경로 6 m 안.
"""
import copy
import functools
import json
import math
from dataclasses import dataclass

from vtd_rl import rule_stack as rs

sf = rs.score_fma


class _RecordingSheet(sf.Sheet):
    def __init__(self, n):
        super().__init__(n)
        self.calls = []

    def hit(self, sec, item, level, why):
        self.calls.append((sec, item, level))
        super().hit(sec, item, level, why)


@dataclass
class MatchInputs:
    route: list
    lims: list
    lc_flag: list
    tl_stops: dict
    cws: list


@dataclass
class OracleResult:
    hits: list
    state: list
    respawns: dict
    notes: list


@functools.lru_cache(maxsize=None)
def _json(rel):
    with open(rs.path(rel), encoding="utf-8") as f:
        return json.load(f)


def match_inputs(board) -> MatchInputs:
    route = [list(p) for p in board.route.pts]
    lims = [p.get("lim") for p in board.lane_plan]
    lc_flag = [bool(p[4]) if len(p) > 4 else False for p in board.ego_lanes]
    cws = [c for c in _json("routes/crosswalks.json")["crosswalks"]
           if min((math.hypot(p[0] - c["x"], p[1] - c["y"]) for p in route), default=99) < 6.0]
    return MatchInputs(route, lims, lc_flag, _json("routes/tl_map_livinglab.json"), cws)


def make_lim_at(secs, lims):
    def lim_at(x, y):
        if not lims:
            return None
        i = secs.index_of(x, y)
        return lims[i] if i < len(lims) else None
    return lim_at


def make_lc_at(secs, lc_flag):
    def lc_at(x, y):
        if not lc_flag:
            return False
        i = secs.index_of(x, y)
        lo = max(0, i - int(sf.LC_MARGIN_M))
        hi = min(len(lc_flag), i + int(sf.LC_MARGIN_M) + 1)
        return any(lc_flag[lo:hi])
    return lc_at


def score_episode(rows, board, sections=5, use_map=True) -> OracleResult:
    rows = copy.deepcopy(list(rows))
    m = match_inputs(board)
    secs = sf.Sections(m.route, sections)
    sheet = _RecordingSheet(secs.n)
    if use_map:
        rs.check_lanes.classify(rows, rs.load_map())
    notes = []
    sf.item_speed(rows, sheet, secs, make_lim_at(secs, m.lims))
    if use_map:
        sf.item_lane_geometry(rows, sheet, secs, make_lc_at(secs, m.lc_flag))
    sf.item_traffic_light(rows, sheet, secs, m.tl_stops, notes)
    sf.item_contact(rows, sheet, secs)
    sf.item_pedestrian(rows, sheet, secs)
    sf.item_crosswalk_stop(rows, sheet, secs, m.cws)
    sf.item_turn_signal(rows, sheet, secs)
    respawns = sf.item_respawn(rows, sheet, secs)
    return OracleResult(sheet.calls, [dict(s) for s in sheet.state], respawns, notes)
```

- [ ] **Step 4: 통과 확인**

Run: `env -u PYTHONPATH .venv/bin/pytest tests/referee/test_oracle.py -q; echo rc=$?`
Expected: `rc=0`

- [ ] **Step 5: 커밋**

```bash
git add vtd_rl/referee/oracle.py tests/referee/test_oracle.py
git commit -m "사후 채점기 — score_fma.main 과 같은 조립, 감점 호출 기록"
```

---

### Task 3: 대본 운전자와 운전자 무관 한 판 달리기

**Files:**
- Create: `vtd_rl/drivers/__init__.py`, `vtd_rl/drivers/scripted.py`
- Modify: `vtd_rl/rollout.py`
- Test: `tests/test_scripted.py`

**Interfaces:**
- Consumes: `World`, `StepInfo`(`t, s, lateral, index, v`), `World.cfg.dt`, `World.clock`, `Board.route`(`pts, cum, heading_at`), `ShadowTeacher(board).act(state, now)`, `RowRecorder`, `rs.control.PurePursuit().steer(x, y, heading, speed, path)`, `rs.control.LongPI().accel(v_target, v_cur, dt)`, `rs.Command`
- Produces:
  - 일정 — 모두 `__call__(s: float, t: float)` 를 가진다:
    - `Cruise(v)` → v
    - `StopAt(s_stop, hold, v, decel=2.0)` → s_stop 앞에서는 `min(v, sqrt(2·decel·남은거리))`. 남은 거리가 0.3 m 이하가 되면 hold 초 동안 0, 그 뒤로 v
    - `Offset(value=0.0)` → value
    - `LaneShift(s_start, delta, length)` → s_start 부터 length m 동안 0 에서 delta 로 선형 변화, 그 뒤 delta
    - `SignalWindow(s_on, s_off, side)` → `s_on ≤ s < s_off` 이면 side, 아니면 0
  - `ScriptedDriver(board, speed, offset, signal)` + `.act(state, info: StepInfo | None, dt) -> Command`. `state.speed` 를 참속도로 채우고 `Command.d_ego` 를 경로 횡위치로 채운다. `reason`·`cap_by` 는 `"SCRIPT"`
  - `TeacherDriver(board)` + `.bind(world)` + `.act(state, info, dt)`
  - `EpisodeRun(result: EpisodeResult, rows: list[dict])`, `run_episode(board, driver, config=None, log_csv=None, seed=0) -> EpisodeRun`. 운전자에 `bind` 가 있으면 월드를 만든 뒤 부른다
  - `run_teacher_episode(board, config=None, log_csv=None, seed=0) -> EpisodeResult` — 서명·반환은 그대로, 내부는 `run_episode` 를 쓴다

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_scripted.py`
```python
import math

import pytest

from vtd_rl.drivers.scripted import Cruise, LaneShift, Offset, ScriptedDriver, SignalWindow, StopAt
from vtd_rl.rollout import run_episode, run_teacher_episode
from vtd_rl.world.board import load_board, slice_board

H = {"name": "course_H", "route": "routes/HL_FMA_NEW_H.json", "lane": "routes/HL_FMA_NEW_H_lane.json"}


def h_slice():
    return slice_board(load_board(H), 0.0, 250.0, "H_0_250")


def test_일정():
    assert Cruise(5.0)(10.0, 0.0) == 5.0
    st = StopAt(50.0, hold=2.0, v=8.0)
    assert st(0.0, 0.0) == 8.0
    assert st(45.0, 1.0) == pytest.approx(math.sqrt(2 * 2.0 * 5.0))
    assert st(49.8, 3.0) == 0.0
    assert st(49.9, 4.9) == 0.0
    assert st(49.9, 5.1) == 8.0
    assert st(60.0, 6.0) == 8.0
    ls = LaneShift(100.0, delta=3.2, length=40.0)
    assert ls(90.0, 0) == 0.0 and ls(120.0, 0) == pytest.approx(1.6) and ls(200.0, 0) == 3.2
    sw = SignalWindow(10.0, 20.0, side=1)
    assert (sw(5.0, 0), sw(15.0, 0), sw(25.0, 0)) == (0, 1, 0)
    assert Offset(1.5)(0.0, 0.0) == 1.5


def test_대본_운전자는_경로를_따라_완주():
    b = h_slice()
    run = run_episode(b, ScriptedDriver(b, Cruise(8.0), Offset(0.0), SignalWindow(0, 0, 0)))
    assert run.result.outcome == "goal"
    assert len(run.rows) == run.result.steps
    assert max(abs(r["d_ego"]) for r in run.rows[100:]) < 1.5
    assert {r["reason"] for r in run.rows} == {"SCRIPT"}
    assert max(r["v"] for r in run.rows) == pytest.approx(8.0, abs=1.0)


def test_정지_일정을_따른다():
    b = h_slice()
    run = run_episode(b, ScriptedDriver(b, StopAt(60.0, 4.0, 8.0), Offset(0.0), SignalWindow(0, 0, 0)))
    stopped = [r for r in run.rows if r["t"] > 2.0 and r["v"] <= 1 / 3.6]
    assert stopped and stopped[-1]["t"] - stopped[0]["t"] > 2.5
    assert run.result.outcome == "goal"


def test_횡오프셋과_지시등을_따른다():
    b = h_slice()
    run = run_episode(b, ScriptedDriver(b, Cruise(6.0), LaneShift(150.0, 1.2, 20.0),
                                        SignalWindow(140.0, 180.0, 1)))
    late = [r["d_ego"] for r in run.rows if r["t"] > 32.0]
    assert late and min(late) > 0.8
    assert any(r["sig"] == 1 for r in run.rows) and run.rows[-1]["sig"] == 0


def test_선생님_래퍼는_그대로():
    assert run_teacher_episode(h_slice()).outcome == "goal"
```

- [ ] **Step 2: 실패 확인**

Run: `env -u PYTHONPATH .venv/bin/pytest tests/test_scripted.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'vtd_rl.drivers'`

- [ ] **Step 3: 운전자 구현**

`vtd_rl/drivers/__init__.py`
```python
"""운전자 — 대본(시험용)·선생님."""
```

`vtd_rl/drivers/scripted.py`
```python
"""대본 운전자 — 속도·횡오프셋·지시등을 경로 거리 s 와 시각 t 의 일정으로 준다.

일부러 규칙을 어기는 판을 만들어 심판과 채점기가 같은 판정을 내는지 보는 시험 도구다.
조향은 규칙 스택 PurePursuit, 속도는 LongPI 를 그대로 쓴다.
"""
import math

from vtd_rl import rule_stack as rs

AHEAD_PTS = 40          # 조향 경로로 넘길 앞쪽 경로점 수(점 간격 약 1.4 m)
STOP_REACHED = 0.3      # StopAt: 남은 거리가 이 안이면 도착으로 친다[m]


class Cruise:
    def __init__(self, v):
        self.v = v

    def __call__(self, s, t):
        return self.v


class StopAt:
    def __init__(self, s_stop, hold, v, decel=2.0):
        self.s_stop, self.hold, self.v, self.decel = s_stop, hold, v, decel
        self._since = None
        self._done = False

    def __call__(self, s, t):
        if self._done:
            return self.v
        if self._since is None:
            remain = self.s_stop - s
            if remain > STOP_REACHED:
                return min(self.v, math.sqrt(2.0 * self.decel * remain))
            self._since = t
        if t - self._since < self.hold:
            return 0.0
        self._done = True
        return self.v


class Offset:
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self, s, t):
        return self.value


class LaneShift:
    def __init__(self, s_start, delta, length):
        self.s_start, self.delta, self.length = s_start, delta, length

    def __call__(self, s, t):
        return self.delta * max(0.0, min(1.0, (s - self.s_start) / self.length))


class SignalWindow:
    def __init__(self, s_on, s_off, side):
        self.s_on, self.s_off, self.side = s_on, s_off, side

    def __call__(self, s, t):
        return self.side if self.s_on <= s < self.s_off else 0


class ScriptedDriver:
    def __init__(self, board, speed, offset, signal):
        self.board, self.speed, self.offset, self.signal = board, speed, offset, signal
        self.pp = rs.control.PurePursuit()
        self.pi = rs.control.LongPI()

    def act(self, state, info, dt):
        route = self.board.route
        if info is None:
            s, lat, v, t, idx = 0.0, 0.0, 0.0, 0.0, 0
        else:
            s, lat, v, t, idx = info.s, info.lateral, info.v, info.t, info.index
        path = []
        for i in range(idx, min(len(route.pts), idx + AHEAD_PTS)):
            h = route.heading_at(i)
            off = self.offset(route.cum[i], t)
            x, y = route.pts[i]
            path.append((x - off * math.sin(h), y + off * math.cos(h)))
        state.speed = v
        return rs.Command(steer=self.pp.steer(state.x, state.y, state.heading, v, path),
                          accel=self.pi.accel(self.speed(s, t), v, dt),
                          turn=int(self.signal(s, t)), reason="SCRIPT", cap_by="SCRIPT",
                          lane_offset=self.offset(s, t), d_ego=lat)


class TeacherDriver:
    """ShadowTeacher 를 운전자 인터페이스로 감싼다. now 는 world.clock 이다."""

    def __init__(self, board):
        from vtd_rl.teacher.shadow import ShadowTeacher
        self.teacher = ShadowTeacher(board)
        self.world = None

    def bind(self, world):
        self.world = world

    def act(self, state, info, dt):
        return self.teacher.act(state, self.world.clock)
```

- [ ] **Step 4: 한 판 달리기 일반화**

`vtd_rl/rollout.py` 전체를 다음으로 바꾼다:
```python
"""운전자로 한 판 끝까지 달리기 — 행 기록 포함."""
import time
from dataclasses import dataclass

from vtd_rl.referee.rows import RowRecorder
from vtd_rl.world.world import World


@dataclass
class EpisodeResult:
    board: str
    outcome: str
    sim_time: float
    steps: int
    wall_time: float
    distance: float

    @property
    def steps_per_sec(self) -> float:
        return self.steps / max(self.wall_time, 1e-9)


@dataclass
class EpisodeRun:
    result: EpisodeResult
    rows: list


def run_episode(board, driver, config=None, log_csv=None, seed=0) -> EpisodeRun:
    world = World(board, config, seed=seed)
    state = world.reset()
    if hasattr(driver, "bind"):
        driver.bind(world)
    recorder = RowRecorder(log_csv)
    rows, info = [], None
    t0 = time.perf_counter()
    try:
        while info is None or not info.done:
            cmd = driver.act(state, info, world.cfg.dt)
            rows.append(recorder.record(world.t, state, cmd))
            state, info = world.step(cmd.steer, cmd.accel, cmd.turn)
    finally:
        recorder.close()
    result = EpisodeResult(board.name, info.outcome, info.t, len(rows), time.perf_counter() - t0, info.s)
    return EpisodeRun(result, rows)


def run_teacher_episode(board, config=None, log_csv=None, seed=0) -> EpisodeResult:
    from vtd_rl.drivers.scripted import TeacherDriver
    return run_episode(board, TeacherDriver(board), config, log_csv, seed).result
```

- [ ] **Step 5: 통과 확인**

Run: `env -u PYTHONPATH .venv/bin/pytest -q; echo rc=$?`
Expected: `rc=0` (M1 선생님 테스트 포함)

- [ ] **Step 6: 커밋**

```bash
git add vtd_rl/drivers vtd_rl/rollout.py tests/test_scripted.py
git commit -m "대본 운전자(속도·오프셋·지시등 일정)·운전자 무관 한 판 달리기와 행 기록"
```

---

### Task 4: 일치 검증용 대본 판 목록

**Files:**
- Create: `vtd_rl/referee/scenarios.py`
- Test: `tests/referee/test_scenarios.py`

**Interfaces:**
- Consumes: `load_board`, `slice_board(..., signals=)`, `ScriptedDriver`, 일정 클래스, `TeacherDriver`, `run_episode`, `score_episode`, `rs.Actor(id, type, size=[length, width, height], spawn, motion)`, `rs.score_fma.FRONT`
- Produces:
  - `Scenario(name: str, board: Board, make_driver: Callable[[Board], driver], expect: frozenset[int], forbid: frozenset[int], note: str)`
  - `SCENARIOS: dict[str, Callable[[], Scenario]]` — 이름 15개
  - `episode_rows(name) -> tuple[Scenario, tuple[dict, ...]]` — 프로세스 안에서 캐시한다. 쓰는 쪽은 행 사전을 바꾸지 않는다(채점기와 심판 비교는 복사본을 쓴다)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/referee/test_scenarios.py`
```python
import pytest

from vtd_rl.referee.oracle import score_episode
from vtd_rl.referee.scenarios import SCENARIOS, episode_rows


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_대본_판은_채점기에서_기대한_항목을_일으킨다(name):
    sc, rows = episode_rows(name)
    fired = {i for _, i, _ in score_episode(rows, sc.board, sections=5, use_map=True).hits}
    assert sc.expect <= fired, f"{name}: 기대 {sorted(sc.expect)} / 채점기 {sorted(fired)}"
    assert not (sc.forbid & fired), f"{name}: 금지 {sorted(sc.forbid)} / 채점기 {sorted(fired)}"
```

- [ ] **Step 2: 실패 확인**

Run: `env -u PYTHONPATH .venv/bin/pytest tests/referee/test_scenarios.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'vtd_rl.referee.scenarios'`

- [ ] **Step 3: 구현**

`vtd_rl/referee/scenarios.py`
```python
"""일치 검증용 대본 판. 각 판은 채점기에서 expect 항목을 일으키고 forbid 항목은 일으키지 않아야 한다.

위치 근거(2026-09-15 측정):
- 코스 H 0~250 m: 제한 50, 중앙선까지 l≈1.5, 교차로 106~135 m, 신호151 정지선 DB 점 s=96.3,
  신호 횡단보도 중심 s=104.5(횡 +3.14)·s=130.7(횡 +2.99).
- 코스 G 640~780 m: 683 m 부터 보호구역 30.
- 코스 G 2575~2841 m: 교차로 없는 구간, 왼쪽에 같은 방향 차로(l≈4.5).
대본 수치는 기대 항목이 일어나도록 조정해도 된다. 채점기·심판·월드 문턱은 바꾸지 않는다.
"""
import functools
import math
from dataclasses import dataclass

from vtd_rl import rule_stack as rs
from vtd_rl.drivers.scripted import (Cruise, LaneShift, Offset, ScriptedDriver, SignalWindow,
                                     StopAt, TeacherDriver)
from vtd_rl.rollout import run_episode
from vtd_rl.world.board import load_board, slice_board

KMH = 1 / 3.6
FRONT = rs.score_fma.FRONT
H_TL151 = 96.3          # 신호151 정지선 DB 점의 경로 s
H_CW = 130.7            # 신호 횡단보도 중심의 경로 s(횡 +2.99)


@dataclass
class Scenario:
    name: str
    board: object
    make_driver: object
    expect: frozenset
    forbid: frozenset
    note: str


@functools.lru_cache(maxsize=None)
def _course(letter):
    return load_board({"name": f"course_{letter}", "route": f"routes/HL_FMA_NEW_{letter}.json",
                       "lane": f"routes/HL_FMA_NEW_{letter}_lane.json"})


def _slice(letter, s0, s1, signals="always_green"):
    return slice_board(_course(letter), s0, s1, f"{letter}_{s0}_{s1}", signals=signals)


def _script(speed, offset=None, signal=None):
    return lambda b: ScriptedDriver(b, speed, offset or Offset(0.0), signal or SignalWindow(0, 0, 0))


def _route_point(board, s, lateral):
    x, y, h = board.route.point_at(s)
    return x - lateral * math.sin(h), y + lateral * math.cos(h), h


def _static(board, aid, kind, s, size):
    x, y, _ = _route_point(board, s, 0.0)
    return rs.Actor(id=aid, type=kind, size=list(size), spawn={"at_time": 0.0},
                    motion={"kind": "static", "pos": [x, y]})


def _scenario(name, board, make_driver, expect, note, forbid=()):
    return Scenario(name, board, make_driver, frozenset(expect), frozenset(forbid), note)


def speeder():
    return _scenario("speeder", _slice("G", 2575, 2841), _script(Cruise(80 * KMH)), {1},
                     "제한 50 에서 80 km/h")


def zone_speeder():
    return _scenario("zone_speeder", _slice("G", 640, 780), _script(Cruise(45 * KMH)), {2},
                     "보호구역 30 에서 45 km/h")


def red_runner():
    return _scenario("red_runner", _slice("H", 0, 250, "always_red"), _script(Cruise(40 * KMH)), {7},
                     "적색 무정차 통과(중대)")


def red_far_stopper():
    return _scenario("red_far_stopper", _slice("H", 0, 250, "always_red"),
                     _script(StopAt(H_TL151 - FRONT - 5.0, 3.0, 30 * KMH)), {7},
                     "앞범퍼가 정지선 5 m 앞에 섰다가 통과(경미)")


def flash_runner():
    return _scenario("flash_runner", _slice("H", 0, 250, "always_flash"), _script(Cruise(40 * KMH)), {9},
                     "적색점멸 무정차 통과")


def green_idler():
    return _scenario("green_idler", _slice("H", 0, 250), _script(StopAt(H_TL151 - 18.0, 12.0, 30 * KMH)),
                     {8}, "녹색에 정지선 18 m 앞 12 초 정차")


def crosswalk_stopper():
    return _scenario("crosswalk_stopper", _slice("H", 0, 250),
                     _script(StopAt(H_CW - FRONT / 2, 5.0, 20 * KMH), LaneShift(110.0, 1.0, 10.0)),
                     {12}, "횡단보도 위 5 초 정지(횡 +1 m 로 붙어서)")


def obstacle_rammer():
    b = _slice("H", 0, 250)
    b.scenario.actors = [_static(b, 901, "obstacle", 150.0, (0.15, 0.46, 0.61))]
    return _scenario("obstacle_rammer", b, _script(Cruise(25 * KMH)), {11}, "라바콘 들이받기")


def vehicle_rammer():
    b = _slice("H", 0, 250)
    b.scenario.actors = [_static(b, 902, "vehicle", 150.0, (4.5, 1.8, 1.5))]
    return _scenario("vehicle_rammer", b, _script(Cruise(25 * KMH)), {14}, "서 있는 차 들이받기")


def ped_ignorer():
    b = _slice("H", 0, 250)
    x0, y0, h = _route_point(b, 180.0, -4.0)
    cx, cy, _ = _route_point(b, 180.0, 0.0)
    b.scenario.actors = [rs.Actor(id=903, type="pedestrian", size=[0.6, 0.7, 1.8],
                                  spawn={"ego_within": 30.0, "of": [cx, cy]},
                                  motion={"kind": "crossing", "pos": [x0, y0],
                                          "vel": [-1.3 * math.sin(h), 1.3 * math.cos(h)]})]
    return _scenario("ped_ignorer", b, _script(Cruise(25 * KMH)), {10}, "건너는 사람 앞을 안 서고 통과")


def centerline_crosser():
    return _scenario("centerline_crosser", _slice("H", 0, 250),
                     _script(Cruise(20 * KMH), LaneShift(150.0, 1.8, 15.0)), {4},
                     "중앙선 너머로 1.8 m 붙어 달리기")


def sidewalk_rider():
    return _scenario("sidewalk_rider", _slice("H", 0, 250),
                     _script(Cruise(15 * KMH), LaneShift(150.0, -4.8, 25.0)), {5},
                     "오른쪽 보도 쪽으로 4.8 m 붙어 달리기")


def unsignaled_lane_change():
    return _scenario("unsignaled_lane_change", _slice("G", 2575, 2841),
                     _script(Cruise(30 * KMH), LaneShift(100.0, 3.3, 40.0)), {13},
                     "지시등 없이 왼쪽 차로로")


def signaled_lane_change():
    return _scenario("signaled_lane_change", _slice("G", 2575, 2841),
                     _script(Cruise(30 * KMH), LaneShift(100.0, 3.3, 40.0), SignalWindow(55.0, 150.0, 1)),
                     set(), "5 초 앞서 좌측 지시등을 켜고 왼쪽 차로로", forbid={13})


def teacher_H_0_250():
    return _scenario("teacher_H_0_250", _slice("H", 0, 250), lambda b: TeacherDriver(b), set(),
                     "선생님 — 기대 항목 없이 일치만 본다")


SCENARIOS = {f.__name__: f for f in (
    speeder, zone_speeder, red_runner, red_far_stopper, flash_runner, green_idler,
    crosswalk_stopper, obstacle_rammer, vehicle_rammer, ped_ignorer, centerline_crosser,
    sidewalk_rider, unsignaled_lane_change, signaled_lane_change, teacher_H_0_250)}


@functools.lru_cache(maxsize=None)
def episode_rows(name):
    sc = SCENARIOS[name]()
    run = run_episode(sc.board, sc.make_driver(sc.board))
    return sc, tuple(run.rows)
```

- [ ] **Step 4: 통과 확인과 대본 조정**

Run: `env -u PYTHONPATH .venv/bin/pytest tests/referee/test_scenarios.py -q; echo rc=$?`
Expected: `rc=0`

판 하나가 기대 항목을 못 일으키면 그 판의 대본 수치만 고친다. 원인은 행에서 찾는다. `run_episode(sc.board, sc.make_driver(sc.board))` 를 직접 돌려 `result.outcome` 으로 월드가 판을 일찍 끝냈는지 본다. 행 복사본을 `rs.check_lanes.classify(rows, rs.load_map())` 에 넣어 `center_intrusion`·`sidewalk_intrusion`·`lane_intrusion`·`off` 를 본다. 판마다 볼 점:
- `sidewalk_rider`: 보도 겹침 깊이가 0.5 m 에 못 미치면 오프셋을 조금씩 키운다. 월드가 `offroad` 로 끝내면 줄인다. 오른쪽 여유는 `max(r, pr) + 1.5 ≈ 5.7 m` 다.
- `crosswalk_stopper`: 차체 중앙(`뒷축 + FRONT/2`)이 횡단보도 중심 3.0 m 안에 서야 한다.
- `ped_ignorer`: 사람이 앞범퍼 3 m 안(`0 ≤ fx ≤ FRONT+3`, `|fy| ≤ 3`)에 있을 때 자차가 움직이고 있어야 한다.
- `zone_speeder`: 교차로(788 m) 전에 판이 끝나도 된다.

**조정해도 안 되는 판**은 이렇게 처리한다. `expect` 를 비운다. `note` 에 "채점기 미발생: <행에서 본 이유>" 를 적는다. 구현 보고에 그 판과 이유를 적는다. 테스트 문턱이나 채점기를 고쳐서 통과시키지 않는다.

- [ ] **Step 5: 커밋**

```bash
git add vtd_rl/referee/scenarios.py tests/referee/test_scenarios.py
git commit -m "일치 검증 대본 판 15개 — 채점기가 기대 항목을 일으킴"
```

---

### Task 5: 심판 뼈대 — 구간 색인, 온라인 구간 추적, Sheet 반영, 일치 비교, ① ② 판정

**Files:**
- Create: `vtd_rl/referee/sections.py`, `vtd_rl/referee/core.py`, `vtd_rl/referee/parity.py`, `vtd_rl/referee/judges/__init__.py`, `vtd_rl/referee/judges/speed.py`
- Test: `tests/referee/test_sections.py`, `tests/referee/test_core.py`, `tests/referee/test_parity_speed.py`

**Interfaces:**
- Consumes: `rs.score_fma`(`Sections`, `Sheet`, `spans`, 속도 문턱), `match_inputs`, `make_lim_at`, `make_lc_at`, `score_episode`, `SCENARIOS`, `episode_rows`
- Produces:
  - `FastSections(route, n)` — `score_fma.Sections` 하위 클래스. `index_of` 가 원본과 같은 점 번호를 준다(거리가 같으면 작은 번호)
  - `Hit(t: float, sec: int, item: int, level: str, why: str = "")` — `t` 는 위반이 일어난 행의 시각이다(판정을 낸 시각이 아니다)
  - `SpanTracker(pred, min_sec)` — `.update(row) -> list[(t0, t1, start_row, span_rows)]`, `.finish() -> list[...]`. `score_fma.spans` 와 같은 구간을 같은 순서로 낸다
  - `Context(board, sections, secs, lim_at, lc_at, tl_stops, cws, notes)`, `Context.build(board, sections=5)`
  - 판정기 규약: 클래스 속성 `items: tuple[int, ...]`, `__init__(self, ctx)`, `step(self, row) -> list[Hit]`, `finish(self) -> list[Hit]`
  - `default_judges(use_map: bool) -> list[type]` — 이 작업에서는 `[SpeedJudge]`. 뒤 작업이 차례로 넣는다
  - `Referee(board, sections=5, use_map=True, judges=None)` — `judges` 는 판정기 **클래스** 목록이다(None 이면 `default_judges(use_map)`). `.step(row) -> list[Hit]` 은 `use_map` 이면 `check_lanes.classify([row], map)` 로 행을 제자리에서 바꾼 뒤 판정기를 순서대로 부른다. `.finish() -> list[Hit]`, `.hits: list[Hit]`, `.sheet: score_fma.Sheet`, `.respawns: dict[int, list[float]]`(항목 15 는 Sheet 대신 여기로), `.ctx`
  - `ALL_ITEMS = tuple(range(1, 16))`, `Parity(want, got, respawns_want, respawns_got, state_want, state_got, frames, referee_seconds)` + `.ok` + `.diff()` + `.per_frame_us`
  - `compare(board, rows, items=ALL_ITEMS, sections=5, use_map=True) -> Parity` — `items` 밖의 호출은 양쪽에서 뺀다. Sheet 최종 상태 비교는 `items == ALL_ITEMS` 일 때만 한다

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/referee/test_sections.py`
```python
import os
import random

import pytest

from vtd_rl import rule_stack as rs
from vtd_rl.referee.sections import FastSections
from vtd_rl.world.board import load_curriculum

STAGE1 = os.path.join(os.path.dirname(__file__), "..", "..", "curricula", "stage1.json")
_, BOARDS = load_curriculum(STAGE1)


@pytest.mark.parametrize("board", BOARDS, ids=lambda b: b.name)
def test_가장_가까운_점은_원본과_같다(board):
    route = [list(p) for p in board.route.pts]
    slow, fast = rs.score_fma.Sections(route, 5), FastSections(route, 5)
    rng = random.Random(0)
    xs, ys = [p[0] for p in route], [p[1] for p in route]
    queries = [(rng.uniform(min(xs) - 40, max(xs) + 40), rng.uniform(min(ys) - 40, max(ys) + 40))
               for _ in range(200)]
    queries += [(p[0] + 0.3, p[1] - 0.2) for p in route[::37]]
    queries += [tuple(p) for p in route[::53]]
    for x, y in queries:
        assert fast.index_of(x, y) == slow.index_of(x, y)
        assert fast.of(x, y) == slow.of(x, y)


def test_거리가_같으면_작은_번호():
    route = [[0.0, 0.0], [10.0, 0.0], [0.0, 0.0], [20.0, 0.0]]
    assert FastSections(route, 2).index_of(0.0, 0.0) == 0
    assert FastSections(route, 2).index_of(5.0, 0.0) == rs.score_fma.Sections(route, 2).index_of(5.0, 0.0)
```

`tests/referee/test_core.py`
```python
import collections
import copy
import random

from vtd_rl import rule_stack as rs
from vtd_rl.referee.core import Hit, Referee, SpanTracker
from vtd_rl.referee.judges.speed import SpeedJudge
from vtd_rl.referee.oracle import score_episode
from vtd_rl.world.board import load_board, slice_board

H = {"name": "course_H", "route": "routes/HL_FMA_NEW_H.json", "lane": "routes/HL_FMA_NEW_H_lane.json"}


def h_slice():
    return slice_board(load_board(H), 0.0, 250.0, "H_0_250")


def test_온라인_구간은_spans_와_같다():
    rng = random.Random(3)
    for _ in range(300):
        rows = [dict(t=round(k * 0.05, 2), on=rng.random() < 0.6) for k in range(rng.randint(0, 120))]
        pred = lambda r: r["on"]
        want = rs.score_fma.spans(rows, pred, 0.3)
        tr = SpanTracker(pred, 0.3)
        got = []
        for r in rows:
            got += tr.update(r)
        got += tr.finish()
        assert [(t0, t1, r0) for t0, t1, r0, _ in got] == want
        for t0, t1, _, span_rows in got:
            assert span_rows == [r for r in rows if t0 <= r["t"] <= t1]


def test_심판은_항목15를_Sheet_대신_따로_모은다():
    class Fake:
        items = (1, 15)

        def __init__(self, ctx):
            self.ctx = ctx

        def step(self, r):
            return [Hit(r["t"], 0, 1, "minor"), Hit(r["t"], 1, 15, "major")]

        def finish(self):
            return [Hit(9.0, 0, 1, "major")]

    b = h_slice()
    x, y, h = b.route.point_at(10.0)
    ref = Referee(b, use_map=False, judges=[Fake])
    ref.step(dict(t=1.0, x=x, y=y, h=h, v=1.0, tl=0, tid=0, reason="", sig=0, clr=99.9,
                  d_ego=0.0, objs="", cap_by=""))
    ref.finish()
    assert ref.sheet.state[0] == {1: "major"} and ref.respawns == {1: [1.0]}
    assert [hit.item for hit in ref.hits] == [1, 15, 1]


def test_속도_판정은_합성_행에서_채점기와_같다():
    b = h_slice()
    rows, s = [], 0.0
    for k in range(300):
        v = [10.0, 14.2, 20.0][k // 100]            # 규정 · 경미(+1.1 km/h) · 중대(+22 km/h)
        x, y, h = b.route.point_at(s)
        rows.append(dict(t=round(k * 0.05, 2), x=x, y=y, h=h, v=v, tl=0, tid=0, reason="", sig=0,
                         clr=99.9, d_ego=0.0, objs="", cap_by=""))
        s += v * 0.05
    want = collections.Counter(score_episode(rows, b, use_map=False).hits)
    ref = Referee(b, use_map=False, judges=[SpeedJudge])
    for r in rows:
        ref.step(copy.deepcopy(r))
    ref.finish()
    assert collections.Counter((hit.sec, hit.item, hit.level) for hit in ref.hits) == want
    assert {lv for _, _, lv in want} == {"minor", "major"}
```

`tests/referee/test_parity_speed.py`
```python
import pytest

from vtd_rl.referee.parity import compare
from vtd_rl.referee.scenarios import SCENARIOS, episode_rows


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_속도_판정_일치(name):
    sc, rows = episode_rows(name)
    p = compare(sc.board, rows, items=(1, 2))
    assert p.ok, p.diff()
```

- [ ] **Step 2: 실패 확인**

Run: `env -u PYTHONPATH .venv/bin/pytest tests/referee/test_sections.py tests/referee/test_core.py tests/referee/test_parity_speed.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'vtd_rl.referee.sections'`

- [ ] **Step 3: 구간 색인 구현**

`vtd_rl/referee/sections.py`
```python
"""score_fma.Sections 와 같은 답을 빠르게 — 경로점 격자 색인.

원본 index_of 는 경로점 전체를 훑어 가장 가까운 점 번호를 준다(거리가 같으면 앞 번호).
여기서는 CELL 격자의 고리를 안에서 밖으로 넓혀 간다. 고리 r 까지 본 뒤 남은 점은 모두
r*CELL + (자기 칸 경계까지 거리) 이상 떨어져 있다. 지금까지의 최소 거리가 그보다 작으면 멈춘다.
거리 식은 원본과 같은 식으로 계산해 동률 판단도 원본과 같게 한다.
"""
import math

from vtd_rl import rule_stack as rs

CELL = 12.0


class FastSections(rs.score_fma.Sections):
    def __init__(self, route, n):
        super().__init__(route, n)
        self._grid = {}
        for i, (x, y) in enumerate(self.route):
            self._grid.setdefault((math.floor(x / CELL), math.floor(y / CELL)), []).append(i)
        self._last = (None, None, None)       # 같은 행을 여러 판정기가 물으므로 직전 답을 둔다

    def index_of(self, x, y):
        if self._last[0] == x and self._last[1] == y:
            return self._last[2]
        cx, cy = math.floor(x / CELL), math.floor(y / CELL)
        edge = min(x - cx * CELL, (cx + 1) * CELL - x, y - cy * CELL, (cy + 1) * CELL - y)
        best_i, best_d = None, None
        ring = 0
        while True:
            for gx in range(cx - ring, cx + ring + 1):
                for gy in range(cy - ring, cy + ring + 1):
                    if max(abs(gx - cx), abs(gy - cy)) != ring:
                        continue
                    for i in self._grid.get((gx, gy), ()):
                        d = (self.route[i][0] - x) ** 2 + (self.route[i][1] - y) ** 2
                        if best_d is None or d < best_d or (d == best_d and i < best_i):
                            best_i, best_d = i, d
            bound = ring * CELL + edge
            if best_d is not None and best_d < bound * bound:
                break
            ring += 1
        self._last = (x, y, best_i)
        return best_i
```

- [ ] **Step 4: 심판 뼈대 구현**

`vtd_rl/referee/core.py`
```python
"""온라인 심판 뼈대.

- SpanTracker: score_fma.spans() 를 한 행씩. 참이 이어지다 거짓 행이 오면 그 구간을 닫는다(길이 조건 같음).
- Context: score_fma.main() 과 같은 보조 데이터(구간·제한속도·차로변경 표식·정지선·횡단보도).
- Referee: 행마다 (지도 판정) -> 판정기들 -> Sheet 반영. 판정기는 미래 행을 보지 않고, 필요하면 늦게 낸다.
"""
from dataclasses import dataclass, field

from vtd_rl import rule_stack as rs
from vtd_rl.referee.oracle import make_lc_at, make_lim_at, match_inputs
from vtd_rl.referee.sections import FastSections

sf = rs.score_fma


@dataclass
class Hit:
    t: float
    sec: int
    item: int
    level: str
    why: str = ""


class SpanTracker:
    def __init__(self, pred, min_sec):
        self.pred, self.min_sec = pred, min_sec
        self._rows = []

    def _close(self):
        rows, self._rows = self._rows, []
        if rows and rows[-1]["t"] - rows[0]["t"] >= self.min_sec - 1e-9:
            return [(rows[0]["t"], rows[-1]["t"], rows[0], rows)]
        return []

    def update(self, row):
        if self.pred(row):
            self._rows.append(row)
            return []
        return self._close()

    def finish(self):
        return self._close()


@dataclass
class Context:
    board: object
    sections: int
    secs: object
    lim_at: object
    lc_at: object
    tl_stops: dict
    cws: list
    notes: list = field(default_factory=list)

    @classmethod
    def build(cls, board, sections=5):
        m = match_inputs(board)
        secs = FastSections(m.route, sections)
        return cls(board, sections, secs, make_lim_at(secs, m.lims), make_lc_at(secs, m.lc_flag),
                   m.tl_stops, m.cws)


def default_judges(use_map):
    from vtd_rl.referee.judges.speed import SpeedJudge
    return [SpeedJudge]


class Referee:
    def __init__(self, board, sections=5, use_map=True, judges=None):
        self.ctx = Context.build(board, sections)
        self.map = rs.load_map() if use_map else None
        classes = default_judges(use_map) if judges is None else judges
        self.judges = [cls(self.ctx) for cls in classes]
        self.sheet = sf.Sheet(self.ctx.secs.n)
        self.hits = []
        self.respawns = {}

    def _apply(self, hits):
        for h in hits:
            if h.item == 15:
                self.respawns.setdefault(h.sec, []).append(h.t)
            else:
                self.sheet.hit(h.sec, h.item, h.level, h.why)
            self.hits.append(h)
        return hits

    def step(self, row):
        if self.map is not None:
            rs.check_lanes.classify([row], self.map)
        out = []
        for j in self.judges:
            out += j.step(row)
        return self._apply(out)

    def finish(self):
        out = []
        for j in self.judges:
            out += j.finish()
        return self._apply(out)
```

`vtd_rl/referee/parity.py`
```python
"""일치 검증 — 같은 행을 사후 채점기와 심판에 넣고 감점 호출을 비교한다."""
import collections
import time
from dataclasses import dataclass

from vtd_rl.referee.core import Referee
from vtd_rl.referee.oracle import score_episode

ALL_ITEMS = tuple(range(1, 16))


@dataclass
class Parity:
    want: collections.Counter
    got: collections.Counter
    respawns_want: dict
    respawns_got: dict
    state_want: object          # 전 항목 비교일 때만 list, 아니면 None
    state_got: object
    frames: int
    referee_seconds: float

    @property
    def ok(self) -> bool:
        return (self.want == self.got and self.respawns_want == self.respawns_got
                and self.state_want == self.state_got)

    @property
    def per_frame_us(self) -> float:
        return self.referee_seconds / max(self.frames, 1) * 1e6

    def diff(self) -> str:
        return (f"채점기에만 {dict(self.want - self.got)} / 심판에만 {dict(self.got - self.want)} / "
                f"리스폰 채점기 {self.respawns_want} 심판 {self.respawns_got} / "
                f"Sheet 채점기 {self.state_want} 심판 {self.state_got}")


def compare(board, rows, items=ALL_ITEMS, sections=5, use_map=True) -> Parity:
    items = tuple(items)
    oracle = score_episode(rows, board, sections, use_map)
    ref = Referee(board, sections, use_map)
    t0 = time.perf_counter()
    for r in rows:
        ref.step(dict(r))
    ref.finish()
    seconds = time.perf_counter() - t0
    want = collections.Counter(h for h in oracle.hits if h[1] in items)
    got = collections.Counter((h.sec, h.item, h.level) for h in ref.hits
                              if h.item in items and h.item != 15)
    full = items == ALL_ITEMS
    return Parity(want, got,
                  oracle.respawns if 15 in items else {}, ref.respawns if 15 in items else {},
                  oracle.state if full else None,
                  [dict(s) for s in ref.sheet.state] if full else None,
                  len(rows), seconds)
```

`vtd_rl/referee/judges/__init__.py`
```python
"""항목별 판정기 — 각 모듈은 score_fma 의 item_* 하나를 한 행씩 받는 형태로 옮긴 것이다."""
```

`vtd_rl/referee/judges/speed.py`
```python
"""① 제한속도 · ② 보호구역 속도 — score_fma.item_speed 의 한 행 판."""
from vtd_rl import rule_stack as rs
from vtd_rl.referee.core import Hit

sf = rs.score_fma


class SpeedJudge:
    items = (1, 2)

    def __init__(self, ctx):
        self.ctx = ctx

    def step(self, r):
        lim = self.ctx.lim_at(r["x"], r["y"])
        if lim is None:
            return []
        over = r["v"] - lim
        if lim <= sf.ZONE_LIM:
            item, major, tol = 2, sf.ZONE_MAJOR, sf.ZONE_TOL
        else:
            item, major, tol = 1, sf.SPD_MAJOR, sf.SPD_TOL
        level = "major" if over > major else ("minor" if over > tol else None)
        if level is None:
            return []
        return [Hit(r["t"], self.ctx.secs.of(r["x"], r["y"]), item, level,
                    f"t={r['t']:.1f} {r['v']*3.6:.0f}>{lim*3.6:.0f}km/h")]

    def finish(self):
        return []
```

- [ ] **Step 5: 통과 확인**

Run: `env -u PYTHONPATH .venv/bin/pytest tests/referee -q; echo rc=$?`
Expected: `rc=0`

- [ ] **Step 6: 커밋**

```bash
git add vtd_rl/referee tests/referee
git commit -m "심판 뼈대 — 구간 격자 색인·온라인 구간 추적·Sheet 반영·일치 비교, ①② 속도 판정"
```

---

### Task 6: ⑦ ⑧ ⑨ 신호 판정기

**Files:**
- Create: `vtd_rl/referee/judges/traffic_light.py`, `tests/referee/test_parity_traffic_light.py`
- Modify: `vtd_rl/referee/core.py` (`default_judges`)

**Interfaces:**
- Consumes: `Hit`, `SpanTracker`, `Context`(`secs`, `tl_stops`, `notes`), `compare`, `Referee(..., judges=[...])`, `sf.FRONT`, `sf.STOP_NEAR`, `sf.STOP_HOLD`, `sf.STOP_V`, `sf.GREEN_NEAR`, `sf.GREEN_MINOR`, `sf.GREEN_MAJOR`, `sf.SAFE_MAJORITY`, `sf.LAWFUL_WAIT`, `sf._safe_object_cause`
- Produces: `TrafficLightJudge` (`items = (7, 8, 9)`)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/referee/test_parity_traffic_light.py`
```python
import pytest

from vtd_rl.referee.core import Referee
from vtd_rl.referee.judges.traffic_light import TrafficLightJudge
from vtd_rl.referee.parity import compare
from vtd_rl.referee.scenarios import SCENARIOS, episode_rows


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_신호_판정_일치(name):
    sc, rows = episode_rows(name)
    p = compare(sc.board, rows, items=(7, 8, 9))
    assert p.ok, p.diff()


def test_적색_무정차는_선을_넘는_프레임에_낸다():
    sc, rows = episode_rows("red_runner")
    ref = Referee(sc.board, use_map=False, judges=[TrafficLightJudge])
    emitted_at = None
    for r in rows:
        if ref.step(dict(r)) and emitted_at is None:
            emitted_at = r["t"]
    ref.finish()
    assert emitted_at is not None
    assert [h.t for h in ref.hits if h.item == 7] == [emitted_at]


def test_녹색_정차는_정차가_끝날_때_낸다():
    sc, rows = episode_rows("green_idler")
    ref = Referee(sc.board, use_map=False, judges=[TrafficLightJudge])
    for r in rows:
        for h in ref.step(dict(r)):
            assert h.item == 8 and r["t"] - h.t >= 10.0
    ref.finish()
    assert [h.level for h in ref.hits] == ["major"]
```

- [ ] **Step 2: 실패 확인**

Run: `env -u PYTHONPATH .venv/bin/pytest tests/referee/test_parity_traffic_light.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'vtd_rl.referee.judges.traffic_light'`

- [ ] **Step 3: 구현**

`vtd_rl/referee/judges/traffic_light.py`
```python
"""⑦ 적색 정지 · ⑧ 녹색 무의미 정차 · ⑨ 적색점멸 — score_fma.item_traffic_light 의 한 행 판.

⑦⑨ 는 (신호, 상태)마다 선을 처음 넘는 프레임에 확정한다. 채점기는 판 끝에 보므로, 같은 (신호, 상태)로
다시 다가가 서는 드문 경우만 답이 다르다(M2a 계획 Global Constraints).
⑧ 은 정차 구간 전체의 다수결 면책이 있어 구간이 닫힐 때 낸다.
"""
import math

from vtd_rl import rule_stack as rs
from vtd_rl.referee.core import Hit, SpanTracker

sf = rs.score_fma
TL_RED, TL_GREEN, TL_LEFT, TL_GREEN_LEFT, TL_FLASH = 1, 3, 4, 5, 6   # item_traffic_light 안의 지역 상수


def _stop_point(ctx, tid):
    return ctx.tl_stops.get(str(tid)) or ctx.tl_stops.get(tid)


def _forward(p, r):
    dx, dy = p[0] - r["x"], p[1] - r["y"]
    return dx * math.cos(-r["h"]) - dy * math.sin(-r["h"])


class TrafficLightJudge:
    items = (7, 8, 9)

    def __init__(self, ctx):
        self.ctx = ctx
        self.held, self.far_held, self.ahead = {}, {}, {}
        self.stopped_since, self.far_stopped_since = {}, {}
        self.crossed = set()
        self.idle = SpanTracker(self._green_idle, sf.GREEN_MINOR)

    def step(self, r):
        return self._red(r) + self._idle_hits(self.idle.update(r))

    def finish(self):
        return self._idle_hits(self.idle.finish())

    def _red(self, r):
        tid = r["tid"]
        p = _stop_point(self.ctx, tid)
        if tid < 0 or p is None:
            return []
        fwd = _forward(p, r) - sf.FRONT                       # 앞범퍼 기준
        key = (tid, r["tl"])
        if fwd > 0.0:
            self.ahead[key] = True
        if r["tl"] not in (TL_RED, TL_FLASH):
            return []
        if 0.0 <= fwd < sf.STOP_NEAR and r["v"] <= sf.STOP_V:
            self.far_stopped_since.pop(key, None)
            t0 = self.stopped_since.setdefault(key, r["t"])
            if r["t"] - t0 >= sf.STOP_HOLD:
                self.held[key] = True
        elif fwd >= sf.STOP_NEAR and r["v"] <= sf.STOP_V:
            self.stopped_since.pop(key, None)
            t0 = self.far_stopped_since.setdefault(key, r["t"])
            if r["t"] - t0 >= sf.STOP_HOLD:
                self.far_held[key] = True
        else:
            self.stopped_since.pop(key, None)
            self.far_stopped_since.pop(key, None)
        if not (fwd < -1.0 and self.ahead.get(key)) or key in self.crossed:
            return []
        self.crossed.add(key)
        if self.held.get(key):
            return []
        item = 7 if r["tl"] == TL_RED else 9
        sec = self.ctx.secs.of(r["x"], r["y"])
        if self.far_held.get(key):
            return [Hit(r["t"], sec, item, "minor",
                        f"t={r['t']:.1f} 신호{tid} 정지선 {sf.STOP_NEAR}m 이상 앞에 정지")]
        return [Hit(r["t"], sec, item, "major", f"t={r['t']:.1f} 신호{tid} 정지 없이 통과")]

    def _green_idle(self, r):
        p = _stop_point(self.ctx, r["tid"])
        if p is None or r["tl"] not in (TL_GREEN, TL_LEFT, TL_GREEN_LEFT):
            return False
        return 0.0 < _forward(p, r) < sf.GREEN_NEAR and r["v"] <= sf.STOP_V   # 뒷축 기준

    def _idle_hits(self, spans):
        out = []
        for t0, t1, r0, span_rows in spans:
            held_s = t1 - t0
            why = f"t={t0:.1f}~{t1:.1f} 신호{r0['tid']} 녹색에 {held_s:.1f}초 정차"
            obj_ok = sum(1 for r in span_rows if sf._safe_object_cause(r))
            if obj_ok > len(span_rows) * sf.SAFE_MAJORITY:
                continue
            lawful = [r["reason"] for r in span_rows if r["reason"] in sf.LAWFUL_WAIT]
            if len(lawful) > len(span_rows) * sf.SAFE_MAJORITY:
                self.ctx.notes.append(f"[⑧ 판정 불명] {why}")
                continue
            out.append(Hit(t0, self.ctx.secs.of(r0["x"], r0["y"]), 8,
                           "major" if held_s >= sf.GREEN_MAJOR else "minor", why))
        return out
```

`vtd_rl/referee/core.py` 의 `default_judges` 를 바꾼다:
```python
def default_judges(use_map):
    from vtd_rl.referee.judges.speed import SpeedJudge
    from vtd_rl.referee.judges.traffic_light import TrafficLightJudge
    return [SpeedJudge, TrafficLightJudge]
```

- [ ] **Step 4: 통과 확인**

Run: `env -u PYTHONPATH .venv/bin/pytest tests/referee -q; echo rc=$?`
Expected: `rc=0`

- [ ] **Step 5: 커밋**

```bash
git add vtd_rl/referee tests/referee/test_parity_traffic_light.py
git commit -m "심판 — ⑦⑧⑨ 신호 판정(선을 넘는 프레임·정차 구간이 닫힐 때), 채점기와 일치"
```

---

### Task 7: ⑪ ⑭ 접촉 · ⑫ 횡단보도 위 정지 판정기

**Files:**
- Create: `vtd_rl/referee/judges/contact.py`, `tests/referee/test_parity_contact.py`
- Modify: `vtd_rl/referee/core.py` (`default_judges`)

**Interfaces:**
- Consumes: `Hit`, `SpanTracker`, `Context`(`secs`, `cws`), `compare`, `sf.parse_objs`, `sf.PED_L`, `sf.PED_W`, `sf.FRONT`, `sf.STOP_V`, `sf.CW_STOP_S`
- Produces: `ContactJudge` (`items = (11, 14)`), `CrosswalkStopJudge` (`items = (12,)`)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/referee/test_parity_contact.py`
```python
import pytest

from vtd_rl.referee.parity import compare
from vtd_rl.referee.scenarios import SCENARIOS, episode_rows


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_접촉과_횡단보도_정지_판정_일치(name):
    sc, rows = episode_rows(name)
    p = compare(sc.board, rows, items=(11, 12, 14))
    assert p.ok, p.diff()
```

- [ ] **Step 2: 실패 확인**

Run: `env -u PYTHONPATH .venv/bin/pytest tests/referee/test_parity_contact.py -q`
Expected: FAIL — `obstacle_rammer`·`vehicle_rammer`·`ped_ignorer`·`crosswalk_stopper` 에서 `채점기에만 {...}`

- [ ] **Step 3: 구현**

`vtd_rl/referee/judges/contact.py`
```python
"""⑪ 장애물 충돌 · ⑭ 차량·보행자 접촉 · ⑫ 횡단보도 위 정지 — score_fma.item_contact, item_crosswalk_stop."""
import math

from vtd_rl import rule_stack as rs
from vtd_rl.referee.core import Hit, SpanTracker

sf = rs.score_fma
PERSON_MIN_H = 1.2      # item_contact 안의 숫자 — 사람·이륜차 최저 키[m]
CW_RADIUS = 3.0         # item_crosswalk_stop 안의 숫자 — 횡단보도 중심 반경[m]


class ContactJudge:
    items = (11, 14)

    def __init__(self, ctx):
        self.ctx = ctx

    def step(self, r):
        if r["clr"] >= 0.0:
            return []
        sec = self.ctx.secs.of(r["x"], r["y"])
        near = min(sf.parse_objs(r["objs"]), key=lambda o: o[6], default=None)
        if near is None:
            return [Hit(r["t"], sec, 11, "minor", f"t={r['t']:.1f} 여유 {r['clr']:.2f}m")]
        _fx, _fy, ol, ow, oh, _sp, _clr = near
        person = ol <= sf.PED_L and ow <= sf.PED_W and oh >= PERSON_MIN_H
        vehicle = ol > sf.PED_L
        item = 14 if (person or vehicle) else 11
        return [Hit(r["t"], sec, item, "major" if item == 14 else "minor",
                    f"t={r['t']:.1f} 여유 {r['clr']:.2f}m (상대 {ol:.1f}x{ow:.1f}m)")]

    def finish(self):
        return []


class CrosswalkStopJudge:
    items = (12,)

    def __init__(self, ctx):
        self.ctx = ctx
        self.span = SpanTracker(self._on_crosswalk, sf.CW_STOP_S)

    def _on_crosswalk(self, r):
        fx = r["x"] + sf.FRONT / 2 * math.cos(r["h"])     # 채점기가 쓰는 '차체 중앙쯤'
        fy = r["y"] + sf.FRONT / 2 * math.sin(r["h"])
        return (r["v"] <= sf.STOP_V
                and any(math.hypot(fx - c["x"], fy - c["y"]) < CW_RADIUS for c in self.ctx.cws))

    def _hits(self, spans):
        return [Hit(t0, self.ctx.secs.of(r0["x"], r0["y"]), 12, "minor",
                    f"t={t0:.1f}~{t1:.1f} 횡단보도 위 {t1-t0:.1f}초 정지")
                for t0, t1, r0, _ in spans]

    def step(self, r):
        return self._hits(self.span.update(r))

    def finish(self):
        return self._hits(self.span.finish())
```

`vtd_rl/referee/core.py` 의 `default_judges`:
```python
def default_judges(use_map):
    from vtd_rl.referee.judges.contact import ContactJudge, CrosswalkStopJudge
    from vtd_rl.referee.judges.speed import SpeedJudge
    from vtd_rl.referee.judges.traffic_light import TrafficLightJudge
    return [SpeedJudge, TrafficLightJudge, ContactJudge, CrosswalkStopJudge]
```

- [ ] **Step 4: 통과 확인**

Run: `env -u PYTHONPATH .venv/bin/pytest tests/referee -q; echo rc=$?`
Expected: `rc=0`

- [ ] **Step 5: 커밋**

```bash
git add vtd_rl/referee tests/referee/test_parity_contact.py
git commit -m "심판 — ⑪⑭ 접촉·⑫ 횡단보도 위 정지 판정, 채점기와 일치"
```

---

### Task 8: ⑩ 보행자 대응 판정기

**Files:**
- Create: `vtd_rl/referee/judges/pedestrian.py`, `tests/referee/test_parity_pedestrian.py`
- Modify: `vtd_rl/referee/core.py` (`default_judges`)

**Interfaces:**
- Consumes: `Hit`, `Context`(`secs`), `compare`, `episode_rows`, `Referee(..., judges=[...])`, `sf.parse_objs`, `sf.STOP_V`, `sf.PED_L`, `sf.PED_W`, `sf.PED_NEAR`, `sf.FRONT`
- Produces: `PedestrianJudge` (`items = (10,)`) — 후보 프레임에서 1 초가 지난 첫 행에서 판정한다

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/referee/test_parity_pedestrian.py`
```python
import pytest

from vtd_rl.referee.core import Referee
from vtd_rl.referee.judges.pedestrian import PedestrianJudge
from vtd_rl.referee.parity import compare
from vtd_rl.referee.scenarios import SCENARIOS, episode_rows


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_보행자_판정_일치(name):
    sc, rows = episode_rows(name)
    p = compare(sc.board, rows, items=(10,))
    assert p.ok, p.diff()


def test_판정은_1초_뒤에_낸다():
    sc, rows = episode_rows("ped_ignorer")
    ref = Referee(sc.board, use_map=False, judges=[PedestrianJudge])
    for r in rows:
        for h in ref.step(dict(r)):
            assert r["t"] > h.t + 1.0
    ref.finish()
    assert any(h.item == 10 for h in ref.hits)
```

- [ ] **Step 2: 실패 확인**

Run: `env -u PYTHONPATH .venv/bin/pytest tests/referee/test_parity_pedestrian.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'vtd_rl.referee.judges.pedestrian'`

- [ ] **Step 3: 구현**

`vtd_rl/referee/judges/pedestrian.py`
```python
"""⑩ 보행자 대응 — score_fma.item_pedestrian 의 한 행 판.

채점기는 후보 프레임 t 에서 [t-6, t+1] 초 안에 한 번이라도 섰는지 본다. 그래서 1 초 뒤에 판정한다.
한 프레임에 사람이 여럿이어도 '섰나'는 시각만 보므로 프레임당 감점은 많아야 한 번이다.
"""
from collections import deque

from vtd_rl import rule_stack as rs
from vtd_rl.referee.core import Hit

sf = rs.score_fma
PED_STOP_WIN = 6.0      # item_pedestrian 안의 지역 상수 — 이 시간 안에 섰으면 '섰다'[s]
PED_AFTER = 1.0         # item_pedestrian.stopped_near 의 뒤쪽 창[s]


class PedestrianJudge:
    items = (10,)

    def __init__(self, ctx):
        self.ctx = ctx
        self._stops = deque()      # STOP_V 이하였던 행의 t
        self._pending = deque()    # (후보 행, 첫 사람의 fy)

    def _person_ahead(self, r):
        if r["v"] <= sf.STOP_V:
            return None
        for fx, fy, ol, ow, _oh, sp, _clr in sf.parse_objs(r["objs"]):
            if not (ol <= sf.PED_L and ow <= sf.PED_W):
                continue
            if not (0.0 <= fx <= sf.FRONT + sf.PED_NEAR):
                continue
            if not (0.3 <= sp <= 2.5):
                continue
            if abs(fy) <= sf.PED_NEAR:
                return fy
        return None

    def _judge(self, r, fy):
        t_at = r["t"]
        if any(t_at - PED_STOP_WIN <= s <= t_at + PED_AFTER for s in self._stops):
            return []
        return [Hit(t_at, self.ctx.secs.of(r["x"], r["y"]), 10, "major",
                    f"t={t_at:.1f} 보행자 옆 {abs(fy):.1f}m 를 {r['v']*3.6:.0f}km/h 로 안 서고 통과")]

    def step(self, r):
        if r["v"] <= sf.STOP_V:
            self._stops.append(r["t"])
        out = []
        while self._pending and r["t"] > self._pending[0][0]["t"] + PED_AFTER:
            out += self._judge(*self._pending.popleft())
        fy = self._person_ahead(r)
        if fy is not None:
            self._pending.append((r, fy))
        oldest = self._pending[0][0]["t"] if self._pending else r["t"]
        while self._stops and self._stops[0] < oldest - PED_STOP_WIN:
            self._stops.popleft()
        return out

    def finish(self):
        out = []
        while self._pending:
            out += self._judge(*self._pending.popleft())
        return out
```

`vtd_rl/referee/core.py` 의 `default_judges`:
```python
def default_judges(use_map):
    from vtd_rl.referee.judges.contact import ContactJudge, CrosswalkStopJudge
    from vtd_rl.referee.judges.pedestrian import PedestrianJudge
    from vtd_rl.referee.judges.speed import SpeedJudge
    from vtd_rl.referee.judges.traffic_light import TrafficLightJudge
    return [SpeedJudge, TrafficLightJudge, ContactJudge, PedestrianJudge, CrosswalkStopJudge]
```

- [ ] **Step 4: 통과 확인**

Run: `env -u PYTHONPATH .venv/bin/pytest tests/referee -q; echo rc=$?`
Expected: `rc=0`

- [ ] **Step 5: 커밋**

```bash
git add vtd_rl/referee tests/referee/test_parity_pedestrian.py
git commit -m "심판 — ⑩ 보행자 대응(1 초 뒤 판정), 채점기와 일치"
```

---

### Task 9: ③ ④ ⑤ ⑥ 차로 기하 판정기

**Files:**
- Create: `vtd_rl/referee/judges/lane_geometry.py`, `tests/referee/test_parity_lane_geometry.py`
- Modify: `vtd_rl/referee/core.py` (`default_judges`)

**Interfaces:**
- Consumes: `Hit`, `SpanTracker`, `Context`(`secs`, `lc_at`), `compare`, `Referee.step` 이 붙인 차로 판정 키(`lane`, `off`, `lane_intrusion`, `center_intrusion`, `sidewalk_intrusion`, `_bnd`, `obb_t_min`, `obb_t_max` 등), `rs.check_lanes.lane_change_events`, `sf.LANE_EDGE_M`, `sf.LANE_EDGE_S`, `sf.CENTER_M`, `sf.CENTER_S`, `sf.WALK_M`, `sf.WALK_S`
- Produces: `LaneGeometryJudge` (`items = (3, 4, 5, 6)`) — `use_map=False` 인 심판에는 넣지 않는다

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/referee/test_parity_lane_geometry.py`
```python
import pytest

from vtd_rl.referee.core import Referee, default_judges
from vtd_rl.referee.judges.lane_geometry import LaneGeometryJudge
from vtd_rl.referee.parity import compare
from vtd_rl.referee.scenarios import SCENARIOS, episode_rows


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_차로_기하_판정_일치(name):
    sc, rows = episode_rows(name)
    p = compare(sc.board, rows, items=(3, 4, 5, 6))
    assert p.ok, p.diff()


def test_지도_없는_심판에는_넣지_않는다():
    assert LaneGeometryJudge in default_judges(True)
    assert LaneGeometryJudge not in default_judges(False)


def test_차로_유지_판정은_2_5초_뒤에_낸다():
    sc, rows = episode_rows("centerline_crosser")
    ref = Referee(sc.board, use_map=True, judges=[LaneGeometryJudge])
    for r in rows:
        for h in ref.step(dict(r)):
            if h.item == 3:
                assert r["t"] >= h.t + 2.5
    ref.finish()
    assert any(h.item == 4 for h in ref.hits)
```

- [ ] **Step 2: 실패 확인**

Run: `env -u PYTHONPATH .venv/bin/pytest tests/referee/test_parity_lane_geometry.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'vtd_rl.referee.judges.lane_geometry'`

- [ ] **Step 3: 구현**

`vtd_rl/referee/judges/lane_geometry.py`
```python
"""③ 차로 유지 · ④ 중앙선 · ⑤ 보도 · ⑥ 실선 차로변경 — score_fma.item_lane_geometry 의 한 행 판.

행에는 Referee.step 이 check_lanes.classify 결과를 붙여 준다.
- 출발 직후 제외: 첫 행에서 8 m 넘게 가고 |d_ego| < 0.5 인 첫 행부터 본다(원본의 live).
- ⑥ 사건은 lane_change_events 를 [직전 행, 이 행] 에 불러 얻는다. 원본도 직전 행과만 비교한다.
- ③ 은 행마다 '차로변경 사건 ±2.5 초 안인가'를 봐야 해서 행을 2.5 초 붙잡아 두었다가 구간 추적에 넣는다.
"""
import math
from collections import deque

from vtd_rl import rule_stack as rs
from vtd_rl.referee.core import Hit, SpanTracker

sf = rs.score_fma
cl = rs.check_lanes
LC_WIN = 2.5            # item_lane_geometry 안의 지역 상수 — 차로변경 사건 앞뒤 이 시간은 '변경 중'[s]
SPAWN_M = 8.0           # item_lane_geometry 안의 숫자 — 출발점에서 이만큼은 뺀다[m]
ATTACHED_D = 0.5        # item_lane_geometry 안의 숫자 — 경로에 붙었다고 볼 |d_ego|[m]


def _over(key, limit):
    return lambda r: r.get(key) is not None and r[key] + 1e-9 >= limit


class LaneGeometryJudge:
    items = (3, 4, 5, 6)

    def __init__(self, ctx):
        self.ctx = ctx
        self.start = None
        self.live = False
        self.prev = None                 # lane_change_events 의 직전 행
        self.changes = deque()           # 차로변경 사건 시각
        self.held = deque()              # ③ 판정을 기다리는 행
        self.edge = SpanTracker(self._edge_pred, sf.LANE_EDGE_S)
        self.center = SpanTracker(_over("center_intrusion", sf.CENTER_M), sf.CENTER_S)
        self.walk = SpanTracker(_over("sidewalk_intrusion", sf.WALK_M), sf.WALK_S)
        self.n_edge, self.n_solid = {}, {}

    def _edge_pred(self, r):
        if not _over("lane_intrusion", sf.LANE_EDGE_M)(r):
            return False
        if self.ctx.lc_at(r["x"], r["y"]):
            return False
        return not any(abs(r["t"] - ct) < LC_WIN for ct in self.changes)

    def step(self, r):
        if self.start is None:
            self.start = (r["x"], r["y"])
        if not self.live:
            if not (math.hypot(r["x"] - self.start[0], r["y"] - self.start[1]) > SPAWN_M
                    and abs(r.get("d_ego") or 0.0) < ATTACHED_D):
                return []
            self.live = True
        out = self._lane_changes(r)
        self.held.append(r)
        out += self._release(lambda h: r["t"] >= h["t"] + LC_WIN)
        out += self._major_hits(4, self.center.update(r))
        out += self._major_hits(5, self.walk.update(r))
        return out

    def finish(self):
        if not self.live:
            return []
        out = self._release(lambda h: True)
        out += self._edge_hits(self.edge.finish())
        out += self._major_hits(4, self.center.finish())
        out += self._major_hits(5, self.walk.finish())
        return out

    def _lane_changes(self, r):
        events = cl.lane_change_events([self.prev, r] if self.prev is not None else [r])
        self.prev = r if r.get("lane") is not None else None
        out = []
        for row, p, mk in events:
            self.changes.append(row["t"])
            if p.get("off") or row.get("off"):
                continue
            if "solid" not in str(mk or ""):
                continue
            sec = self.ctx.secs.of(row["x"], row["y"])
            self.n_solid[sec] = self.n_solid.get(sec, 0) + 1
            out.append(Hit(row["t"], sec, 6, "major" if self.n_solid[sec] >= 2 else "minor",
                           f"t={row['t']:.1f} 실선({mk}) 넘어 차로변경"))
        return out

    def _release(self, ready):
        out = []
        while self.held and ready(self.held[0]):
            out += self._edge_hits(self.edge.update(self.held.popleft()))
        if self.held:
            oldest = self.held[0]["t"]
            while self.changes and self.changes[0] <= oldest - LC_WIN:
                self.changes.popleft()
        return out

    def _edge_hits(self, spans):
        out = []
        for t0, t1, _r0, span_rows in spans:
            first_by_sec = {}
            for r in span_rows:
                first_by_sec.setdefault(self.ctx.secs.of(r["x"], r["y"]), r)
            for sec in first_by_sec:
                self.n_edge[sec] = self.n_edge.get(sec, 0) + 1
                out.append(Hit(t0, sec, 3, "major" if self.n_edge[sec] >= 2 else "minor",
                               f"t={t0:.1f}~{t1:.1f} 차로 경계 물림"))
        return out

    def _major_hits(self, item, spans):
        return [Hit(t0, self.ctx.secs.of(r0["x"], r0["y"]), item, "major", f"t={t0:.1f}~{t1:.1f}")
                for t0, t1, r0, _ in spans]
```

이 코드가 원본과 같은 이유:
- 붙잡은 행 `h` 는 `r.t ≥ h.t + 2.5` 인 행 `r` 까지 들어온 뒤 놓는다. 그 뒤에 생길 사건은 `t > r.t ≥ h.t + 2.5` 라서 `|h.t - ct| < 2.5` 를 만족시킬 수 없다. 그래서 놓는 순간 `_edge_pred(h)` 는 원본의 `during_change` 와 같은 답을 낸다.
- 사건 버리기: 앞으로 판정할 행은 모두 `t ≥ oldest` 다. `ct ≤ oldest - 2.5` 인 사건은 그 행들의 창에 들 수 없다.

`vtd_rl/referee/core.py` 의 `default_judges`:
```python
def default_judges(use_map):
    from vtd_rl.referee.judges.contact import ContactJudge, CrosswalkStopJudge
    from vtd_rl.referee.judges.lane_geometry import LaneGeometryJudge
    from vtd_rl.referee.judges.pedestrian import PedestrianJudge
    from vtd_rl.referee.judges.speed import SpeedJudge
    from vtd_rl.referee.judges.traffic_light import TrafficLightJudge
    judges = [SpeedJudge]
    if use_map:
        judges.append(LaneGeometryJudge)
    return judges + [TrafficLightJudge, ContactJudge, PedestrianJudge, CrosswalkStopJudge]
```

- [ ] **Step 4: 통과 확인**

Run: `env -u PYTHONPATH .venv/bin/pytest tests/referee -q; echo rc=$?`
Expected: `rc=0`. Task 4 에서 `centerline_crosser` 의 `expect` 를 비웠다면 `test_차로_유지_판정은_2_5초_뒤에_낸다` 의 마지막 줄을 `expect` 가 남은 다른 차로 판(예: `sidewalk_rider` 의 항목 5)으로 바꾼다.

- [ ] **Step 5: 커밋**

```bash
git add vtd_rl/referee tests/referee/test_parity_lane_geometry.py
git commit -m "심판 — ③④⑤⑥ 차로 기하 판정(③ 은 2.5 초 뒤), 채점기와 일치"
```

---

### Task 10: ⑬ 차로변경 지시등 · ⑮ 리스폰 판정기

**Files:**
- Create: `vtd_rl/referee/judges/turn_signal.py`, `tests/referee/test_parity_turn_signal.py`, `tests/referee/test_turn_signal_online.py`
- Modify: `vtd_rl/referee/core.py` (`default_judges`)

**Interfaces:**
- Consumes: `Hit`, `Context`(`secs`), `compare`, `score_episode`, `Referee(..., judges=[...])`, `sf.SIG_SEC`, `sf.RESPAWN_JUMP`
- Produces: `TurnSignalJudge` (`items = (13,)`), `RespawnJudge` (`items = (15,)`, `Hit.t` = 튄 뒤 행의 t, `Hit.sec` = 뒤 행 위치의 구간)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/referee/test_parity_turn_signal.py`
```python
import pytest

from vtd_rl.referee.parity import compare
from vtd_rl.referee.scenarios import SCENARIOS, episode_rows


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_지시등_판정_일치(name):
    sc, rows = episode_rows(name)
    p = compare(sc.board, rows, items=(13,))
    assert p.ok, p.diff()
```

`tests/referee/test_turn_signal_online.py` — 합성 행으로 창·앵커 규칙을 촘촘히 본다:
```python
import collections
import copy
import random

from vtd_rl.referee.core import Referee
from vtd_rl.referee.judges.turn_signal import RespawnJudge, TurnSignalJudge
from vtd_rl.referee.oracle import score_episode
from vtd_rl.world.board import load_board, slice_board

H = {"name": "course_H", "route": "routes/HL_FMA_NEW_H.json", "lane": "routes/HL_FMA_NEW_H_lane.json"}
BOARD = slice_board(load_board(H), 0.0, 250.0, "H_0_250")


def synthetic(rng, n=700):
    """경로를 따라 가며 d_ego 가 가끔 2~4 m 씩 움직이고 지시등이 제멋대로 켜지는 행."""
    rows, s, d, target, sig = [], 0.0, 0.0, 0.0, 0
    for k in range(n):
        if rng.random() < 0.01:
            target = d + rng.choice([-1, 1]) * rng.uniform(2.0, 4.0)
        if rng.random() < 0.02:
            sig = rng.choice([0, 1, 2])
        d += max(-0.08, min(0.08, target - d))
        x, y, h = BOARD.route.point_at(s)
        rows.append(dict(t=round(k * 0.05, 2), x=x, y=y, h=h, v=8.0, tl=0, tid=0, reason="",
                         sig=sig, clr=99.9, d_ego=round(d, 2), objs="", cap_by=""))
        s += 0.35
    return rows


def run_referee(rows, judge):
    ref = Referee(BOARD, use_map=False, judges=[judge])
    for r in rows:
        ref.step(copy.deepcopy(r))
    ref.finish()
    return ref


def test_합성_행_지시등_판정_일치():
    rng = random.Random(7)
    fired = 0
    for _ in range(40):
        rows = synthetic(rng)
        want = collections.Counter(h for h in score_episode(rows, BOARD, use_map=False).hits if h[1] == 13)
        got = collections.Counter((h.sec, h.item, h.level) for h in run_referee(rows, TurnSignalJudge).hits)
        assert got == want
        fired += sum(want.values())
    assert fired > 0


def test_판정은_늦어도_8초_안에_낸다():
    rows = synthetic(random.Random(11), n=900)
    ref = Referee(BOARD, use_map=False, judges=[TurnSignalJudge])
    for r in rows:
        for h in ref.step(copy.deepcopy(r)):
            assert r["t"] - h.t <= 8.0 + 1e-9
    ref.finish()
    assert any(h.item == 13 for h in ref.hits)


def test_합성_순간이동은_채점기와_같이_센다():
    rows = synthetic(random.Random(5), n=300)
    for r in rows[100:]:
        r["y"] += 15.0
    for r in rows[200:]:
        r["y"] += 15.0
    want = score_episode(rows, BOARD, use_map=False).respawns
    ref = run_referee(rows, RespawnJudge)
    assert ref.respawns == want and sum(map(len, want.values())) == 2
```

- [ ] **Step 2: 실패 확인**

Run: `env -u PYTHONPATH .venv/bin/pytest tests/referee/test_parity_turn_signal.py tests/referee/test_turn_signal_online.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'vtd_rl.referee.judges.turn_signal'`

- [ ] **Step 3: 구현**

원본 `item_turn_signal` 의 앵커 루프를 먼저 읽는다. 온라인판은 같은 루프를 **행이 모자라면 멈췄다가 다음 행에서 이어 가는** 형태다.
- 출발점 8 m 안 행은 버퍼에 넣지 않는다(원본이 목록에서 뺀다).
- 앵커 `i` 에서 창을 넓히다가 (a) `|moved| ≥ 3.0` 이 되거나 (b) 다음 행이 창(8 초) 밖이면 그 앵커를 판정한다. 둘 다 아니고 행이 모자라면 기다린다. `finish()` 는 모자란 채로 판정한다(원본의 목록 끝).
- 뒤로 지시등을 재는 부분은 행마다 "같은 `sig` 가 이어진 첫 행의 t"(`run_start`)를 적어 두고 `t[k0] - run_start[k0]` 로 계산한다. 원본의 뒤로 훑기와 같은 값이다.
- 판정이 끝난 앵커 앞의 행은 버린다.

`vtd_rl/referee/judges/turn_signal.py`
```python
"""⑬ 차로변경 지시등 — score_fma.item_turn_signal 의 한 행 판(최대 8 초 늦게 낸다).
⑮ 리스폰 — score_fma.item_respawn 의 한 행 판(오프라인 세계에는 순간이동이 없어 합성 행으로 검증한다)."""
import math

from vtd_rl import rule_stack as rs
from vtd_rl.referee.core import Hit

sf = rs.score_fma
LANE_W, WIN = 3.0, 8.0      # item_turn_signal 안의 지역 상수
ONSET_D = 0.3               # item_turn_signal 안의 숫자 — 기동 개시로 볼 횡이동[m]
SPAWN_M = 8.0               # item_turn_signal 안의 숫자 — 출발점에서 이만큼은 뺀다[m]
RESPAWN_DT = 1.0            # item_respawn 안의 숫자 — 이 시간 안의 순간이동만[s]


class TurnSignalJudge:
    items = (13,)

    def __init__(self, ctx):
        self.ctx = ctx
        self.start = None
        self.rows = []              # (행, 같은 sig 가 이어진 첫 행의 t) — 앵커부터
        self.i = 0                  # 앵커(버퍼 안 번호)
        self.j = 0                  # 앵커 창에서 마지막으로 본 행
        self.moved = 0.0
        self.last_sig = None
        self.last_run_start = None

    def step(self, r):
        if self.start is None:
            self.start = (r["x"], r["y"])
        if math.hypot(r["x"] - self.start[0], r["y"] - self.start[1]) <= SPAWN_M:
            return []
        if r["sig"] != self.last_sig:
            self.last_sig, self.last_run_start = r["sig"], r["t"]
        self.rows.append((r, self.last_run_start))
        return self._advance(final=False)

    def finish(self):
        return self._advance(final=True)

    def _advance(self, final):
        out, rows = [], self.rows
        while self.i < len(rows):
            r0 = rows[self.i][0]
            while (abs(self.moved) < LANE_W and self.j + 1 < len(rows)
                   and rows[self.j + 1][0]["t"] - r0["t"] < WIN):
                self.j += 1
                self.moved = rows[self.j][0]["d_ego"] - r0["d_ego"]
            if abs(self.moved) >= LANE_W:
                out += self._judge(self.i, self.j, self.moved)
                self.i = self.j + 1
            elif self.j + 1 < len(rows) or final:
                self.i += 1
            else:
                break                                   # 창이 아직 열려 있다 — 다음 행을 기다린다
            self.j, self.moved = self.i, 0.0
        del rows[:self.i]
        self.j -= self.i
        self.i = 0
        return out

    def _judge(self, i, j, moved):
        rows = self.rows
        d0 = rows[i][0]["d_ego"]
        k0 = next((k for k in range(i, j + 1) if abs(rows[k][0]["d_ego"] - d0) > ONSET_D), i)
        r0, run_start = rows[k0]
        want = 1 if moved > 0 else 2                    # TS_LEFT / TS_RIGHT
        lead = r0["t"] - run_start if r0["sig"] == want else None
        if lead is not None and lead >= sf.SIG_SEC:
            return []
        return [Hit(r0["t"], self.ctx.secs.of(r0["x"], r0["y"]), 13, "minor",
                    f"t={r0['t']:.1f} 차로변경 선행점등 {0.0 if lead is None else lead:.1f}초 < {sf.SIG_SEC}초")]


class RespawnJudge:
    items = (15,)

    def __init__(self, ctx):
        self.ctx = ctx
        self.prev = None

    def step(self, r):
        a, self.prev = self.prev, r
        if a is None:
            return []
        d = math.hypot(r["x"] - a["x"], r["y"] - a["y"])
        if d > sf.RESPAWN_JUMP and r["t"] - a["t"] < RESPAWN_DT:
            return [Hit(r["t"], self.ctx.secs.of(r["x"], r["y"]), 15, "major",
                        f"t={r['t']:.1f} {d:.0f}m 순간이동")]
        return []

    def finish(self):
        return []
```

`_advance` 가 원본과 같은 이유:
- 원본 안쪽 루프는 `j+1 < n` 이고 창 안이면 `j` 를 늘리고 `moved` 를 다시 잰다. `|moved| ≥ 3` 이면 멈춘다. 위 안쪽 루프의 조건이 이것과 같다.
- 원본은 판정 뒤 `i = j` 로 옮기고 `i += 1` 을 한다. 판정이 없으면 `i += 1` 만 한다. 위 코드도 판정 뒤에는 `self.i = self.j + 1`, 판정이 없으면 `self.i += 1` 이다.
- 원본 `lead` 는 `k0` 에서 뒤로 `sig == want` 가 이어지는 마지막 행까지의 시간이다. 이것은 `t[k0] - (그 연속 구간의 첫 행 t)` 와 같다. `sig != want` 이면 `None` 이다.

`vtd_rl/referee/core.py` 의 `default_judges` 최종형 — `score_fma.main()` 의 호출 순서와 같게 둔다:
```python
def default_judges(use_map):
    from vtd_rl.referee.judges.contact import ContactJudge, CrosswalkStopJudge
    from vtd_rl.referee.judges.lane_geometry import LaneGeometryJudge
    from vtd_rl.referee.judges.pedestrian import PedestrianJudge
    from vtd_rl.referee.judges.speed import SpeedJudge
    from vtd_rl.referee.judges.traffic_light import TrafficLightJudge
    from vtd_rl.referee.judges.turn_signal import RespawnJudge, TurnSignalJudge
    judges = [SpeedJudge]
    if use_map:
        judges.append(LaneGeometryJudge)
    return judges + [TrafficLightJudge, ContactJudge, PedestrianJudge, CrosswalkStopJudge,
                     TurnSignalJudge, RespawnJudge]
```

- [ ] **Step 4: 통과 확인**

Run: `env -u PYTHONPATH .venv/bin/pytest tests/referee -q; echo rc=$?`
Expected: `rc=0`. `fired > 0` 이나 `any(h.item == 13 ...)` 에서만 실패하면 합성 행이 3 m 이동을 만들지 못한 것이다. `synthetic` 의 이동 폭이나 `n` 을 키운다(판정 규칙은 그대로).

- [ ] **Step 5: 커밋**

```bash
git add vtd_rl/referee tests/referee/test_parity_turn_signal.py tests/referee/test_turn_signal_online.py
git commit -m "심판 — ⑬ 차로변경 지시등(최대 8 초 뒤)·⑮ 리스폰, 채점기와 일치"
```

---

### Task 11: 전 항목 일치 검증과 성적표(M2a 완료 증거)

**Files:**
- Create: `tests/referee/test_parity_all.py`, `scripts/referee_parity_report.py`, `docs/reports/m2a-referee-parity.md`(스크립트가 만든다)
- Modify: `README.md` (심판 한 단락)

**Interfaces:**
- Consumes: `SCENARIOS`, `episode_rows`, `compare`, `Parity`(`want`, `got`, `respawns_want`, `ok`, `frames`, `referee_seconds`, `per_frame_us`), `run_episode`, `TeacherDriver`, `load_curriculum`, `rs.commit`

- [ ] **Step 1: 전 항목 일치 테스트 작성**

`tests/referee/test_parity_all.py`
```python
import os

import pytest

from vtd_rl.drivers.scripted import TeacherDriver
from vtd_rl.referee.parity import compare
from vtd_rl.referee.scenarios import SCENARIOS, episode_rows
from vtd_rl.rollout import run_episode
from vtd_rl.world.board import load_curriculum

STAGE1 = os.path.join(os.path.dirname(__file__), "..", "..", "curricula", "stage1.json")


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_대본_판_전_항목_일치(name):
    sc, rows = episode_rows(name)
    p = compare(sc.board, rows)
    assert p.ok, p.diff()


@pytest.mark.slow
@pytest.mark.parametrize("board", load_curriculum(STAGE1)[1], ids=lambda b: b.name)
def test_선생님_연습_코스_전_항목_일치(board):
    run = run_episode(board, TeacherDriver(board))
    p = compare(board, run.rows)
    assert p.ok, p.diff()
```

- [ ] **Step 2: 실행**

Run: `env -u PYTHONPATH .venv/bin/pytest tests/referee/test_parity_all.py -q; echo rc=$?`
Expected: `rc=0`. 실패하면 메시지의 항목·구간·등급을 보고 **해당 판정기**를 고친다. 대본·문턱은 바꾸지 않는다. 불일치가 ⑦⑨ 재접근(Global Constraints) 때문이면 멈추고 그 판·신호·시각을 보고한다.

- [ ] **Step 3: 성적표 스크립트 작성**

`scripts/referee_parity_report.py`
```python
"""M2a 완료 증거 — 대본 판·선생님 코스마다 채점기와 심판의 감점 호출, 일치 여부, 심판 프레임당 시간.

    env -u PYTHONPATH .venv/bin/python scripts/referee_parity_report.py
"""
import collections
import contextlib
import datetime
import os
import platform
import sys

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO)
from vtd_rl import rule_stack as rs  # noqa: E402
from vtd_rl.drivers.scripted import TeacherDriver  # noqa: E402
from vtd_rl.referee.parity import compare  # noqa: E402
from vtd_rl.referee.scenarios import SCENARIOS, episode_rows  # noqa: E402
from vtd_rl.rollout import run_episode  # noqa: E402
from vtd_rl.world.board import load_curriculum  # noqa: E402


def items_text(counter):
    by = collections.Counter()
    for (_sec, item, level), n in counter.items():
        by[(item, level)] += n
    return ", ".join(f"{i}{'M' if lv == 'major' else 'm'}×{n}" for (i, lv), n in sorted(by.items())) or "—"


def main():
    results = []
    with contextlib.redirect_stdout(sys.stderr):
        for name in sorted(SCENARIOS):
            sc, rows = episode_rows(name)
            results.append((name, sc.note, sorted(sc.expect), compare(sc.board, rows)))
        _, boards = load_curriculum(os.path.join(REPO, "curricula", "stage1.json"))
        for b in boards:
            run = run_episode(b, TeacherDriver(b))
            results.append((f"teacher_{b.name}", f"선생님 전 코스({run.result.outcome})", [],
                            compare(b, run.rows)))

    fired = {item for *_, p in results for (_sec, item, _lv) in p.want}
    if any(p.respawns_want for *_, p in results):
        fired.add(15)
    never = [i for i in range(1, 16) if i not in fired]
    all_ok = all(p.ok for *_, p in results)
    frames = sum(p.frames for *_, p in results)
    seconds = sum(p.referee_seconds for *_, p in results)
    lines = [
        "# M2a 성적표 — 온라인 심판과 대회 채점기(score_fma) 일치", "",
        f"- 날짜 {datetime.date.today().isoformat()} · 머신 `{platform.node()}` · 규칙 스택 `{rs.commit()[:7]}`",
        "- 일치: 한 판에서 `(구간, 항목, 등급)` 감점 호출 다중집합, 리스폰 사전, 구간별 최종 감점표가 모두 같다(구간 5개)",
        "- 표기: 항목번호 + M(중대)/m(경미) × 호출 횟수",
        f"- 채점기가 한 번이라도 일으킨 항목: {sorted(fired)}",
        f"- 한 번도 일어나지 않은 항목: {never or '없음'} (⑮ 는 tests/referee/test_turn_signal_online.py 합성 행으로 검증)",
        f"- 심판 평균 {seconds / max(frames, 1) * 1e6:.0f} µs/프레임(지도 판정 포함, {frames} 프레임)", "",
        "| 판 | 설명 | 기대 항목 | 채점기 | 심판 | 일치 | µs/프레임 |",
        "|---|---|---|---|---|---|---:|",
    ]
    for name, note, expect, p in results:
        lines.append(f"| {name} | {note} | {expect or '—'} | {items_text(p.want)} | {items_text(p.got)} | "
                     f"{'✅' if p.ok else '❌'} | {p.per_frame_us:.0f} |")
    lines += ["", f"**전체 일치: {'예' if all_ok else '아니오'}**"]
    out = os.path.join(REPO, "docs", "reports", "m2a-referee-parity.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(out)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 성적표 만들기**

Run: `env -u PYTHONPATH .venv/bin/python scripts/referee_parity_report.py; echo rc=$?`
Expected: `docs/reports/m2a-referee-parity.md` 가 만들어지고 `rc=0`(전체 일치).

- [ ] **Step 5: README 한 단락**

`README.md` 의 `## 테스트` 절과 `## M1 성적표 다시 만들기` 절 사이에 추가한다:
```markdown
## 온라인 심판
`vtd_rl/referee` 는 주행 한 판을 한 프레임씩 받아 대회 채점기(`score_fma.py`)와 같은 감점 판정을 낸다.
채점기가 뒤 프레임을 보는 항목(녹색 정차, 보행자, 차로 유지, 차로변경 지시등)은 그만큼, 최대 8 초 늦게 낸다.
일치 검증 결과: [docs/reports/m2a-referee-parity.md](docs/reports/m2a-referee-parity.md)

    env -u PYTHONPATH .venv/bin/python scripts/referee_parity_report.py
```

- [ ] **Step 6: 전체 테스트**

Run: `env -u PYTHONPATH .venv/bin/pytest -q; echo rc=$?`
Expected: `rc=0`

- [ ] **Step 7: 커밋**

```bash
git add tests/referee/test_parity_all.py scripts/referee_parity_report.py docs/reports/m2a-referee-parity.md README.md
git commit -m "M2a 증거 — 대본 판·선생님 코스 전 항목 채점기 일치, 심판 프레임당 시간"
```
