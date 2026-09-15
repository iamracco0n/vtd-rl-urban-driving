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


# 대본 판·선생님 코스에서 안 일어나는 항목을 대신 검증하는 합성 행 테스트
SYNTHETIC_TESTS = {6: "tests/referee/test_lane_geometry_synthetic.py",
                   15: "tests/referee/test_turn_signal_online.py"}


def circled(item):
    return chr(0x2460 + item - 1)


def never_text(never):
    if not never:
        return "없음"
    cited = [f"{circled(i)} 는 {SYNTHETIC_TESTS[i]}" for i in never if i in SYNTHETIC_TESTS]
    return f"{never}" + (f" ({', '.join(cited)} 합성 행으로 검증)" if cited else "")


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
        f"- 한 번도 일어나지 않은 항목: {never_text(never)}",
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
