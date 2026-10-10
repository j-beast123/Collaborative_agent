"""Render a portable, read-only review of training dialogues and rubric scores.

The caller owns snapshot extraction.  This module only serializes the supplied
snapshot into one HTML document; it does not read or modify experiment files.
"""

from __future__ import annotations

import json
from typing import Any


def render_review(data: dict[str, Any]) -> str:
    """Return self-contained HTML with every supplied original text preserved.

    JSON is embedded as inert script content. Escaping HTML-significant code
    points prevents a dialogue containing ``</script>`` from leaving that data
    block. All dynamic visible content is independently HTML-escaped by the UI.
    """
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    for character, escaped in (
        ("&", "\\u0026"),
        ("<", "\\u003c"),
        (">", "\\u003e"),
        ("\u2028", "\\u2028"),
        ("\u2029", "\\u2029"),
    ):
        payload = payload.replace(character, escaped)
    return _HTML.replace("__REVIEW_DATA__", payload, 1)


_HTML = r'''<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>학습 대화 · 루브릭 · 실제 채점 검토</title>
<style>
:root{--bg:#f5f7fa;--paper:#fff;--ink:#17243a;--muted:#64748b;--line:#dce3eb;--blue:#2858a5;--blue-soft:#edf3ff;--green:#137754;--green-soft:#eaf8f0;--red:#ac3434;--red-soft:#fff0ee;--amber:#8a5b06;--amber-soft:#fff8e5;--shadow:0 2px 5px #17243a05}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.65 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}button,input,select{font:inherit}button,select{cursor:pointer}button{color:inherit;background:var(--paper);border:1px solid var(--line);border-radius:7px;padding:6px 11px}button:hover{border-color:#93abc9;background:#f7faff}button:disabled{cursor:default;opacity:.4}button:focus-visible,input:focus-visible,select:focus-visible,summary:focus-visible{outline:3px solid #84aef5;outline-offset:3px}a{color:var(--blue)}h1,h2,h3,p{margin:0}h1{font-size:25px;letter-spacing:-.7px;line-height:1.35}h2{font-size:19px;letter-spacing:-.4px}h3{font-size:16px}pre{font:inherit;white-space:pre-wrap;overflow-wrap:anywhere;word-break:normal;margin:5px 0 0}small,.muted{color:var(--muted)}code,.mono{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12px;overflow-wrap:anywhere}header{max-width:1600px;margin:auto;padding:28px 30px 20px}.eyebrow{font-size:11px;font-weight:750;letter-spacing:1.5px;color:var(--blue);margin-bottom:7px}.header-row{display:flex;align-items:start;justify-content:space-between;gap:16px}.snapshot{margin-top:9px;color:var(--muted);font-size:12px;overflow-wrap:anywhere}.snapshot span{display:inline-block;margin-right:18px}.kpis{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin-top:20px}.kpi{background:var(--paper);border:1px solid var(--line);border-radius:10px;padding:12px 16px;box-shadow:var(--shadow)}.kpi strong{font-size:24px;letter-spacing:-.8px;display:block;line-height:1.3}.kpi span{color:var(--muted);font-size:12px}.layout{max-width:1600px;margin:auto;padding:0 30px 40px;display:grid;grid-template-columns:240px minmax(0,1fr);gap:22px;align-items:start}aside{position:sticky;top:16px;border:1px solid var(--line);border-radius:11px;background:var(--paper);overflow:hidden}.side-heading{padding:14px 14px 10px;font-size:12px;font-weight:750}.case-search{padding:0 12px 12px}.case-search input{width:100%;min-width:0}.case-list{max-height:calc(100vh - 220px);min-height:80px;overflow:auto;padding:0 7px 7px}.case-button{width:100%;padding:10px;margin:2px 0;text-align:left;border-color:transparent;border-radius:7px}.case-button.active{background:var(--blue-soft);border-color:#c8d9f3;color:#1c468a}.case-top{display:flex;justify-content:space-between;gap:6px;align-items:center}.case-number{font-weight:750}.case-id{font-size:11px;overflow-wrap:anywhere;margin:3px 0}.case-meta{font-size:11px;color:var(--muted)}main{min-width:0}.panel{border:1px solid var(--line);border-radius:11px;background:var(--paper);box-shadow:var(--shadow);margin-bottom:15px}.panel-header{padding:17px 20px;border-bottom:1px solid var(--line)}.panel-body{padding:17px 20px}.case-title{display:flex;align-items:center;flex-wrap:wrap;gap:10px}.case-intro{margin:13px 0 0}.field{margin:13px 0}.field:first-child{margin-top:0}.field-label{font-size:11px;line-height:1.5;font-weight:750;letter-spacing:.3px;color:var(--muted);display:block;margin-bottom:5px}.original{background:#f8fafc;border:1px solid #e8edf3;border-radius:7px;padding:11px 13px}.options{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin-top:10px}.option{display:flex;gap:9px;align-items:baseline;border:1px solid var(--line);border-radius:7px;padding:9px 11px}.option strong{color:var(--blue);min-width:16px}.option pre{margin:0}.notes{font-size:12px;color:var(--muted);margin-top:11px}.tabs{display:flex;gap:5px;padding:4px;background:#eaf0f6;border-radius:9px;margin:19px 0 12px}.tab{flex:1;border:none;background:transparent;padding:9px 10px;font-weight:650;color:var(--muted)}.tab.active{background:white;color:var(--blue);box-shadow:0 1px 4px #21354e12}.tab-count{font-size:11px;font-weight:500;margin-left:5px}.filters{display:flex;align-items:end;gap:9px;flex-wrap:wrap;margin:12px 0}.filter-label{display:flex;flex-direction:column;gap:4px;min-width:120px;font-size:11px;font-weight:650;color:var(--muted)}.search-label{flex:1;min-width:200px}input,select{border:1px solid var(--line);border-radius:7px;background:white;color:var(--ink);padding:8px 10px;min-height:38px;font-size:13px}select{max-width:280px}.result-info{display:flex;align-items:center;justify-content:space-between;gap:10px;margin:12px 0;color:var(--muted);font-size:12px}.view-hint{font-size:12px;color:var(--muted);line-height:1.7;margin:8px 0 14px}.record{border:1px solid var(--line);border-radius:10px;background:white;margin-bottom:12px;box-shadow:var(--shadow);overflow:hidden}.record:target,.record.focused{outline:2px solid #80a7e8;outline-offset:2px}.record>summary{padding:14px 17px;cursor:pointer;list-style:none}.record>summary::-webkit-details-marker{display:none}.record>summary:before{content:'›';float:left;font-size:22px;line-height:1.2;color:#8796ab;margin-right:10px}.record[open]>summary:before{transform:rotate(90deg)}.record[open]>summary{border-bottom:1px solid var(--line)}.record-title{display:flex;gap:9px;align-items:center;flex-wrap:wrap;font-weight:700;line-height:1.5}.record-meta{color:var(--muted);font-size:11px;margin:6px 0 0 21px;overflow-wrap:anywhere}.record-content{padding:17px}.badge{display:inline-flex;align-items:center;border-radius:5px;background:#f0f3f7;color:#536176;border:1px solid #e1e7ef;font-size:11px;font-weight:650;line-height:1.45;padding:3px 7px;vertical-align:middle;white-space:normal;overflow-wrap:anywhere}.badge.success,.badge.keep,.badge.s2{color:var(--green);background:var(--green-soft);border-color:#c6e8d7}.badge.failure,.badge.drop,.badge.s0{color:var(--red);background:var(--red-soft);border-color:#f0d0ca}.badge.s1{color:var(--amber);background:var(--amber-soft);border-color:#eddfb8}.badge.unknown,.badge.sNA,.badge.sUNSCORABLE{color:#59677b;background:#eef1f5;border-color:#dce2e9}.badge.blue{background:var(--blue-soft);color:var(--blue);border-color:#cddcf3}.score-tags{display:flex;flex-wrap:wrap;gap:6px;margin-top:9px}.score-tag{padding:4px 7px;font-size:11px;text-align:left}.score-tag .badge{margin-right:5px;padding:1px 5px}.dialogue{display:flex;flex-direction:column;gap:10px}.utterance{border:1px solid var(--line);border-left:3px solid #c3ccd8;border-radius:7px;padding:12px 14px;background:#fff}.utterance.agent{border-left-color:#739fe0;background:#f8fbff}.utterance.target{border:2px solid #6796dc;border-left-width:5px;background:#edf4ff}.utterance-head{display:flex;align-items:center;gap:9px;flex-wrap:wrap;margin-bottom:7px}.speaker{font-size:12px;font-weight:750}.turn-meta{font-size:11px;color:var(--muted)}.node-id{margin-left:auto;font-size:10px}.target-marker{color:var(--blue);font-size:11px;font-weight:750}.subdetails{margin:14px 0 0;border-top:1px solid #e7ecf2;padding-top:11px}.subdetails>summary{cursor:pointer;color:var(--blue);font-size:12px;font-weight:650}.subdetails-body{margin-top:12px}.two-col{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.conditions{padding:13px;background:#f4f7fc;border:1px solid #e0e8f4;border-radius:8px;margin-bottom:12px}.anchors{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px}.anchor{border:1px solid var(--line);border-radius:8px;padding:11px;background:#fbfcfe}.anchor.matching{border:2px solid #78a0dc;background:#f0f6ff;padding:10px}.anchor pre{font-size:12px}.anchor-top{display:flex;align-items:center;gap:7px;flex-wrap:wrap}.anchor-top span:last-child{color:var(--muted);font-size:10px}.decision-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:9px;margin:14px 0}.decision{padding:12px;border:1px solid var(--line);border-radius:8px}.decision .field-label{margin-bottom:7px}.decision p{font-size:12px;margin-top:7px;color:var(--muted)}.decision-score{font-size:17px;font-weight:700;overflow-wrap:anywhere}.score-lead{display:flex;align-items:center;flex-wrap:wrap;gap:11px;margin-bottom:13px}.score-large{font-size:16px;padding:6px 11px}.rationale{border-left:3px solid #abbad0;background:#f7f9fc;padding:12px 14px;margin-top:14px;border-radius:0 7px 7px 0}.rationale pre{font-size:13px}.text-button{color:var(--blue);border:none;background:transparent;padding:2px 0;text-decoration:underline;text-underline-offset:3px;font-size:12px}.inline-links{display:flex;flex-wrap:wrap;gap:12px;margin:10px 0}.empty{border:1px dashed #c8d3e0;border-radius:10px;padding:30px;background:white;color:var(--muted);text-align:center}.pager{display:flex;align-items:center;justify-content:center;flex-wrap:wrap;gap:10px;margin:20px 0}.pager span{font-size:12px;color:var(--muted)}.raw{font:11px/1.65 ui-monospace,SFMono-Regular,Consolas,monospace;color:#47566c;background:#f7f9fb;border:1px solid #e5eaf0;border-radius:7px;padding:12px}.fact-list{display:grid;grid-template-columns:140px minmax(0,1fr);gap:8px 12px;margin:0}.fact-list dt{color:var(--muted);font-size:12px}.fact-list dd{margin:0;min-width:0;font-size:12px;overflow-wrap:anywhere}.fact-list pre{margin:0}.legend{font-size:11px;color:var(--muted);margin:10px 0 0}.alert{background:#fff8e9;border:1px solid #ebdbba;border-radius:7px;padding:10px 13px;font-size:12px;color:#795924;margin:12px 0}.hidden{display:none!important}.footer{font-size:11px;color:var(--muted);text-align:center;margin-top:25px}noscript{display:block;padding:30px;color:#9a3333}
@media(min-width:1300px){.layout{grid-template-columns:260px minmax(0,1fr)}}
@media(max-width:900px){header{padding:22px 18px 16px}.layout{padding:0 18px 28px;grid-template-columns:185px minmax(0,1fr);gap:14px}.kpis{gap:8px}.kpi{padding:10px 12px}.kpi strong{font-size:21px}.anchors{grid-template-columns:1fr}.two-col{grid-template-columns:1fr}.record-content{padding:14px}}
@media(max-width:650px){header{padding:20px 13px 14px}h1{font-size:22px}.header-row{display:block}.header-row>button{margin-top:10px}.kpis{grid-template-columns:repeat(3,minmax(0,1fr))}.layout{display:block;padding:0 13px 24px}aside{position:static;margin-bottom:15px}.side-heading{padding:10px 12px 6px}.case-search{padding-bottom:8px}.case-list{max-height:170px;display:flex;gap:4px;overflow:auto}.case-button{min-width:155px;max-width:200px}.tabs{gap:0}.tab{font-size:12px;padding:8px 3px}.tab-count{margin-left:3px}.panel-header,.panel-body{padding:14px}.options,.decision-grid{grid-template-columns:1fr}.filter-label{min-width:100px}.search-label{flex-basis:100%}.filters select{width:100%}.record-title{font-size:13px}.fact-list{grid-template-columns:100px minmax(0,1fr)}.node-id{margin-left:0}.snapshot span{display:block}.score-large{font-size:14px}}
@media print{body{background:white}header{padding:0 0 15px}.layout{display:block;padding:0}aside,.filters,.tabs,.pager,.header-row>button,.result-info button{display:none}.kpi,.record,.panel{box-shadow:none}.record{break-inside:avoid}.utterance{break-inside:avoid}.record>summary{list-style:none}.record>summary:before{display:none}.score-tag{border:0}.view-hint{margin-top:10px}.footer{display:none}}
</style>
</head>
<body>
<header>
  <div class="header-row"><div><div class="eyebrow">TRAINING REVIEW · READ-ONLY SNAPSHOT</div><h1>학습 대화 · 루브릭 · 실제 채점</h1><div class="snapshot" id="snapshot"></div></div><button id="download" title="페이지에 포함된 전체 스냅샷을 JSON으로 저장">전체 데이터 JSON 저장</button></div>
  <div class="kpis" id="kpis"></div>
  <div id="snapshot-notes"></div>
</header>
<div class="layout">
  <aside aria-label="문제 선택"><div class="side-heading">문제 선택</div><div class="case-search"><input id="case-query" type="search" placeholder="번호 · ID · 문제 검색" aria-label="문제 검색"></div><nav class="case-list" id="case-list"></nav></aside>
  <main id="main"><section class="panel" id="case-context" aria-label="원본 문제"></section>
    <div class="tabs" role="tablist" aria-label="검토 항목"><button class="tab" id="tab-trajectories" role="tab" data-view="trajectories" aria-controls="rows">원본 대화 <span class="tab-count" id="count-trajectories"></span></button><button class="tab" id="tab-rubrics" role="tab" data-view="rubrics" aria-controls="rows">루브릭 <span class="tab-count" id="count-rubrics"></span></button><button class="tab" id="tab-scores" role="tab" data-view="scores" aria-controls="rows">응답별 채점 <span class="tab-count" id="count-scores"></span></button></div>
    <div class="filters"><label class="filter-label search-label">현재 문제 안에서 검색<input id="query" type="search" placeholder="원문 · 루브릭 · 판단 근거 · ID"></label><label class="filter-label">실제 응답 점수<select id="score-filter"><option value="all">모든 점수</option><option value="0">0점</option><option value="1">1점</option><option value="2">2점</option><option value="NA">NA · 적용 안 됨</option><option value="UNSCORABLE">UNSCORABLE · 채점 불가</option></select></label><label class="filter-label">상태<select id="status-filter"><option value="all">모든 상태</option></select></label><button id="reset">필터 초기화</button></div>
    <p class="view-hint" id="view-hint"></p>
    <div class="result-info"><span id="result-count" role="status" aria-live="polite"></span><button id="expand-visible">현재 페이지 펼치기</button></div>
    <div id="rows" role="tabpanel" aria-labelledby="tab-trajectories"></div><div class="pager" id="pager"></div>
    <p class="footer">스냅샷에 저장된 원문을 표시합니다. 검색·필터·탭 전환은 실험 데이터와 실행 상태를 변경하지 않습니다.</p>
  </main>
</div>
<noscript>이 페이지의 대화와 채점을 탐색하려면 JavaScript를 활성화하세요. 전체 원본 데이터는 이 HTML 파일 내부의 review-data 블록에 포함되어 있습니다.</noscript>
<script id="review-data" type="application/json">__REVIEW_DATA__</script>
<script>
'use strict';
const DATA = JSON.parse(document.getElementById('review-data').textContent);
const CASES = Array.isArray(DATA.cases) ? DATA.cases : [];
const SUMMARY = DATA.summary || {};
const PAGE_SIZE = 20;
const $ = id => document.getElementById(id);
const E = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const text = value => value == null ? '' : (typeof value === 'string' ? value : JSON.stringify(value, null, 2));
const list = value => Array.isArray(value) ? value : [];
const has = value => value !== null && value !== undefined && value !== '';
const num = value => Number(value).toLocaleString('ko-KR');
const statusText = value => has(value) ? String(value) : '상태 미기록';
const badge = (value, style = '') => `<span class="badge ${E(style)}">${E(value)}</span>`;
const scoreBadge = (value, large = false) => badge(['0','1','2'].includes(String(value)) ? `${value}점` : (value || '미기록'), `s${['0','1','2','NA','UNSCORABLE'].includes(String(value)) ? value : 'UNSCORABLE'}${large ? ' score-large' : ''}`);
const block = (label, value, cls = '') => `<div class="field"><span class="field-label">${E(label)}</span><pre class="${E(cls)}">${E(has(value) ? text(value) : '기록 없음')}</pre></div>`;
const raw = (label, value) => has(value) ? `<details class="subdetails"><summary>${E(label)}</summary><div class="subdetails-body"><pre class="raw">${E(text(value))}</pre></div></details>` : '';
const extra = (label, content, open = false) => `<details class="subdetails"${open ? ' open' : ''}><summary>${E(label)}</summary><div class="subdetails-body">${content}</div></details>`;
const state = {caseIndex:0, view:'trajectories', query:'', score:'all', status:'all', page:0, focus:null};
let rubricIndex = new Map(), trajectoryIndex = new Map(), nodeScores = new Map();
const searchCache = new WeakMap();
function searchable(item) {
  if (!searchCache.has(item)) searchCache.set(item, JSON.stringify(item).toLocaleLowerCase());
  return searchCache.get(item);
}
function currentCase() { return CASES[state.caseIndex] || {}; }
function caseItems(view = state.view) { return list(currentCase()[view]); }
function fieldLabel(key) {
  const labels = {user_side_context:'User side · 사용자가 제공한 맥락',agent_side_state:'Agent side · 응답 직전 상태',user_side:'User side · 사용자가 제공한 맥락',agent_side:'Agent side · 응답 직전 상태',behavior:'행동 기준',rationale:'판단 근거',reason:'이유',feedback:'피드백',helpful:'도움이 된 부분',unresolved:'해결되지 않은 부분',final_option:'최종 선택',outcome:'결과',requested_level:'요청한 개입 점수',requested_score:'요청한 개입 점수',source_score:'원본 점수',witness_found:'Witness 확인',retained:'유지 여부',status:'상태',reasoning:'판단 근거',evidence:'근거',matched:'일치 여부',match:'적용 여부'};
  return labels[key] || key.replace(/_/g, ' ');
}
function objectFields(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return `<pre>${E(text(value) || '기록 없음')}</pre>`;
  return Object.entries(value).filter(([key]) => !/prompt/i.test(key)).map(([key,val]) => block(fieldLabel(key),val)).join('') || '<p class="muted">기록 없음</p>';
}
function reindex() {
  const c = currentCase();
  rubricIndex = new Map(list(c.rubrics).map(r => [String(r.rubric_id), r]));
  trajectoryIndex = new Map(list(c.trajectories).map(t => [String(t.trajectory_id), t]));
  nodeScores = new Map();
  list(c.scores).forEach((s,index) => {
    const key = String(s.response_node_id);
    if (!nodeScores.has(key)) nodeScores.set(key, []);
    nodeScores.get(key).push({score:s,index});
  });
}
function scoreLinks(nodeId) {
  if (!has(nodeId)) return '';
  const scores = nodeScores.get(String(nodeId)) || [];
  if (!scores.length) return '';
  return `<div class="score-tags" aria-label="이 응답의 실제 채점">${scores.map(({score:s,index}) => `<button class="score-tag" data-go-score="${index}" title="실제 채점과 판단 근거 보기">${scoreBadge(s.score_label)}${E(s.rubric_id)}</button>`).join('')}</div>`;
}
function dialogue(items, targetId = null, attachScores = true) {
  if (!list(items).length) return '<p class="muted">저장된 대화가 없습니다.</p>';
  let ordinal = 0;
  return `<div class="dialogue">${items.map(d => {
    const agent = /^(agent|assistant)$/i.test(String(d.speaker || ''));
    if (agent) ordinal++;
    const turn = d.turn ?? d.turn_id ?? d.round;
    const target = has(targetId) && agent && (String(d.node_id ?? '') === String(targetId) || (/^\d+$/.test(String(targetId)) && String(turn ?? ordinal) === String(targetId)));
    return `<section class="utterance${agent ? ' agent' : ''}${target ? ' target' : ''}"><div class="utterance-head"><span class="speaker">${E(d.speaker || 'speaker 미기록')}</span>${has(turn) ? `<span class="turn-meta">Turn ${E(turn)}</span>` : ''}${has(d.stage) ? `<span class="turn-meta">${E(d.stage)}</span>` : ''}${target ? '<span class="target-marker">선택된 원본 Agent 응답</span>' : ''}${has(d.node_id) ? `<code class="node-id">${E(d.node_id)}</code>` : ''}</div><pre>${E(text(d.text))}</pre>${attachScores ? scoreLinks(d.node_id) : ''}</section>`;
  }).join('')}</div>`;
}
function conditions(r) {
  const a = r.applicability || {};
  if (typeof a !== 'object' || Array.isArray(a)) return `<div class="conditions">${block('적용 조건', a)}</div>`;
  const user = a.user_side_context ?? a.user_side;
  const agent = a.agent_side_state ?? a.agent_side;
  const others = Object.fromEntries(Object.entries(a).filter(([k]) => !['user_side_context','user_side','agent_side_state','agent_side'].includes(k)));
  return `<div class="conditions"><div class="two-col">${block('User side · 사용자가 제공한 맥락',user)}${block('Agent side · 응답 직전 상태',agent)}</div>${Object.keys(others).length ? objectFields(others) : ''}</div>`;
}
function anchors(r, selected = null) {
  return `<div class="anchors">${['0','1','2'].map(level => {
    const anchor = (r.scores || {})[level];
    const behavior = typeof anchor === 'object' && anchor !== null ? (anchor.behavior ?? anchor) : anchor;
    const more = anchor && typeof anchor === 'object' ? Object.fromEntries(Object.entries(anchor).filter(([k]) => k !== 'behavior')) : {};
    return `<div class="anchor${String(selected) === level ? ' matching' : ''}"><div class="anchor-top">${scoreBadge(level)}<span>${String(selected) === level ? '실제 채점 기준' : '행동 기준'}</span></div><pre>${E(has(behavior) ? text(behavior) : '기록 없음')}</pre>${typeof behavior === 'string' && Object.keys(more).length ? raw('추가 기준 원문',more) : ''}</div>`;
  }).join('')}</div>`;
}
function outcomeBadge(value) {
  const label = {success:'SUCCESS · 성공',failure:'FAILURE · 실패',unknown:'UNKNOWN · 미확정'}[value] || statusText(value);
  return badge(label,['success','failure','unknown'].includes(value) ? value : 'unknown');
}
function trajectoryCard(t, index, open) {
  const related = caseItems('rubrics').filter(r => String(r.source_trajectory_id) === String(t.trajectory_id));
  return `<details class="record" id="trajectory-${index}"${open ? ' open' : ''}><summary><div class="record-title"><span>대화 ${index+1}</span>${outcomeBadge(t.outcome)}${has(t.final_option) ? badge(`최종 선택 ${t.final_option}`) : ''}${badge(`${list(t.dialogue).length}개 발화`)}</div><div class="record-meta">${E(t.trajectory_id)} · 출처 루브릭 ${related.length}개</div></summary><div class="record-content">${dialogue(t.dialogue)}${has(t.reflection) ? extra('원본 Reflection / 피드백',objectFields(t.reflection)) : ''}${related.length ? extra('이 대화에서 생성한 루브릭',related.map(r => `<div class="inline-links"><button class="text-button" data-go-rubric="${E(r.rubric_id)}">${E(r.name || r.rubric_id)} · ${E(r.rubric_id)}</button>${badge(statusText(r.status),String(r.status || '').toLowerCase().includes('keep') ? 'keep' : '')}</div>`).join('')) : ''}</div></details>`;
}
function sourceScores(r) {
  const gate = r.gate || {};
  const anchor = gate.source_anchor;
  const a = r.analysis || {};
  const generator = r.generator_source_score;
  const independent = r.source_independent_gate_score;
  const requested = a.requested_level ?? a.requested_score ?? a.intervention_requested_score ?? (r.witness || {}).requested_level;
  return `<div class="decision-grid"><div class="decision"><span class="field-label">원본 응답 · Generator 자체 점수</span><div class="decision-score">${has(generator) ? scoreBadge(generator) : '미기록'}</div><p>루브릭 생성 측에 기록된 원본 응답 점수</p></div><div class="decision"><span class="field-label">원본 응답 · 독립 Gate 채점</span><div class="decision-score">${has(independent) ? scoreBadge(independent) : '미기록'}</div><p>독립 평가에 저장된 원본 응답 점수</p></div>${has(requested) ? `<div class="decision"><span class="field-label">개입 요청 점수</span><div class="decision-score">${scoreBadge(requested)}</div><p>개입에 요청한 목표값이며 실제 응답 채점은 별도 표시</p></div>` : ''}</div>${has(anchor) ? extra('Gate의 원본 응답 anchor',objectFields(anchor)) : ''}`;
}
function rubricCard(r, index, open) {
  const gate = r.gate || {};
  const source = trajectoryIndex.get(String(r.source_trajectory_id));
  const gateLabel = gate.passed === true ? 'PASS' : gate.passed === false ? 'FAIL' : '미기록 / 대기';
  const status = statusText(r.status);
  const problems = list(r.problems);
  const keepStyle = /keep|retained/i.test(status) ? 'keep' : (/drop|reject/i.test(status) ? 'drop' : '');
  const relatedScores = caseItems('scores').filter(s => String(s.rubric_id) === String(r.rubric_id));
  return `<details class="record" id="rubric-${index}"${open ? ' open' : ''}><summary><div class="record-title"><span>${E(r.name || '이름 미기록')}</span>${badge(status,keepStyle)}${badge(`Gate ${gateLabel}`,gate.passed === true ? 'success' : gate.passed === false ? 'failure' : 'unknown')}</div><div class="record-meta">${E(r.rubric_id)} · 출처 ${E(r.source_trajectory_id)} · 선택 턴 ${E(r.selected_agent_turn_id ?? '미기록')}</div></summary><div class="record-content">${conditions(r)}${has(r.required_behavior) ? block('Required behavior · 요구 행동',r.required_behavior,'original') : ''}${anchors(r)}<div class="decision-grid"><div class="decision"><span class="field-label">생성 후 독립 Gate 판정</span>${badge(gateLabel,gate.passed === true ? 'success' : gate.passed === false ? 'failure' : 'unknown')}${has(gate.reasons) ? block('Gate 판단 이유',gate.reasons) : ''}</div><div class="decision"><span class="field-label">KEEP / 루브릭 선택 상태</span>${badge(status,keepStyle)}<p>Gate 판정과 별개로 기록된 루브릭 선택 결과</p>${has(r.source_outcome) ? `<p>출처 결과 ${outcomeBadge(r.source_outcome)}</p>` : ''}</div></div>${problems.length ? `<div class="alert">${block('검증 중 기록된 문제',problems)}</div>` : ''}${sourceScores(r)}${has(r.intervention_instruction) ? extra('저장된 개입 지시 원문',`<pre class="original">${E(text(r.intervention_instruction))}</pre>`) : ''}${source ? extra('출처 원본 대화 · 선택된 Agent 응답 강조',dialogue(source.dialogue,r.selected_agent_turn_id)+(has(source.reflection) ? extra('출처 Reflection / 피드백',objectFields(source.reflection)) : '')) : `<p class="notes">출처 대화 ${E(r.source_trajectory_id || '')}가 이 스냅샷에 없습니다.</p>`}${has(r.witness) ? extra('Witness / 개입 결과 기록',objectFields(r.witness)) : ''}${raw('독립 Gate 평가 원본 JSON',gate.assessment)}${raw('루브릭 분석 원본 JSON',r.analysis)}${relatedScores.length ? `<div class="inline-links"><button class="text-button" data-rubric-scores="${E(r.rubric_id)}">이 루브릭으로 채점한 응답 ${relatedScores.length}건 보기</button></div>` : '<p class="notes">이 루브릭으로 저장된 실제 응답 채점이 없습니다.</p>'}</div></details>`;
}
function scoreCard(s, index, open) {
  const r = rubricIndex.get(String(s.rubric_id));
  const applicability = s.applicable === true ? '적용됨' : s.applicable === false ? '적용 안 됨' : '적용 여부 미기록';
  const trajectories = list(s.containing_trajectories);
  return `<details class="record" id="score-${index}"${open ? ' open' : ''}><summary><div class="record-title">${scoreBadge(s.score_label)}<span>${E(r?.name || s.rubric_id || '루브릭 미기록')}</span>${badge(statusText(s.scoring_status))}</div><div class="record-meta">${E(s.response_node_id)} · Agent ${E(s.agent_turn ?? '?')}턴 · ${E(s.rubric_id)}</div></summary><div class="record-content"><div class="score-lead">${scoreBadge(s.score_label,true)}${badge(applicability,s.applicable === true ? 'blue' : 'unknown')}<span class="muted">정규화 점수: <strong>${has(s.normalized_score) ? E(s.normalized_score) : '없음'}</strong></span></div>${block('채점 대상 Agent 응답 · 원문',s.response,'original')}<div class="rationale">${block('이 실제 점수의 판단 근거 · 저장된 원문',s.rationale || '별도의 판단 근거가 저장되지 않았습니다.')}</div>${extra(`응답 직전 전체 대화 (${list(s.prefix).length}개 발화)`,dialogue(s.prefix,null,false))}${r ? `<div class="field"><span class="field-label">적용한 루브릭 · ${E(r.rubric_id)}</span>${conditions(r)}${has(r.required_behavior) ? block('Required behavior · 요구 행동',r.required_behavior,'original') : ''}${anchors(r,s.score_label)}<div class="inline-links"><button class="text-button" data-go-rubric="${E(r.rubric_id)}">루브릭 출처 · Gate · 선택 결과 보기</button></div></div>` : `<div class="alert">루브릭 ${E(s.rubric_id)}의 원문이 이 스냅샷에 없습니다.</div>`}${has(s.evidence) ? raw('실제 채점 근거 / 원본 평가 기록',s.evidence) : ''}${trajectories.length ? extra(`이 응답을 포함하는 원본 대화 (${trajectories.length}개)`,trajectories.map(id => {
    const trajectoryId = typeof id === 'object' ? (id.trajectory_id ?? id.leaf_id ?? text(id)) : id;
    const t = trajectoryIndex.get(String(trajectoryId));
    return `<div class="inline-links">${t ? `<button class="text-button" data-go-trajectory="${E(trajectoryId)}">${E(trajectoryId)}</button>${outcomeBadge(t.outcome)}` : `<span class="mono">${E(trajectoryId)}</span>`}</div>`;
  }).join('')) : ''}</div></details>`;
}
function renderHeader() {
  $('snapshot').innerHTML = `<span>스냅샷 ${E(SUMMARY.snapshot_kst || '시각 미기록')}</span><span>원본 실행 <code>${E(SUMMARY.source_run || '경로 미기록')}</code></span>`;
  const totals = CASES.reduce((a,c) => {a.trajectories += list(c.trajectories).length;a.rubrics += list(c.rubrics).length;a.scores += list(c.scores).length;list(c.scores).forEach(s => a.responses.add(`${c.case_id}\u0000${s.response_node_id}`));return a;},{trajectories:0,rubrics:0,scores:0,responses:new Set()});
  const kpis = [[CASES.length,'문제'],[totals.trajectories,'원본 대화 경로'],[totals.rubrics,'생성 루브릭'],[totals.responses.size,'채점 대상 응답'],[totals.scores,'실제 채점 · 응답 × 루브릭']];
  $('kpis').innerHTML = kpis.map(([n,label]) => `<div class="kpi"><strong>${num(n)}</strong><span>${E(label)}</span></div>`).join('');
  const notes = has(SUMMARY.notes) ? block('스냅샷 참고 사항',SUMMARY.notes) : '';
  $('snapshot-notes').innerHTML = notes || has(SUMMARY.counts) ? extra('스냅샷 범위와 집계',notes+raw('추출 시점 집계 원본',SUMMARY.counts)) : '';
}
function renderCaseList() {
  const q = $('case-query').value.trim().toLocaleLowerCase();
  const choices = CASES.map((c,index) => ({c,index})).filter(({c}) => !q || `${c.case_number} ${c.case_id} ${c.question} ${c.scenario}`.toLocaleLowerCase().includes(q));
  $('case-list').innerHTML = choices.map(({c,index}) => `<button class="case-button${state.caseIndex === index ? ' active' : ''}" data-case-index="${index}" aria-current="${state.caseIndex === index ? 'true' : 'false'}"><div class="case-top"><span class="case-number">문제 ${E(c.case_number ?? index+1)}</span>${badge(statusText(c.status))}</div><div class="case-id">${E(c.case_id)}</div><div class="case-meta">대화 ${list(c.trajectories).length} · 루브릭 ${list(c.rubrics).length} · 채점 ${list(c.scores).length}</div></button>`).join('') || '<p class="notes" style="padding:12px">검색한 문제가 없습니다.</p>';
}
function renderContext() {
  const c = currentCase();
  if (!CASES.length) {$('case-context').innerHTML = '<div class="panel-body"><h2>저장된 문제가 없습니다.</h2></div>';return;}
  const options = c.options && typeof c.options === 'object' ? Object.entries(c.options) : [];
  const scenario = has(c.scenario) && text(c.scenario) !== text(c.question) ? block('원본 시나리오',c.scenario,'original') : '';
  $('case-context').innerHTML = `<div class="panel-header"><div class="case-title"><h2>문제 ${E(c.case_number ?? state.caseIndex+1)}</h2>${badge(statusText(c.status))}<span class="mono muted">${E(c.case_id)}</span></div></div><div class="panel-body">${block('원본 문제',c.question,'original')}${scenario}${options.length ? `<div class="field-label">원본 보기</div><div class="options">${options.map(([key,value]) => `<div class="option"><strong>${E(key)}</strong><pre>${E(text(value))}</pre></div>`).join('')}</div>` : ''}${has(c.counts) ? raw('문제별 추출 집계',c.counts) : ''}</div>`;
}
function refreshTabs() {
  ['trajectories','rubrics','scores'].forEach(view => {
    $(`tab-${view}`).classList.toggle('active',state.view === view);
    $(`tab-${view}`).setAttribute('aria-selected',String(state.view === view));
    $(`tab-${view}`).tabIndex = state.view === view ? 0 : -1;
    $(`count-${view}`).textContent = num(caseItems(view).length);
  });
  $('rows').setAttribute('aria-labelledby',`tab-${state.view}`);
  const key = state.view === 'trajectories' ? 'outcome' : state.view === 'rubrics' ? 'status' : 'scoring_status';
  const statuses = [...new Set(caseItems().map(item => statusText(item[key])))].sort();
  if (state.status !== 'all' && !statuses.includes(state.status)) state.status = 'all';
  $('status-filter').innerHTML = '<option value="all">모든 상태</option>'+statuses.map(s => `<option value="${E(s)}">${E(s)}</option>`).join('');
  $('status-filter').value = state.status;
  $('score-filter').value = state.score;
  $('query').value = state.query;
  const hints = {trajectories:'각 경로의 전체 원본 대화를 표시합니다. Agent 발화 아래의 점수 태그를 누르면 해당 응답의 실제 점수와 판단 근거로 이동합니다.',rubrics:'User side · Agent side 조건과 0/1/2점 행동 기준을 함께 표시합니다. 독립 Gate 통과, KEEP 선택 결과, Generator 원본 점수, 독립 Gate 원본 점수는 각각 별도의 기록입니다.',scores:'채점 단위는 응답 1개 × 루브릭 1개입니다. 표시 점수는 0/1/2이며 정규화 점수는 별도 값입니다. NA는 적용 안 됨, UNSCORABLE은 채점 불가로 0점과 구분합니다.'};
  $('view-hint').textContent = hints[state.view];
}
function itemScores(item) {
  if (state.view === 'scores') return [item];
  if (state.view === 'rubrics') return caseItems('scores').filter(s => String(s.rubric_id) === String(item.rubric_id));
  const ids = new Set(list(item.dialogue).filter(d => has(d.node_id)).map(d => String(d.node_id)));
  return caseItems('scores').filter(s => ids.has(String(s.response_node_id)) || list(s.containing_trajectories).some(id => String(id?.trajectory_id ?? id) === String(item.trajectory_id)));
}
function filteredItems() {
  const q = state.query.trim().toLocaleLowerCase();
  const key = state.view === 'trajectories' ? 'outcome' : state.view === 'rubrics' ? 'status' : 'scoring_status';
  return caseItems().map((item,index) => ({item,index})).filter(({item}) => (state.status === 'all' || statusText(item[key]) === state.status) && (state.score === 'all' || itemScores(item).some(s => String(s.score_label) === state.score)) && (!q || searchable(item).includes(q) || (state.view === 'scores' && searchable(rubricIndex.get(String(item.rubric_id)) || {}).includes(q))));
}
function renderRows() {
  const items = filteredItems();
  const pages = Math.ceil(items.length/PAGE_SIZE);
  state.page = Math.min(state.page,Math.max(0,pages-1));
  const start = state.page*PAGE_SIZE;
  const visible = items.slice(start,start+PAGE_SIZE);
  const render = {trajectories:trajectoryCard,rubrics:rubricCard,scores:scoreCard}[state.view];
  $('rows').innerHTML = visible.map(({item,index},position) => render(item,index,state.focus === index || position === 0)).join('') || '<div class="empty">현재 필터에 해당하는 기록이 없습니다.<br>다른 문제나 검토 항목을 선택하거나 필터를 초기화하세요.</div>';
  $('result-count').textContent = `${num(items.length)} / ${num(caseItems().length)}건${items.length ? ` · ${num(start+1)}–${num(start+visible.length)} 표시` : ''}`;
  $('pager').innerHTML = pages > 1 ? `<button data-page="${state.page-1}"${state.page === 0 ? ' disabled' : ''}>이전</button><span>${state.page+1} / ${pages} 페이지 · 페이지당 ${PAGE_SIZE}건</span><button data-page="${state.page+1}"${state.page+1 >= pages ? ' disabled' : ''}>다음</button>` : '';
  $('expand-visible').disabled = !visible.length;
  $('expand-visible').textContent = '현재 페이지 펼치기';
  if (state.focus !== null) {
    const prefix = {trajectories:'trajectory',rubrics:'rubric',scores:'score'}[state.view];
    const element = $(`${prefix}-${state.focus}`);
    if (element) {element.classList.add('focused');element.scrollIntoView({block:'start',behavior:'smooth'});}
    state.focus = null;
  }
}
function resetFilters() {state.query='';state.score='all';state.status='all';state.page=0;state.focus=null;}
function switchView(view) {state.view=view;state.status='all';state.page=0;state.focus=null;refreshTabs();renderRows();}
function navigate(view,index) {
  if (index < 0 || !Number.isInteger(index)) return;
  resetFilters();state.view=view;state.page=Math.floor(index/PAGE_SIZE);state.focus=index;refreshTabs();renderRows();
}
function selectCase(index) {
  if (!CASES[index]) return;
  state.caseIndex=index;resetFilters();reindex();renderCaseList();renderContext();refreshTabs();renderRows();
}
$('case-query').addEventListener('input',renderCaseList);
$('query').addEventListener('input',event => {state.query=event.target.value;state.page=0;renderRows();});
$('score-filter').addEventListener('change',event => {state.score=event.target.value;state.page=0;renderRows();});
$('status-filter').addEventListener('change',event => {state.status=event.target.value;state.page=0;renderRows();});
$('reset').addEventListener('click',() => {resetFilters();refreshTabs();renderRows();});
$('expand-visible').addEventListener('click',() => {
  const records = [...$('rows').querySelectorAll('details.record')];
  const expand = records.some(record => !record.open);
  records.forEach(record => {record.open=expand;});
  $('expand-visible').textContent = expand ? '현재 페이지 접기' : '현재 페이지 펼치기';
});
document.querySelector('.tabs').addEventListener('keydown',event => {
  if (!['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) return;
  const tabs=['trajectories','rubrics','scores'];
  let index=tabs.indexOf(state.view);
  index=event.key==='Home' ? 0 : event.key==='End' ? 2 : (index+(event.key==='ArrowRight' ? 1 : 2))%3;
  event.preventDefault();switchView(tabs[index]);$(`tab-${tabs[index]}`).focus();
});
document.addEventListener('click',event => {
  const button=event.target.closest('button');if(!button)return;
  const d=button.dataset;
  if(d.caseIndex!==undefined)selectCase(Number(d.caseIndex));
  else if(d.view)switchView(d.view);
  else if(d.goScore!==undefined)navigate('scores',Number(d.goScore));
  else if(d.goRubric!==undefined)navigate('rubrics',caseItems('rubrics').findIndex(r=>String(r.rubric_id)===d.goRubric));
  else if(d.goTrajectory!==undefined)navigate('trajectories',caseItems('trajectories').findIndex(t=>String(t.trajectory_id)===d.goTrajectory));
  else if(d.rubricScores!==undefined){resetFilters();state.view='scores';state.query=d.rubricScores;refreshTabs();renderRows();}
  else if(d.page!==undefined){state.page=Number(d.page);renderRows();$('result-count').scrollIntoView({block:'start',behavior:'smooth'});}
});
$('download').addEventListener('click',() => {
  const blob=new Blob([JSON.stringify(DATA,null,2)+'\n'],{type:'application/json;charset=utf-8'});
  const url=URL.createObjectURL(blob),link=document.createElement('a');
  link.href=url;link.download='training_review_snapshot.json';document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
renderHeader();reindex();renderCaseList();renderContext();refreshTabs();renderRows();
</script>
</body>
</html>
'''
