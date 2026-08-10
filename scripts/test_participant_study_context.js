"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const root = path.resolve(__dirname, "..");
const templateSource = fs.readFileSync(
  path.join(root, "web/templates/participant_study.html"),
  "utf8",
);
const scriptSource = fs.readFileSync(
  path.join(root, "web/static/participant_study.js"),
  "utf8",
);

function extractFunction(source, name) {
  const marker = `function ${name}(`;
  const start = source.indexOf(marker);
  assert.notEqual(start, -1, `missing ${name}`);
  const openingBrace = source.indexOf("{", start);
  assert.notEqual(openingBrace, -1, `missing ${name} body`);
  let depth = 0;
  for (let index = openingBrace; index < source.length; index += 1) {
    if (source[index] === "{") depth += 1;
    if (source[index] === "}") depth -= 1;
    if (depth === 0) return source.slice(start, index + 1);
  }
  assert.fail(`unterminated ${name} body`);
}

function fakeClassList(initial = []) {
  const values = new Set(initial);
  return {
    add(...names) { names.forEach((name) => values.add(name)); },
    remove(...names) { names.forEach((name) => values.delete(name)); },
    toggle(name, force) {
      if (force === true) values.add(name);
      else if (force === false) values.delete(name);
      else if (values.has(name)) values.delete(name);
      else values.add(name);
      return values.has(name);
    },
    contains(name) { return values.has(name); },
  };
}

function runScenario({ sessionState, confirmResult }) {
  const removals = [];
  const alerts = [];
  const steps = Array.from({ length: 5 }, () => ({
    classList: fakeClassList(["active", "done"]),
  }));
  const nodes = Object.fromEntries([
    "consentForm", "workflowPanel", "consentPanel", "receiptBox", "withdrawalCode",
    "withdrawSessionId", "withdrawCodeInput", "withdrawResult", "systemCheckResults",
  ].map((id) => [id, {
    classList: fakeClassList(),
    textContent: "stale",
    value: "stale",
  }]));
  nodes.consentPanel.classList.add("hidden");
  let resetCount = 0;
  nodes.consentForm.reset = () => { resetCount += 1; };
  let resultChildrenCleared = 0;
  nodes.systemCheckResults.replaceChildren = () => { resultChildrenCleared += 1; };
  let confirmCount = 0;
  let scrollCount = 0;
  const sandbox = {
    STORAGE_KEY: "lexigaze.participantStudy.v1",
    state: {
      context: { study_session_id: "ST-OLD", access_token: "secret" },
      sessionState,
      withdrawalCode: "WD-OLD",
      receiptText: "old receipt",
    },
    sessionStorage: {
      removeItem(key) { removals.push(key); },
    },
    document: {
      getElementById(id) { return nodes[id]; },
      querySelectorAll(selector) {
        assert.equal(selector, ".step");
        return steps;
      },
    },
    window: {
      confirm() {
        confirmCount += 1;
        return confirmResult;
      },
      scrollTo() { scrollCount += 1; },
    },
    showAlert(message, type) { alerts.push({ message, type }); },
  };
  sandbox.$ = (id) => nodes[id];

  vm.runInNewContext(
    [
      extractFunction(scriptSource, "clearContext"),
      extractFunction(scriptSource, "showConsentInviteForm"),
      extractFunction(scriptSource, "startAnotherInvite"),
      "startAnotherInvite();",
    ].join("\n"),
    sandbox,
  );

  return {
    sandbox,
    nodes,
    steps,
    removals,
    alerts,
    confirmCount,
    resetCount,
    resultChildrenCleared,
    scrollCount,
  };
}

assert.match(
  templateSource,
  /id="startAnotherInviteBtn"[\s\S]*?>開始另一個邀請 \/ Visit<\/button>/,
);
assert.match(templateSource, /這不會撤回、刪除或變更任何伺服器資料/);
assert.match(templateSource, /請先保存撤回碼與同意憑證/);
assert.match(
  scriptSource,
  /startAnotherInviteBtn"\)\.addEventListener\("click", startAnotherInvite\)/,
);

const resetSource = extractFunction(scriptSource, "startAnotherInvite");
assert.match(resetSource, /clearContext\(\)/);
assert.match(resetSource, /showConsentInviteForm\(\)/);
assert.match(resetSource, /state\.sessionState === "completed"/);
assert.match(resetSource, /保存撤回碼與同意憑證/);
assert.match(resetSource, /generalCollectionDraft/);
assert.match(resetSource, /generalCollectionPractice/);
assert.doesNotMatch(resetSource, /\b(?:api|fetch|withdraw)\s*\(/);

const completedCancelled = runScenario({ sessionState: "completed", confirmResult: false });
assert.equal(completedCancelled.confirmCount, 1);
assert.deepEqual(completedCancelled.removals, []);
assert.equal(completedCancelled.sandbox.state.context.study_session_id, "ST-OLD");
assert.equal(completedCancelled.resetCount, 0);
assert.equal(completedCancelled.alerts.length, 0);

const completed = runScenario({ sessionState: "completed", confirmResult: true });
assert.equal(completed.confirmCount, 1);
assert.deepEqual(new Set(completed.removals), new Set([
  "lexigaze.generalCollectionDraft.ST-OLD",
  "lexigaze.generalCollectionPractice.ST-OLD",
  "lexigaze.participantStudy.v1",
]));
assert.equal(completed.sandbox.state.context, null);
assert.equal(completed.sandbox.state.sessionState, null);
assert.equal(completed.sandbox.state.withdrawalCode, null);
assert.equal(completed.sandbox.state.receiptText, null);
assert.equal(completed.resetCount, 1);
assert.equal(completed.resultChildrenCleared, 1);
assert.equal(completed.scrollCount, 1);
assert.equal(completed.nodes.workflowPanel.classList.contains("hidden"), true);
assert.equal(completed.nodes.consentPanel.classList.contains("hidden"), false);
assert.equal(completed.nodes.receiptBox.classList.contains("hidden"), true);
assert.equal(completed.nodes.withdrawSessionId.value, "");
assert.equal(completed.nodes.withdrawCodeInput.value, "");
assert.equal(completed.steps[0].classList.contains("active"), true);
assert.equal(completed.steps[0].classList.contains("done"), false);
assert.equal(completed.steps[1].classList.contains("active"), false);
assert.equal(completed.alerts.length, 1);
assert.equal(completed.alerts[0].type, "info");
assert.match(completed.alerts[0].message, /伺服器資料沒有被撤回或刪除/);

const activeCancelled = runScenario({
  sessionState: "assessment_in_progress",
  confirmResult: false,
});
assert.equal(activeCancelled.confirmCount, 1);
assert.deepEqual(activeCancelled.removals, []);
assert.equal(activeCancelled.sandbox.state.context.study_session_id, "ST-OLD");
assert.equal(activeCancelled.resetCount, 0);
assert.equal(activeCancelled.alerts.length, 0);

const activeConfirmed = runScenario({
  sessionState: "assessment_in_progress",
  confirmResult: true,
});
assert.equal(activeConfirmed.confirmCount, 1);
assert.equal(activeConfirmed.sandbox.state.context, null);
assert.equal(activeConfirmed.nodes.consentPanel.classList.contains("hidden"), false);
assert.equal(activeConfirmed.alerts.length, 1);

console.log("participant study context reset tests passed");
