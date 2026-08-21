import assert from "node:assert/strict";
import fs from "node:fs";

const html = fs.readFileSync(new URL("../web/index.html", import.meta.url), "utf8");
const source = html.match(/function parseThinking\([\s\S]*?\n}\n\nfunction renderMessages/)?.[0]
  .replace(/\n\nfunction renderMessages$/, "");
assert.ok(source, "parseThinking not found");
const parseThinking = Function(`${source}; return parseThinking`)();

// `thinking` is request mode, not content type. Extra legacy argument must not
// turn normal assistant content into a reasoning-only details block.
assert.deepEqual(parseThinking("Привет! Чем могу помочь?", true), [
  { kind: "text", text: "Привет! Чем могу помочь?" },
]);
assert.deepEqual(parseThinking("<think>план</think>ответ"), [
  { kind: "think", text: "план" },
  { kind: "text", text: "ответ" },
]);

assert.match(html, /attachbtn"\)\.disabled = !enabled/);
assert.match(html, /Gemma 4 GGUF сейчас загружена без vision mmproj runtime/);
assert.match(html, /history\.reasoning_content = m\.thinkText/);
assert.match(html, /if \(!history\.content\.trim\(\)\) history\.content = m\.thinkText/);
assert.match(html, /if \(j\.error\) throw new Error/);
assert.match(html, /модель завершила генерацию без текстового ответа/);
assert.match(html, /"instruct-reasoning": \{ temperature: 0\.7/);
assert.match(html, /for \(const name of Object\.keys\(SERVER_PRESETS\)\) delete SERVER_PRESETS\[name\]/);
assert.match(html, /const nextSamplingModelKey = \[m\.sampling_family/);
assert.match(html, /const enableThink = thinkingEnabled && modeUsesThinking\(mode\)/);
assert.match(html, /thinking: modeUsesThinking\(\$\("mode"\)\.value\)/);
assert.match(html, /Math\.round\(numeric \* 1e6\) \/ 1e6/);
const clearPresetsAt = html.indexOf("Object.keys(SERVER_PRESETS)");
const loadPresetsAt = html.indexOf("Object.assign(SERVER_PRESETS, m.sampling_presets)");
const applyPresetAt = html.indexOf('applyPreset($("mode").value)', loadPresetsAt);
assert.ok(clearPresetsAt > 0 && clearPresetsAt < loadPresetsAt && loadPresetsAt < applyPresetAt,
  "model switch must clear stale presets before loading and applying the active family");

console.log("webui render regression: ok");
