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

console.log("webui render regression: ok");
