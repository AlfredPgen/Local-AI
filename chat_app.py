"""Chat with a tinyGPT checkpoint in the browser (a small local app, standard library only).

    python chat_app.py                         # opens http://127.0.0.1:8765 with tinyGPT_muon_sft.pt
    python chat_app.py --checkpoint tinyGPT_muon_best.pt --port 8766

The page lists the checkpoints in this folder; switching one loads it (one model in memory
at a time). Each question is answered on its own, in the fine-tuning template, because the
models were fine-tuned on single questions; earlier turns stay on screen only. Answers stream
word by word. The server listens on 127.0.0.1 only: nothing is reachable from other machines.
"""
import argparse
import glob
import json
import os
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

import finetune  # noqa: E402
import tiny_gpt  # noqa: E402


class Engine:
    """One loaded checkpoint; generation is serialised (one answer at a time)."""

    def __init__(self, device):
        self.device = device
        self.amp_dtype, _, _ = tiny_gpt.choose_precision(device, "auto", training=False)
        self.lock = threading.Lock()
        self.name = self.model = self.tok = None

    def load(self, name):
        path = os.path.join(HERE, os.path.basename(name))  # only files in this folder
        if not (path.endswith(".pt") and os.path.isfile(path)):
            raise ValueError(f"no checkpoint {name}")
        with self.lock:
            if self.name == os.path.basename(path):
                return
            self.model = self.tok = None
            tiny_gpt.free_device_memory(self.device)
            self.model, self.tok, _, _ = tiny_gpt.load_for_inference(path, self.device, False)  # weights only
            self.model.eval()
            self.name = os.path.basename(path)

    def stream(self, question, max_new_tokens, temperature, seed):
        """Yields the answer's text as it grows (the full text so far, decoded from all its tokens, so a
        character split over two tokens is never shown half-made)."""
        with self.lock, torch.no_grad():
            model, tok = self.model, self.tok
            prefix = [tok.bos_id] if tok.bos_id >= 0 else []
            ids = prefix + tok.encode(finetune.TEMPLATE.format(prompt=question.strip()))
            start = len(ids)
            decoder = tiny_gpt.IncrementalDecoder(model, self.device, self.amp_dtype)
            gen = torch.Generator().manual_seed(seed)
            for _ in range(max_new_tokens):
                logits = decoder.next_logits(ids)
                if temperature <= 0:
                    nxt = int(logits.argmax())
                else:
                    nxt = int(tiny_gpt.sample_next(logits[None], gen, temperature, 40, 0.9)[0])
                if nxt == tok.eos_id:
                    break
                ids.append(nxt)
                yield tok.decode(ids[start:])


def checkpoints():
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(HERE, "*.pt"))
                  if not p.endswith("_partial.pt"))


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>tinyGPT chat</title>
<style>
:root{--bg:#f7f6f2;--panel:#fff;--ink:#1b1b19;--muted:#6f6e69;--line:#e3e1da;--me:#2a78d6;--me-ink:#fff;--accent:#2a78d6}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--panel:#1f1f1d;--ink:#ecebe6;--muted:#9a9993;--line:#33322f;--me:#2f6fbf;--me-ink:#fff;--accent:#5b9cf0}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;height:100vh;display:flex;flex-direction:column}
header{display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:10px 16px;border-bottom:1px solid var(--line);background:var(--panel)}
header h1{font-size:16px;margin:0 8px 0 0}
header label{font-size:13px;color:var(--muted);display:flex;align-items:center;gap:6px}
select,input[type=number]{font:inherit;font-size:13px;padding:4px 6px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink)}
input[type=number]{width:70px}
#status{font-size:12px;color:var(--muted);margin-left:auto}
#log{flex:1;overflow-y:auto;padding:18px 16px}
.msg{max-width:780px;margin:0 auto 14px;display:flex}
.msg div{padding:10px 14px;border-radius:12px;white-space:pre-wrap;word-wrap:break-word}
.me{justify-content:flex-end}.me div{background:var(--me);color:var(--me-ink);border-bottom-right-radius:4px;max-width:75%}
.bot div{background:var(--panel);border:1px solid var(--line);border-bottom-left-radius:4px;max-width:85%}
.meta{display:block;font-size:11px;color:var(--muted);margin-top:6px}
.empty{color:var(--muted);text-align:center;margin-top:12vh}
form{display:flex;gap:8px;padding:12px 16px;border-top:1px solid var(--line);background:var(--panel)}
textarea{flex:1;font:inherit;resize:none;height:46px;padding:10px 12px;border:1px solid var(--line);border-radius:10px;background:var(--bg);color:var(--ink)}
button{font:inherit;padding:0 18px;border:0;border-radius:10px;background:var(--accent);color:#fff;cursor:pointer}
button:disabled{opacity:.5;cursor:default}
</style></head><body>
<header><h1>tinyGPT chat</h1>
<label>model <select id="model"></select></label>
<label>max tokens <input id="maxtok" type="number" min="10" max="1000" value="200"></label>
<label>temperature <input id="temp" type="number" min="0" max="2" step="0.1" value="0.7"></label>
<span id="status"></span></header>
<div id="log"><p class="empty">Ask a question. Each one is answered on its own; the model does not see earlier turns.</p></div>
<form id="f"><textarea id="q" placeholder="Ask something... (Enter sends, Shift+Enter: new line)"></textarea><button id="send">Send</button></form>
<script>
const $=id=>document.getElementById(id), log=$("log"), status=$("status");
let busy=false;
async function init(){
  const r=await fetch("/api/models"), d=await r.json();
  for(const m of d.models){const o=document.createElement("option");o.value=o.textContent=m;if(m===d.current)o.selected=true;$("model").appendChild(o)}
  status.textContent=d.current?("loaded "+d.current+" on "+d.device):"";
}
$("model").onchange=async e=>{
  status.textContent="loading "+e.target.value+" ...";$("send").disabled=true;
  const r=await fetch("/api/load",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({model:e.target.value})});
  const d=await r.json();status.textContent=d.error?("error: "+d.error):("loaded "+d.current);$("send").disabled=false;
};
function bubble(cls,text){const e=log.querySelector(".empty");if(e)e.remove();
  const w=document.createElement("div");w.className="msg "+cls;const b=document.createElement("div");b.textContent=text;w.appendChild(b);log.appendChild(w);log.scrollTop=log.scrollHeight;return b}
$("q").addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();$("f").requestSubmit()}});
$("f").onsubmit=async e=>{
  e.preventDefault();const q=$("q").value.trim();if(!q||busy)return;
  busy=true;$("send").disabled=true;$("q").value="";bubble("me",q);
  const b=bubble("bot","...");const t0=performance.now();let text="",model=$("model").value;
  try{
    const r=await fetch("/api/ask",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({question:q,max_new_tokens:+$("maxtok").value,temperature:+$("temp").value})});
    const rd=r.body.getReader(),dec=new TextDecoder();let buf="";
    for(;;){const {value,done}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});
      let i;while((i=buf.indexOf("\n"))>=0){const line=buf.slice(0,i);buf=buf.slice(i+1);if(!line)continue;
        const m=JSON.parse(line);if(m.error){text="error: "+m.error}else{text=m.text}b.textContent=text||"...";log.scrollTop=log.scrollHeight}}
  }catch(err){b.textContent="error: "+err}
  const meta=document.createElement("span");meta.className="meta";meta.textContent=model+" · "+((performance.now()-t0)/1000).toFixed(1)+" s";b.appendChild(meta);
  busy=false;$("send").disabled=false;$("q").focus();
};
init();
</script></body></html>"""


def make_handler(engine):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet console
            pass

        def send_json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def read_json(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            if self.path == "/":
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/models":
                self.send_json({"models": checkpoints(), "current": engine.name,
                                "device": tiny_gpt.device_label(engine.device)})
            else:
                self.send_error(404)

        def do_POST(self):
            try:
                req = self.read_json()
            except ValueError:
                return self.send_json({"error": "bad request"}, 400)
            if self.path == "/api/load":
                try:
                    engine.load(str(req.get("model", "")))
                    return self.send_json({"current": engine.name})
                except Exception as exc:  # shown on the page
                    return self.send_json({"error": str(exc)}, 400)
            if self.path != "/api/ask":
                return self.send_error(404)
            question = str(req.get("question", "")).strip()[:4000]
            max_new = max(1, min(int(req.get("max_new_tokens") or 200), 1000))
            temperature = max(0.0, min(float(req.get("temperature") or 0.0), 2.0))
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                last = 0.0
                text = ""
                for text in engine.stream(question, max_new, temperature, int(time.time())):
                    if time.time() - last > 0.05:  # at most ~20 updates a second
                        self.wfile.write((json.dumps({"text": text}) + "\n").encode())
                        self.wfile.flush()
                        last = time.time()
                self.wfile.write((json.dumps({"text": text, "done": True}) + "\n").encode())
            except (BrokenPipeError, ConnectionResetError):  # the page was closed mid-answer
                pass
            except Exception as exc:
                self.wfile.write((json.dumps({"error": str(exc)}) + "\n").encode())

    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="tinyGPT_muon_sft.pt")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()
    engine = Engine(tiny_gpt.select_device(args.device))
    print(f"loading {args.checkpoint} on {tiny_gpt.device_label(engine.device)} ...", flush=True)
    engine.load(args.checkpoint)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(engine))
    url = f"http://127.0.0.1:{args.port}"
    print(f"chat at {url}  (Ctrl+C here stops it)", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, (url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
