import { useEffect, useRef, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import {
  Activity,
  BookOpenCheck,
  Bot,
  CheckCircle2,
  CircleAlert,
  FlaskConical,
  LineChart,
  Paperclip,
  Send,
  Sigma,
  Sparkles,
  Star,
  Wrench,
  X,
} from "lucide-react";

/* ---------- types ---------- */

interface Step {
  step_number: number;
  description: string;
  formula: string | null;
  explanation: string;
}

interface SolveData {
  question: string;
  solution: {
    steps: Step[];
    final_answer: string;
    verification: { verified: boolean; method: string; details: string | null };
    references: string[];
    diagnosis: { error_step: number | null; misconception: string | null } | null;
  };
  source: string;
  tools_used: string[];
  sympy_verified: boolean;
  steps_checked: number;
  corrected: boolean;
  confidence: number | null;
  plots: string[];
  model: string;
  est_cost: number;
  turns: number;
  processing_time: number;
}

interface PracticeData {
  problem: string;
  steps: Step[];
  final_answer: string;
  sympy_verified: boolean;
  topic: string;
  difficulty: string;
}

interface FeedItem {
  kind: "status" | "tool" | "result" | "error";
  text: string;
}

type SSEEvent = Record<string, any> & { type: string };

/* ---------- helpers ---------- */

async function readSSE(
  url: string,
  payload: unknown,
  onEvent: (ev: SSEEvent) => void,
): Promise<void> {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!resp.ok || !resp.body) {
    const err = await resp.json().catch(() => ({}) as any);
    throw new Error(err.detail?.detail || err.detail || resp.statusText);
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, idx).trim();
      buf = buf.slice(idx + 2);
      if (chunk.startsWith("data: ")) onEvent(JSON.parse(chunk.slice(6)));
    }
  }
}

declare global {
  interface Window {
    katex?: { renderToString(tex: string, opts?: object): string };
  }
}

function Formula({ tex }: { tex: string }) {
  // Backend formulas may be LaTeX or SymPy-ish plain text; only KaTeX the former.
  const looksLatex = /[\\{}]|\^\{|_\{/.test(tex) && !tex.includes("**");
  if (looksLatex && window.katex) {
    try {
      const html = window.katex.renderToString(tex, { throwOnError: false, displayMode: false });
      return <span dangerouslySetInnerHTML={{ __html: html }} />;
    } catch {
      /* fall through to plain */
    }
  }
  return <code className="font-mono2 text-[13px] text-accent-foreground">{tex}</code>;
}

function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve((r.result as string).split(",")[1]);
    r.onerror = reject;
    r.readAsDataURL(file);
  });
}

/* ---------- badges ---------- */

function VerifyBadges({ data }: { data: SolveData }) {
  return (
    <div className="flex flex-wrap gap-2">
      <Badge variant="secondary" className="font-mono2 font-normal">
        source: {data.source}
      </Badge>
      {data.sympy_verified ? (
        <Badge className="gap-1 border-transparent bg-ok-soft text-ok hover:bg-ok-soft">
          <CheckCircle2 className="h-3.5 w-3.5" /> SymPy-verified
        </Badge>
      ) : (
        <Badge className="gap-1 border-transparent bg-warn-soft text-warn hover:bg-warn-soft">
          <CircleAlert className="h-3.5 w-3.5" /> not machine-verified
        </Badge>
      )}
      {data.steps_checked > 0 && (
        <Badge className="border-transparent bg-ok-soft text-ok hover:bg-ok-soft">
          {data.steps_checked} steps spot-checked
        </Badge>
      )}
      {data.corrected && (
        <Badge className="border-transparent bg-warn-soft text-warn hover:bg-warn-soft">
          self-corrected
        </Badge>
      )}
      {data.confidence != null && (
        <Badge variant="outline" className="font-mono2 font-normal">
          ~{Math.round(data.confidence * 100)}% historical accuracy
        </Badge>
      )}
    </div>
  );
}

/* ---------- solution view ---------- */

function SolutionView({
  data,
  onRate,
  rated,
}: {
  data: SolveData;
  onRate: (n: number) => void;
  rated: number;
}) {
  const d = data.solution.diagnosis;
  return (
    <div className="flex flex-col gap-4">
      <VerifyBadges data={data} />

      {d && d.error_step != null && (
        <div className="rounded-md bg-warn-soft px-4 py-3 text-sm font-semibold text-warn">
          Error found in your step {d.error_step}: {d.misconception}
        </div>
      )}
      {d && d.error_step == null && d.misconception && (
        <div className="rounded-md bg-ok-soft px-4 py-3 text-sm font-semibold text-ok">
          {d.misconception}
        </div>
      )}

      <ol className="flex flex-col gap-3">
        {data.solution.steps.map((s) => (
          <li key={s.step_number} className="border-l-2 border-primary/60 pl-4">
            <p className="font-semibold">
              {s.step_number}. {s.description}
            </p>
            {s.formula && (
              <p className="my-1 overflow-x-auto">
                <Formula tex={s.formula} />
              </p>
            )}
            <p className="text-sm text-muted-foreground">{s.explanation}</p>
          </li>
        ))}
      </ol>

      {data.plots.map((url) => (
        <img key={url} src={url} alt="graph" className="max-w-full rounded-md border" />
      ))}

      <div className="rounded-md bg-ok-soft px-4 py-3 text-base font-bold">
        {data.solution.final_answer}
      </div>

      <p className="font-mono2 text-xs leading-relaxed text-muted-foreground">
        verification: {data.solution.verification.method} · tools:{" "}
        {data.tools_used.join(", ") || "none"} · {data.turns} turns ·{" "}
        {data.processing_time}s · {data.model} · ~${data.est_cost.toFixed(3)}
        {data.solution.references.length > 0 &&
          ` · refs: ${data.solution.references.join(", ")}`}
      </p>

      <div className="flex items-center gap-2 border-t pt-3">
        <span className="text-sm text-muted-foreground">Rate this solution</span>
        {[1, 2, 3, 4, 5].map((n) => (
          <button key={n} onClick={() => onRate(n)} aria-label={`${n} stars`}>
            <Star
              className={`h-5 w-5 ${n <= rated ? "fill-primary text-primary" : "text-muted-foreground"}`}
            />
          </button>
        ))}
        {rated > 0 && <span className="text-xs text-muted-foreground">saved — thanks</span>}
      </div>
    </div>
  );
}

/* ---------- solve tab ---------- */

const SAMPLES = [
  "Solve the quadratic equation x^2 + 5x + 6 = 0",
  "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 1",
  "Find the eigenvalues of the matrix [[2, 1], [1, 2]]",
  "Find the area bounded by y = x^2 and y = x",
];

function SolvePanel() {
  const [question, setQuestion] = useState("");
  const [image, setImage] = useState<{ b64: string; media: string; preview: string } | null>(null);
  const [adv, setAdv] = useState(false);
  const [diag, setDiag] = useState(false);
  const [busy, setBusy] = useState(false);
  const [feed, setFeed] = useState<FeedItem[]>([]);
  const [result, setResult] = useState<SolveData | null>(null);
  const [rated, setRated] = useState(0);
  const fileRef = useRef<HTMLInputElement>(null);

  const push = (item: FeedItem) => setFeed((f) => [...f, item]);

  async function solve() {
    if (!question.trim() || busy) return;
    setBusy(true);
    setFeed([]);
    setResult(null);
    setRated(0);
    const payload: Record<string, unknown> = { question: question.trim() };
    if (image) {
      payload.image_base64 = image.b64;
      payload.image_media_type = image.media;
    }
    if (adv) payload.effort = "xhigh";
    if (diag) payload.mode = "diagnose";
    try {
      await readSSE("/api/v1/solve", payload, (ev) => {
        if (ev.type === "status") push({ kind: "status", text: ev.message });
        else if (ev.type === "tool_call")
          push({ kind: "tool", text: `${ev.tool} ${ev.input ? JSON.stringify(ev.input) : ""}` });
        else if (ev.type === "tool_result")
          push({ kind: ev.is_error ? "error" : "result", text: `${ev.tool}: ${ev.preview}` });
        else if (ev.type === "error") push({ kind: "error", text: ev.message });
        else if (ev.type === "solution") setResult(ev.data as SolveData);
      });
    } catch (e) {
      push({ kind: "error", text: (e as Error).message });
    } finally {
      setBusy(false);
    }
  }

  async function attach(file: File) {
    const b64 = await fileToBase64(file);
    setImage({ b64, media: file.type, preview: URL.createObjectURL(file) });
  }

  async function rate(n: number) {
    if (!result) return;
    setRated(n);
    await fetch("/api/v1/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: result.question,
        final_answer: result.solution.final_answer,
        accuracy: n,
        clarity: n,
      }),
    }).catch(() => setRated(0));
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[5fr_6fr]">
      <div className="flex flex-col gap-4">
        <Card>
          <CardContent className="flex flex-col gap-3 p-4">
            <Textarea
              id="solve-question"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) solve();
              }}
              placeholder="Ask a math question — e.g. Solve x² + 5x + 6 = 0"
              className="min-h-24 resize-y"
            />
            {image && (
              <div className="flex items-center gap-3">
                <img src={image.preview} alt="attached figure" className="max-h-20 rounded-md border" />
                <Button variant="ghost" size="sm" onClick={() => setImage(null)}>
                  <X className="mr-1 h-4 w-4" /> remove figure
                </Button>
              </div>
            )}
            <div className="flex flex-wrap items-center gap-x-5 gap-y-3">
              <Button onClick={solve} disabled={busy || !question.trim()}>
                <Sigma className="mr-1.5 h-4 w-4" />
                {busy ? "Working…" : "Solve"}
              </Button>
              <Button variant="outline" size="sm" onClick={() => fileRef.current?.click()}>
                <Paperclip className="mr-1.5 h-4 w-4" /> figure
              </Button>
              <input
                ref={fileRef}
                type="file"
                accept="image/png,image/jpeg,image/webp,image/gif"
                className="hidden"
                onChange={(e) => e.target.files?.[0] && attach(e.target.files[0])}
              />
              <div className="flex items-center gap-2">
                <Switch id="adv" checked={adv} onCheckedChange={setAdv} />
                <Label htmlFor="adv" className="text-sm">JEE Advanced effort</Label>
              </div>
              <div className="flex items-center gap-2">
                <Switch id="diag" checked={diag} onCheckedChange={setDiag} />
                <Label htmlFor="diag" className="text-sm">Diagnose my attempt</Label>
              </div>
            </div>
            <div className="flex flex-wrap gap-2">
              {SAMPLES.map((s) => (
                <button
                  key={s}
                  onClick={() => setQuestion(s)}
                  className="rounded-full border px-3 py-1 font-mono2 text-xs text-muted-foreground hover:border-primary hover:text-foreground"
                >
                  {s.length > 42 ? s.slice(0, 42) + "…" : s}
                </button>
              ))}
            </div>
          </CardContent>
        </Card>

        {feed.length > 0 && (
          <Card>
            <CardContent className="p-4">
              <p className="mb-2 flex items-center gap-2 font-display text-sm font-semibold">
                <Activity className="h-4 w-4 text-primary" /> Agent activity
              </p>
              <ul className="flex max-h-72 flex-col gap-1.5 overflow-y-auto font-mono2 text-xs">
                {feed.map((f, i) => (
                  <li
                    key={i}
                    className={
                      f.kind === "error"
                        ? "text-destructive"
                        : f.kind === "tool"
                          ? "text-accent-foreground"
                          : f.kind === "result"
                            ? "text-ok"
                            : "text-muted-foreground"
                    }
                  >
                    {f.kind === "tool" ? "▸ " : f.kind === "result" ? "✓ " : f.kind === "error" ? "✕ " : "… "}
                    {f.text}
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        )}
      </div>

      <Card className={result ? "" : "border-dashed"}>
        <CardContent className="p-5">
          {result ? (
            <SolutionView data={result} onRate={rate} rated={rated} />
          ) : (
            <div className="flex h-full min-h-48 flex-col items-center justify-center gap-2 text-muted-foreground">
              <BookOpenCheck className="h-8 w-8" />
              <p className="text-sm">The verified, step-checked solution appears here.</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

/* ---------- tutor tab ---------- */

interface ChatMsg {
  who: "you" | "tutor";
  text: string;
  pending?: boolean;
}

function TutorPanel() {
  const [msgs, setMsgs] = useState<ChatMsg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const session = useRef<string | null>(null);
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => bottom.current?.scrollIntoView({ behavior: "smooth" }), [msgs]);

  async function send() {
    const text = input.trim();
    if (!text || busy) return;
    setInput("");
    setBusy(true);
    setMsgs((m) => [...m, { who: "you", text }, { who: "tutor", text: "…", pending: true }]);
    const setLast = (t: string, pending = false) =>
      setMsgs((m) => [...m.slice(0, -1), { who: "tutor", text: t, pending }]);
    try {
      await readSSE("/api/v1/tutor", { message: text, session_id: session.current }, (ev) => {
        if (ev.type === "session") session.current = ev.session_id;
        else if (ev.type === "tool_call") setLast("checking your math…", true);
        else if (ev.type === "tutor_reply") setLast(ev.text);
        else if (ev.type === "error") setLast(`✕ ${ev.message}`);
      });
    } catch (e) {
      setLast(`✕ ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card>
      <CardContent className="flex flex-col gap-3 p-4">
        <p className="text-sm text-muted-foreground">
          Guided hints, one step at a time — the tutor checks your algebra with SymPy but
          won't hand over the answer.
        </p>
        <div className="flex max-h-[26rem] min-h-40 flex-col gap-2 overflow-y-auto">
          {msgs.length === 0 && (
            <div className="flex flex-1 items-center justify-center gap-2 text-muted-foreground">
              <Bot className="h-5 w-5" />
              <span className="text-sm">Tell the tutor what you're working on.</span>
            </div>
          )}
          {msgs.map((m, i) => (
            <div
              key={i}
              className={`max-w-[85%] whitespace-pre-wrap rounded-lg px-3.5 py-2 text-sm ${
                m.who === "you"
                  ? "self-end bg-primary text-primary-foreground"
                  : "self-start bg-secondary text-secondary-foreground"
              } ${m.pending ? "italic opacity-70" : ""}`}
            >
              {m.text}
            </div>
          ))}
          <div ref={bottom} />
        </div>
        <div className="flex gap-2">
          <Input
            id="tutor-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && send()}
            placeholder="I'm stuck on integrating x·sin(x)…"
          />
          <Button onClick={send} disabled={busy || !input.trim()}>
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

/* ---------- practice tab ---------- */

function PracticePanel() {
  const [topic, setTopic] = useState("");
  const [difficulty, setDifficulty] = useState<"JEE Main" | "JEE Advanced">("JEE Main");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [problem, setProblem] = useState<PracticeData | null>(null);
  const [revealed, setRevealed] = useState(false);

  async function generate() {
    if (!topic.trim() || busy) return;
    setBusy(true);
    setProblem(null);
    setRevealed(false);
    setStatus("Creating and verifying a problem…");
    try {
      await readSSE("/api/v1/practice", { topic: topic.trim(), difficulty }, (ev) => {
        if (ev.type === "status") setStatus(ev.message);
        else if (ev.type === "tool_call") setStatus(`verifying with ${ev.tool}…`);
        else if (ev.type === "error") setStatus(`✕ ${ev.message}`);
        else if (ev.type === "practice") {
          setProblem(ev.data as PracticeData);
          setStatus("");
        }
      });
    } catch (e) {
      setStatus(`✕ ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card>
      <CardContent className="flex flex-col gap-4 p-4">
        <p className="text-sm text-muted-foreground">
          A fresh, original problem — solved and machine-verified before you ever see it,
          so the answer key is guaranteed.
        </p>
        <div className="flex flex-wrap items-center gap-2">
          <Input
            id="practice-topic"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && generate()}
            placeholder="topic — e.g. definite integrals"
            className="max-w-64"
          />
          <div className="flex overflow-hidden rounded-md border">
            {(["JEE Main", "JEE Advanced"] as const).map((d) => (
              <button
                key={d}
                onClick={() => setDifficulty(d)}
                className={`px-3 py-1.5 text-sm ${
                  difficulty === d
                    ? "bg-primary text-primary-foreground"
                    : "bg-card text-muted-foreground"
                }`}
              >
                {d}
              </button>
            ))}
          </div>
          <Button onClick={generate} disabled={busy || !topic.trim()}>
            <FlaskConical className="mr-1.5 h-4 w-4" />
            {busy ? "Generating…" : "Generate"}
          </Button>
        </div>
        {status && <p className="font-mono2 text-xs text-muted-foreground">{status}</p>}
        {problem && (
          <div className="flex flex-col gap-3">
            <div className="border-l-2 border-primary pl-4 text-[15px]">{problem.problem}</div>
            <div className="flex items-center gap-2">
              <Badge
                className={`border-transparent ${problem.sympy_verified ? "bg-ok-soft text-ok" : "bg-warn-soft text-warn"} hover:bg-ok-soft`}
              >
                {problem.sympy_verified ? "answer key machine-verified" : "answer key unverified"}
              </Badge>
              <Badge variant="secondary" className="font-mono2 font-normal">
                {problem.difficulty}
              </Badge>
            </div>
            {!revealed ? (
              <Button variant="outline" className="self-start" onClick={() => setRevealed(true)}>
                Reveal answer & solution
              </Button>
            ) : (
              <div className="flex flex-col gap-2">
                {problem.steps.map((s) => (
                  <div key={s.step_number} className="border-l-2 border-border pl-4">
                    <p className="text-sm font-semibold">
                      {s.step_number}. {s.description}
                    </p>
                    {s.formula && (
                      <p className="my-0.5 overflow-x-auto">
                        <Formula tex={s.formula} />
                      </p>
                    )}
                    <p className="text-sm text-muted-foreground">{s.explanation}</p>
                  </div>
                ))}
                <div className="rounded-md bg-ok-soft px-4 py-2.5 font-bold">
                  {problem.final_answer}
                </div>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

/* ---------- app shell ---------- */

export default function App() {
  const [kb, setKb] = useState<number | null>(null);
  const [cal, setCal] = useState<number | null>(null);

  useEffect(() => {
    fetch("/api/v1/health")
      .then((r) => r.json())
      .then((d) => setKb(d.kb_entries))
      .catch(() => {});
    fetch("/api/v1/calibration")
      .then((r) => r.json())
      .then((d) => d.verified_accuracy != null && setCal(d.verified_accuracy))
      .catch(() => {});
  }, []);

  return (
    <div className="mx-auto max-w-6xl px-4 pb-16 pt-6">
      <header className="mb-6 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="font-display text-2xl font-bold tracking-tight">
            <Sparkles className="mr-2 inline h-5 w-5 text-primary" />
            Math Agent
          </h1>
          <p className="text-sm text-muted-foreground">
            Claude Opus 5 · SymPy-verified, step-checked solutions
          </p>
        </div>
        <div className="flex gap-4 font-mono2 text-xs text-muted-foreground">
          {kb != null && (
            <span className="flex items-center gap-1.5">
              <BookOpenCheck className="h-3.5 w-3.5" /> {kb} KB problems
            </span>
          )}
          <span className="flex items-center gap-1.5">
            <Wrench className="h-3.5 w-3.5" /> 12 tools
          </span>
          {cal != null && (
            <span className="flex items-center gap-1.5 text-ok">
              <LineChart className="h-3.5 w-3.5" /> verified acc. {Math.round(cal * 100)}%
            </span>
          )}
        </div>
      </header>

      <Tabs defaultValue="solve">
        <TabsList className="mb-4">
          <TabsTrigger value="solve">Solve</TabsTrigger>
          <TabsTrigger value="tutor">Tutor</TabsTrigger>
          <TabsTrigger value="practice">Practice</TabsTrigger>
        </TabsList>
        <TabsContent value="solve">
          <SolvePanel />
        </TabsContent>
        <TabsContent value="tutor">
          <TutorPanel />
        </TabsContent>
        <TabsContent value="practice">
          <PracticePanel />
        </TabsContent>
      </Tabs>
    </div>
  );
}
