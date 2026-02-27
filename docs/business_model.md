# Winnow Business Model

Analysis based on 2,894 real requests from analytics.db (2026-02-23 to 2026-02-25, 3 clients), competitive landscape research, arxiv paper deep-read, and dev tool pricing study (Feb 2026).

## 0. Executive summary

**Is this innovative?** Yes. Winnow is the only neural, task-conditioned, code-specific, line-level pruner. The killer stat: 87.3% AST correctness vs 0.29% for LLMLingua-2 (the most widely-used alternative). It's also the only compression method that *improves* agent success rate (64% vs 62% baseline on SWE-Bench) while cutting tokens. No competitor offers this as a hosted service today.

**Is anyone else doing this?** Not exactly. Compresr (YC W26) does general-purpose context compression but not code-specific. LLMLingua is widely integrated (LangChain, LlamaIndex) but destroys code syntax. LongCodeZip is academically close but not task-conditioned and not offered as a service. Adjacent tools (Cursor, Sourcegraph, Augment) do retrieval, not compression. There is a real market gap.

**Does the business make sense?** Yes, but not as a standalone $15/mo subscription. Dev tool subscription fatigue is acute (39% planned cancellations in 2025). The stronger paths are: (1) API-first usage-based pricing embedded into agent workflows via MCP, (2) B2B enterprise licensing to teams spending $1000s+/mo on LLM inference, (3) licensing to IDE/agent platforms as a built-in feature.

**30-day investment:** $23 (just you) to $684 (100 trial users). Manageable.

## 0.1 Competitive landscape

### Direct competitors

| Competitor | What | Code-specific? | Task-conditioned? | Line-level? | Hosted API? |
|-----------|------|---------------|-------------------|------------|------------|
| **Compresr** (YC W26) | General context compression API | No | Unknown | No | Yes (early) |
| **TinyToken** | Conversation/prompt compression API | No | No | No | Yes |
| **LLMLingua** (Microsoft) | PPL-based token-level compression | No | Partially (LongLLMLingua) | No (token-level) | OSS only |
| **LongCodeZip** (ASE 2025) | PPL-based code compression | Yes | No | Block-level | OSS only |
| **Winnow** | Neural code pruning | **Yes** | **Yes** | **Yes** | **Building** |

**Winnow is unique on all four dimensions.** No competitor is code-specific + task-conditioned + line-level + available as a hosted API.

### Adjacent tools (retrieval, not compression)

Cursor, Sourcegraph Cody, Augment Code, Continue.dev, Aider all do context *retrieval* (find relevant code). None do context *compression* (prune irrelevant lines from already-retrieved code). These are complementary: Winnow operates downstream of retrieval.

### Biggest threats

1. **Compresr (YC W26):** Well-funded, PhD founder in compression. General-purpose today but could pivot to code.
2. **LLMLingua in frameworks:** Default "good enough" for teams that just need something. Inertia advantage.
3. **LLM context windows growing:** If context windows go to 10M+ tokens cheaply, the need for pruning shrinks (but latency and cost still matter).

## 0.2 Technical assessment

### Innovation: genuine, combination is novel

The individual pieces (CRF sequence labeling, multi-layer fusion, reranker backbone) are known techniques. The combination applied to code pruning for agentic workflows is new and well-validated:

- **Only method that improves agent success rate** while reducing tokens (64% vs 62% baseline on SWE-Bench)
- **87.3% AST correctness** vs 0.29% for LLMLingua-2, 12.4% for Selective-Context
- **14.84x compression** at best accuracy on LongCodeQA (vs 5.87x for RAG, 7.39x for LongCodeZip)
- **102ms TTFT** at 8192 tokens (7.5x faster than using a 32B LLM as compressor)

### Replication difficulty: medium-high

- Training data pipeline requires Qwen3-Coder-30B + Qwen3-Next-80B for annotation (~$20K+ compute)
- 61K training samples with 1/6 retention rate from quality filtering
- Backbone choice (Qwen3-Reranker) matters; code pre-training is critical
- Weights and code are open source, so someone *could* replicate, but the know-how and evaluation infrastructure take months

### Limitations

- **Python-only** training data. Unknown generalization to other languages.
- **Goal hint quality matters.** Bad queries produce bad pruning. No robustness analysis provided.
- **Single-worker inference.** Not production-ready for scale without batching.

## 1. Deployment: Modal with L4 GPU

**Why Modal:** Keeps FastAPI native (`@modal.asgi_app()`), scale-to-zero, minimal code changes.

**Why L4 (not T4):** Model uses ~9GB VRAM despite being "0.6B parameters" due to multi-layer fusion (3x hidden states in float32), 8K sequence KV cache, and Flash Attention 2 overhead. T4's 16GB leaves only 7GB headroom, risking OOM on long sequences. L4 (24GB) is comfortable.

| GPU | VRAM | Modal $/hr | Verdict |
|-----|------|-----------|---------|
| T4 | 16GB | ~$0.59 | Too tight, OOM risk |
| **L4** | **24GB** | **~$1.10** | Recommended |
| A10G | 24GB | ~$1.10 | Alternative, same tier |

## 2. Usage patterns (from real data)

### Client profiles

| Client | Requests | Avg input tokens | Avg saved | Savings % | Avg latency |
|--------|----------|-----------------|-----------|-----------|-------------|
| `unknown` (MCP/IDE) | 1,844 | 2,179 | 1,361 | 62.5% | 2.4s |
| `codex` (agent) | 1,066 | 3,051 | 2,262 | 74.1% | 2.6s |

*Note: `codex` also sent 7 monster requests (100K-38M tokens) that are excluded from the table above. Those alone cost $3.29 of the $5.47 total.*

### Request size distribution

| Bucket | % of traffic | Avg tokens | Avg latency |
|--------|-------------|------------|-------------|
| < 1K | 59.5% | 330 | 2.3s |
| 1K-5K | 24.3% | 2,115 | 2.3s |
| 5K-10K | 11.4% | 7,031 | 2.6s |
| 10K-25K | 4.2% | 16,103 | 4.3s |
| 25K-100K | 0.7% | ~50K | 7-11s |
| 100K+ | 0.2% | ~11M | 25-90min |

**Key observation:** 84% of requests are under 5K tokens. Latency has a ~2.3 second floor regardless of input size (GPU kernel launch, model forward pass setup, CUDA overhead).

### Latency percentiles

| p50 | p95 | p99 |
|-----|-----|-----|
| 2.2s | 4.3s | 6.4s |

## 3. Cost structure (the critical insight)

### Per-request cost is stable; per-token cost is NOT

The ~2.3s latency floor means every request costs ~$0.0007 in GPU time regardless of whether it processes 300 tokens or 5,000. This makes per-token cost highly variable:

| Bucket | Cost/request | Cost/M input tokens |
|--------|-------------|-------------------|
| < 1K tokens | $0.0007 | **$2.14** |
| 1K-5K | $0.0007 | **$0.33** |
| 5K-10K | $0.0008 | $0.11 |
| 10K-25K | $0.0013 | $0.08 |
| 25K-50K | $0.002 | $0.06 |
| 50K-100K | $0.0035 | $0.05 |

**The cost per request is stable ($0.0007-0.0035). The cost per million tokens varies 43x.** This means pure per-token pricing is dangerous: if most users send small files (likely for MCP/IDE use), cost per M tokens is $0.33-2.14, not the blended $0.07 that large-request outliers dilute it to.

### Total cost snapshot

| Metric | Value |
|--------|-------|
| Total requests (2.2 days) | 2,894 |
| Total GPU time | 4.97 hours |
| Total cost at L4 | $5.47 |
| Cost from 100K+ monsters (7 reqs) | $3.29 (60%) |
| Cost from everything else (2,887 reqs) | $2.19 (40%) |
| Avg cost per normal request | $0.00076 |

## 4. Value to users

Winnow saves tokens that would otherwise go to LLM APIs. The value depends on whether the user hits cached or uncached pricing:

### Without prompt caching (first-turn, cold context)

| LLM | Input price/M | Avg tokens saved/req | User saves/req |
|-----|--------------|---------------------|---------------|
| Claude Sonnet 4 | $3/M | 1,800 | $0.0054 |
| Claude Opus 4 | $15/M | 1,800 | $0.027 |
| GPT-4o | $2.50/M | 1,800 | $0.0045 |

### With prompt caching (repeat context, multi-turn)

All major LLM providers offer prompt caching: Anthropic at 90% discount ($0.30/M cached), OpenAI at 50% ($1.25/M cached). If pruned tokens would have been cached, the savings shrink:

| LLM | Cached price/M | Avg tokens saved/req | User saves/req |
|-----|---------------|---------------------|---------------|
| Claude Sonnet 4 (cached) | $0.30/M | 1,800 | $0.00054 |
| Claude Opus 4 (cached) | $1.50/M | 1,800 | $0.0027 |
| GPT-4o (cached) | $1.25/M | 1,800 | $0.00225 |

**The sweet spot for Winnow is first-turn and cold-context reads**, where the full input price applies. In multi-turn conversations where caching kicks in, the value drops 5-10x. However, coding agents frequently read *new* files (exploring codebases, grep results, error traces), which are not cacheable. The pruner is most valuable in exactly this agentic exploration pattern.

For a developer making 100 prune calls/day on Sonnet (assuming 70% cold reads, 30% cached):
- Cold: 70 * $0.0054 = $0.378/day
- Cached: 30 * $0.00054 = $0.016/day
- **Saves ~$12/month** blended

## 5. Pricing model

### What doesn't work (and why)

**Per-token:** The 2.3s latency floor makes small requests ($2.14/M) 43x more expensive per-token than large ones ($0.05/M). Unpredictable bills.

**Per-request:** Real users do 300-1000 req/day. At $0.005/req that's $45-150/mo. A medium dev on Sonnet only saves ~$12/mo from pruning. They'd pay more for the pruner than they save.

**Standalone subscription ($15/mo):** Dev tool subscription fatigue is acute. 39% of subscribers planned cancellations in 2025. Developers already pay $60-80/mo for Copilot + Cursor + Claude Pro. A 5th subscription for "context optimization" (an invisible backend improvement) is a hard sell. Windsurf at $15/mo works because it's an *IDE* -- the primary tool, not an add-on.

### Recommended: multi-channel approach

Don't pick one model. Serve three audiences differently:

**Channel 1: Hosted API (individual devs)**

The path of least resistance. Developers using Claude Code, OpenHands, or agent frameworks already configure MCP servers. A hosted MCP endpoint is a natural fit.

| Tier | Price | Includes | Rate limit |
|------|-------|----------|-----------|
| **Trial** | **Free, 14 days** | Unlimited | 2 req/sec |
| **Pro** | **$15/mo** | Unlimited | 5 req/sec |

$15/mo is validated by Windsurf Pro, Tabnine Pro ($12), Continue.dev Team ($10). The value pitch is not just "save $12/mo on tokens" but also: faster agent runs (18-26% fewer rounds), better results (64% vs 62% on SWE-Bench), and context window headroom.

**Channel 2: B2B enterprise API (the real money)**

Teams running agent fleets (10+ devs, automated CI/CD with LLM agents) spend $1,000-10,000+/mo on LLM inference. A 23-38% token reduction is $230-3,800/mo in direct savings. Sell to engineering leads who see the invoice.

| Tier | Price | Includes |
|------|-------|----------|
| **Team** | **$49/mo** | 10 API keys, 10 req/sec per key |
| **Enterprise** | **Custom ($200-500/mo)** | Unlimited keys, SLA, priority queue, dedicated capacity |

**Channel 3: License to platforms**

Cursor, Continue.dev, OpenHands, or agent framework companies could embed Winnow as a feature. License the model + API for a per-seat royalty or flat monthly fee. This is the highest-leverage play but requires business development.

### Recommended pricing (for launch)

### Unit economics per tier

Based on real usage data: active users average 300-500 req/day, power users hit 1,000-1,600 req/day.

| User type | Req/day | GPU cost/day | GPU cost/mo | Revenue/mo | Margin |
|-----------|---------|-------------|-------------|-----------|--------|
| Light (300 req/day) | 300 | $0.23 | $6.90 | $15 | 54% |
| Average (500 req/day) | 500 | $0.38 | $11.40 | $15 | 24% |
| Power user (1,000 req/day) | 1,000 | $0.76 | $22.80 | $15 | -52% |
| Team avg (5 devs, 2,500 total) | 2,500 | $1.90 | $57 | $49 | -16% |

**The power user problem:** Heavy users cost more than $15/mo to serve. This is normal for flat-rate SaaS. The key is that most users are light-to-average, and the rate limit (5 req/sec = max ~18K req/hr) prevents extreme abuse. The distribution subsidizes itself: many light users at 54% margin cover the few power users at -52%.

**Break-even user mix:** If 60% of users are light (300 req/day) and 40% are average (500 req/day):
- Blended cost: 0.6 * $6.90 + 0.4 * $11.40 = **$8.70/mo per user**
- Revenue: $15/mo
- **Blended margin: 42%**

Even with 20% power users mixed in: 0.4 * $6.90 + 0.4 * $11.40 + 0.2 * $22.80 = **$11.88/mo**, still 21% margin at $15.

### Rate limits as cost control

The rate limit is the real guardrail, not a daily cap. It naturally bounds cost while feeling unlimited to humans:

| Rate limit | Max req/hr | Max req/day (8hr active) | Max GPU cost/day |
|-----------|-----------|------------------------|-----------------|
| 2/sec (trial) | 7,200 | ~57,600 | $43.78 |
| 5/sec (pro) | 18,000 | ~144,000 | $109.44 |
| 10/sec (team, per key) | 36,000 | ~288,000 | $218.88 |

In practice, no human coding session sustains the rate limit. Real peak is 264 req/hr (from analytics data). The rate limit only throttles runaway scripts or agent loops.

### 30-day launch investment

| Scenario | Users | Avg req/day/user | GPU cost/day | **30-day total** |
|----------|-------|-----------------|-------------|-----------------|
| Just you testing | 1 | 1,000 | $0.76 | **$23** |
| Soft launch (10 devs on trial) | 10 | 500 | $3.80 | **$114** |
| HN/Reddit post (100 on trial) | 100 | 300 | $22.80 | **$684** |
| Viral launch (1,000 on trial) | 1,000 | 200 | $152 | **$4,560** |

Modal's $30/mo free credits offset the first ~$30. During the 14-day trial, all users are free. Revenue starts day 15 as trials convert.

**Trial conversion math:** At 100 trial users, if 20% convert (industry average for dev tools with good onboarding):
- 20 paying Pro users * $15/mo = $300/mo recurring
- GPU cost for 100 users (including free): ~$684/mo (first month), drops to ~$450/mo after non-converters churn
- Break-even by month 2-3

### Revenue projections (subscription)

| DAU | Paying users (20% of DAU) | Monthly revenue | Monthly GPU cost (all users) | Gross profit | Margin |
|-----|--------------------------|----------------|------------------------------|-------------|--------|
| 100 | 20 | $300 | $684 | -$384 | -128% |
| 500 | 100 | $1,500 | $1,900 | -$400 | -27% |
| 1,000 | 200 | $3,000 | $3,800 | -$800 | -27% |
| 2,000 | 400 | $6,000 | $4,560 | $1,440 | 24% |
| 5,000 | 1,000 | $15,000 | $11,400 | $3,600 | 24% |

**Profitability threshold: ~1,500 DAU** (with 20% conversion rate and 300 req/day average for free users).

The free trial users are the cost drag. Two ways to improve:
1. **Shorter trial (7 days instead of 14):** Cuts trial GPU cost in half
2. **Lower free user rate limit (1 req/sec):** Naturally limits free user consumption
3. **Request batching (Section 6):** Cuts GPU cost 3-4x across the board, making profitability possible at ~500 DAU

### Throughput and scaling

At 2.3s per request, one GPU handles ~1,560 req/hr = ~37,400 req/day. To handle peak traffic:

| DAU | Total req/day | Peak multiplier (3x avg) | GPUs needed (peak) |
|-----|--------------|--------------------------|-------------------|
| 1,000 | 350,000 | 43,750 req/hr | 28 |
| 5,000 | 1,750,000 | 218,750 req/hr | 140 |
| 10,000 | 3,500,000 | 437,500 req/hr | 280 |

Modal auto-scales containers, so this is handled automatically. Cost scales linearly. Batching (Section 6) would cut GPU count by 3-4x.

## 6. Levers to reduce cost further

### Batching (biggest win)

Current: `ThreadPoolExecutor(max_workers=1)`, one request at a time. The GPU sits idle during tokenization and post-processing.

If we batch 4-8 small requests into a single forward pass:
- GPU utilization increases 3-4x
- Cost per request drops to $0.0002-0.0003
- Margin at $0.005/req jumps to 94-96%

### Quantization

Running the backbone in INT8 or INT4 instead of fp16/fp32 could cut VRAM from ~9GB to 4-5GB, enabling T4 ($0.59/hr instead of $1.10). Halves infrastructure cost.

### Caching

Many devs prune the same file with different queries. Cache the backbone hidden states keyed on file hash, only rerun the fusion + CRF head for new queries. Could reduce latency by 60-70% for repeat files.

## 7. Guardrails

| Rule | Reason |
|------|--------|
| Max 100K tokens/request | Prevents $1+ monster requests |
| Rate limit: 2/sec trial, 5/sec pro, 10/sec team | Cost control without daily caps. No human hits this. |
| 14-day trial expiry | Forces conversion decision. Free users cost ~$0.23-0.38/day. |
| Request timeout: 30s | Kills runaway chunked requests |
| Min input: 50 tokens | Prevents trivial requests wasting GPU time |
| GitHub/email auth for API keys | Prevents bot signups for infinite free trials |

## 8. Risks

| Risk | Severity | Impact | Mitigation |
|------|----------|--------|-----------|
| **Compresr (YC W26)** | High | Well-funded competitor with PhD founder. General-purpose today but could pivot to code-specific. | Move fast. Ship the hosted API before they do. First-mover advantage in code-specific pruning. |
| **Prompt caching adoption** | Medium | LLM providers discount cached tokens 50-90%. Weakens token savings value 5-10x for repeat context. | Market to agentic use cases (cold file reads, exploration) where caching doesn't apply. Lead with "better results + faster runs", not just "cheaper tokens". |
| **LLM input prices drop** | Medium | Input tokens get cheaper over time, reducing savings value. | Pruning also reduces latency, agent rounds (18-26% fewer), and context window pressure. Shift messaging. |
| **Context windows grow** | Medium | If 10M+ token windows become cheap, raw need for pruning shrinks. | Larger windows amplify noise. Pruning improves signal-to-noise, not just fits in window. The SWE-Bench improvement (64% vs 62%) shows pruning helps accuracy, not just cost. |
| **Subscription fatigue** | Medium | Developers resist another $15/mo tool. 39% planned cancellations in 2025. | Multi-channel: MCP integration (feels like a feature, not a product), B2B enterprise (invoice savings speak), platform licensing. |
| **Python-only model** | Medium | Unknown generalization to TS, Java, Go, Rust. | Test on multi-language benchmarks. If needed, fine-tune on multi-language data (61K samples is small, retraining is cheap). |
| **Single-threaded bottleneck** | Low | One GPU handles ~37K req/day. | Modal auto-scales. Batching (Section 6) cuts GPU needs 3-4x. |
| **Open source model weights** | Low | Anyone can self-host for free. | Self-hosting requires CUDA + 9GB VRAM + ops overhead. The hosted service value is convenience + no GPU cost. Most devs won't self-host. |

## 9. Implementation priorities

### Phase 1: Ship the hosted API (week 1-2)
1. **100K token cap** in `prune_wrapper.py` (reject oversized inputs before GPU inference)
2. **Auth + rate limiting** in `online_serving.py` (API key auth, per-key rate limit via token bucket)
3. **Modal deployment script** (`deploy/modal_app.py`) with L4 GPU, scale-to-zero
4. **Usage dashboard** -- surface the existing `/stats` endpoint as a web page showing tokens saved, requests, estimated $ saved

### Phase 2: Monetize (week 3-4)
5. **Stripe integration** (subscription billing, 14-day trial, Pro/Team tiers)
6. **Landing page** with the killer stats: "87.3% AST correctness vs 0.29% for LLMLingua", "64% vs 62% on SWE-Bench", "23-54% token savings"
7. **Remote mode for existing MCP server** -- add a `--remote <url>` flag so the existing MCP server can call the hosted `/prune` API instead of local GPU inference. Same MCP interface, Claude Code doesn't know the difference. Users without a GPU can use the service.

### Phase 3: Scale (month 2+)
8. **Request batching** (biggest cost reduction, 3-4x GPU efficiency)
9. **Multi-language evaluation** (TypeScript, Java, Go, Rust) and fine-tuning if needed
10. **B2B outreach** to teams running agent fleets (OpenHands deployments, Claude Agent SDK users)
11. **Platform licensing** conversations with Cursor, Continue.dev, OpenHands

### Phase 4: Moat (month 3+)
12. **INT8 quantization** to enable T4 deployment (halves GPU cost)
13. **Hidden state caching** for repeat files (60-70% latency reduction)
14. **Multi-language fine-tuning** with expanded training data
15. **Cost tracking** in analytics.db for real-time margin monitoring
