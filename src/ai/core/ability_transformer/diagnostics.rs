//! Model diagnostic capture and HTML report generation.
//!
//! Provides attention weight capture during inference and generates
//! self-contained HTML reports with:
//!   - Attention heatmaps per layer per head (self-attention + cross-attention)
//!   - Entity encoder attention patterns
//!   - Action logit distributions (bar charts)
//!   - Pointer distributions (V3)
//!   - Token embedding similarity matrices
//!
//! Usage:
//! ```ignore
//! let model = ActorCriticWeightsV3::from_json(&json_str)?;
//! let diag = model.diagnose_v3(&entities, &types, &threats, &positions, &ability_token_ids);
//! std::fs::write("report.html", diag.to_html(&token_labels))?;
//! ```

use serde::Serialize;

// ---------------------------------------------------------------------------
// Capture structs
// ---------------------------------------------------------------------------

/// Attention weights from a single multi-head attention layer.
#[derive(Debug, Clone, Serialize)]
pub struct AttentionCapture {
    /// Layer index.
    pub layer: usize,
    /// `[n_heads][query_len][key_len]` — attention probabilities (post-softmax).
    pub weights: Vec<Vec<Vec<f32>>>,
}

/// Cross-attention weights from ability CLS attending to entity tokens.
#[derive(Debug, Clone, Serialize)]
pub struct CrossAttentionCapture {
    /// Which ability slot (0-7).
    pub ability_slot: usize,
    /// `[n_heads][key_len]` — attention weights (query is single CLS token).
    pub weights: Vec<Vec<f32>>,
}

/// Complete diagnostic capture from a single inference pass.
#[derive(Debug, Clone, Serialize)]
pub struct DiagnosticCapture {
    // -- Architecture info --
    pub d_model: usize,
    pub n_heads: usize,

    // -- Ability transformer --
    /// Token IDs fed into the transformer.
    pub token_ids: Vec<u32>,
    /// Per-layer self-attention weights `[n_layers]`.
    pub transformer_attention: Vec<AttentionCapture>,
    /// CLS embedding after the transformer `[d_model]`.
    pub cls_embedding: Vec<f32>,

    // -- Entity encoder --
    /// Labels for entities/threats/positions in the entity sequence.
    pub entity_labels: Vec<String>,
    /// Entity type IDs for the full sequence.
    pub entity_type_ids: Vec<usize>,
    /// Per-layer entity encoder self-attention weights.
    pub entity_attention: Vec<AttentionCapture>,
    /// Entity token embeddings after encoder `[n_tokens][d_model]`.
    pub entity_embeddings: Vec<Vec<f32>>,
    /// Pooled entity state `[d_model]`.
    pub pooled_state: Vec<f32>,

    // -- Cross-attention --
    /// Cross-attention weights per ability slot.
    pub cross_attention: Vec<CrossAttentionCapture>,

    // -- Decision outputs --
    /// Action type logits `[n_action_types]`.
    pub action_type_logits: Vec<f32>,
    /// Attack pointer logits `[n_tokens]`.
    pub attack_pointer: Vec<f32>,
    /// Move pointer logits `[n_tokens]`.
    pub move_pointer: Vec<f32>,
    /// Per-ability pointer logits.
    pub ability_pointers: Vec<Option<Vec<f32>>>,
    /// Value head output.
    pub value: f32,
}

// ---------------------------------------------------------------------------
// HTML report generation
// ---------------------------------------------------------------------------

impl DiagnosticCapture {
    /// Generate a self-contained HTML diagnostic report.
    ///
    /// `token_labels` maps token IDs to human-readable labels.
    pub fn to_html(&self, token_labels: &[&str]) -> String {
        let mut html = String::with_capacity(64 * 1024);

        // Header
        html.push_str(HTML_HEADER);

        // Serialize capture data as JSON for JavaScript consumption
        html.push_str("<script>\nconst DIAG = ");
        // Build a JSON-friendly version with labels
        let labels: Vec<String> = self
            .token_ids
            .iter()
            .map(|&id| {
                if (id as usize) < token_labels.len() {
                    token_labels[id as usize].to_string()
                } else {
                    format!("?{id}")
                }
            })
            .collect();
        html.push_str(&serde_json::to_string(&DiagJson {
            d_model: self.d_model,
            n_heads: self.n_heads,
            token_labels: &labels,
            entity_labels: &self.entity_labels,
            entity_type_ids: &self.entity_type_ids,
            transformer_attention: &self.transformer_attention,
            entity_attention: &self.entity_attention,
            cross_attention: &self.cross_attention,
            cls_embedding: &self.cls_embedding,
            entity_embeddings: &self.entity_embeddings,
            pooled_state: &self.pooled_state,
            action_type_logits: &self.action_type_logits,
            attack_pointer: &self.attack_pointer,
            move_pointer: &self.move_pointer,
            ability_pointers: &self.ability_pointers,
            value: self.value,
        }).unwrap_or_else(|_| "{}".to_string()));
        html.push_str(";\n</script>\n");

        // JavaScript visualization code
        html.push_str(HTML_BODY);

        html
    }
}

#[derive(Serialize)]
struct DiagJson<'a> {
    d_model: usize,
    n_heads: usize,
    token_labels: &'a [String],
    entity_labels: &'a [String],
    entity_type_ids: &'a [usize],
    transformer_attention: &'a [AttentionCapture],
    entity_attention: &'a [AttentionCapture],
    cross_attention: &'a [CrossAttentionCapture],
    cls_embedding: &'a [f32],
    entity_embeddings: &'a [Vec<f32>],
    pooled_state: &'a [f32],
    action_type_logits: &'a [f32],
    attack_pointer: &'a [f32],
    move_pointer: &'a [f32],
    ability_pointers: &'a [Option<Vec<f32>>],
    value: f32,
}

// ---------------------------------------------------------------------------
// HTML template
// ---------------------------------------------------------------------------

const HTML_HEADER: &str = r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Model Diagnostic Report</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0d1117; color: #c9d1d9; padding: 20px; }
h1 { color: #58a6ff; margin-bottom: 8px; font-size: 1.6em; }
h2 { color: #79c0ff; margin: 24px 0 12px; font-size: 1.2em; border-bottom: 1px solid #21262d; padding-bottom: 6px; }
h3 { color: #8b949e; margin: 12px 0 8px; font-size: 1em; }
.section { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 16px; margin-bottom: 16px; }
.grid { display: flex; flex-wrap: wrap; gap: 12px; }
.card { background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 12px; }
canvas { image-rendering: pixelated; border: 1px solid #30363d; border-radius: 4px; }
.bar-chart { display: flex; align-items: flex-end; gap: 2px; height: 120px; padding: 4px; }
.bar { background: #388bfd; border-radius: 2px 2px 0 0; min-width: 18px; position: relative; transition: background 0.15s; cursor: default; }
.bar:hover { background: #58a6ff; }
.bar-label { position: absolute; bottom: -18px; left: 50%; transform: translateX(-50%); font-size: 9px; color: #8b949e; white-space: nowrap; }
.bar-value { position: absolute; top: -16px; left: 50%; transform: translateX(-50%); font-size: 9px; color: #c9d1d9; white-space: nowrap; }
.bar.negative { background: #f85149; }
.metric { display: inline-block; background: #21262d; border-radius: 4px; padding: 4px 10px; margin: 2px 4px; font-size: 0.9em; }
.metric .label { color: #8b949e; }
.metric .value { color: #58a6ff; font-weight: 600; }
.tab-row { display: flex; gap: 4px; margin-bottom: 8px; }
.tab { padding: 4px 12px; border-radius: 4px; cursor: pointer; background: #21262d; color: #8b949e; font-size: 0.85em; border: 1px solid transparent; }
.tab.active { background: #388bfd22; color: #58a6ff; border-color: #388bfd; }
.tooltip { position: absolute; background: #1c2128; border: 1px solid #444c56; border-radius: 4px; padding: 4px 8px; font-size: 11px; pointer-events: none; z-index: 10; }
.entity-type-0 { color: #56d364; } /* self */
.entity-type-1 { color: #f85149; } /* enemy */
.entity-type-2 { color: #388bfd; } /* ally */
.entity-type-3 { color: #d29922; } /* threat */
.entity-type-4 { color: #bc8cff; } /* position */
.hidden { display: none; }
</style>
</head>
<body>
<h1>Model Diagnostic Report</h1>
"##;

const HTML_BODY: &str = r##"
<script>
// =========================================================================
// Utility functions
// =========================================================================

function softmax(logits) {
    const max = Math.max(...logits.filter(v => isFinite(v)));
    const exps = logits.map(v => isFinite(v) ? Math.exp(v - max) : 0);
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(v => sum > 0 ? v / sum : 0);
}

function colorScale(v, min, max) {
    const t = max > min ? (v - min) / (max - min) : 0;
    // Blue-white-red diverging scale
    if (t < 0.5) {
        const s = t * 2;
        return `rgb(${Math.round(13 + s*200)}, ${Math.round(17 + s*200)}, ${Math.round(23 + s*200)})`;
    } else {
        const s = (t - 0.5) * 2;
        return `rgb(${Math.round(213 + s*42)}, ${Math.round(217 - s*150)}, ${Math.round(223 - s*180)})`;
    }
}

function heatColor(v) {
    // 0=dark, 1=bright yellow-white
    const r = Math.round(Math.min(255, v * 400));
    const g = Math.round(Math.min(255, v * 300));
    const b = Math.round(Math.min(255, v * 100));
    return `rgb(${r},${g},${b})`;
}

function typeColor(typeId) {
    return ['#56d364','#f85149','#388bfd','#d29922','#bc8cff'][typeId] || '#8b949e';
}

const ACTION_TYPE_NAMES = [
    'Attack', 'Move', 'Hold',
    'Ability 0', 'Ability 1', 'Ability 2', 'Ability 3',
    'Ability 4', 'Ability 5', 'Ability 6', 'Ability 7'
];

// =========================================================================
// Render attention heatmap onto a canvas
// =========================================================================

function renderHeatmap(canvas, weights, rowLabels, colLabels, title) {
    const nRows = weights.length;
    const nCols = weights[0]?.length || 0;
    if (nRows === 0 || nCols === 0) return;

    const cellSize = Math.max(12, Math.min(32, Math.floor(400 / Math.max(nRows, nCols))));
    const labelW = 80;
    const labelH = 60;
    const w = labelW + nCols * cellSize;
    const h = labelH + nRows * cellSize;

    canvas.width = w;
    canvas.height = h;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, w, h);

    // Title
    ctx.fillStyle = '#c9d1d9';
    ctx.font = '11px monospace';
    ctx.fillText(title, 4, 12);

    // Column labels (rotated)
    ctx.save();
    ctx.font = '9px monospace';
    ctx.fillStyle = '#8b949e';
    for (let c = 0; c < nCols; c++) {
        const x = labelW + c * cellSize + cellSize / 2;
        ctx.save();
        ctx.translate(x, labelH - 4);
        ctx.rotate(-Math.PI / 4);
        const label = colLabels[c] || '';
        ctx.fillText(label.length > 8 ? label.slice(0, 8) : label, 0, 0);
        ctx.restore();
    }
    ctx.restore();

    // Row labels
    ctx.font = '9px monospace';
    ctx.fillStyle = '#8b949e';
    for (let r = 0; r < nRows; r++) {
        const y = labelH + r * cellSize + cellSize / 2 + 3;
        const label = rowLabels[r] || '';
        ctx.fillText(label.length > 10 ? label.slice(0, 10) : label, 2, y);
    }

    // Cells
    for (let r = 0; r < nRows; r++) {
        for (let c = 0; c < nCols; c++) {
            const v = weights[r][c];
            ctx.fillStyle = heatColor(v);
            ctx.fillRect(labelW + c * cellSize, labelH + r * cellSize, cellSize - 1, cellSize - 1);
        }
    }

    // Tooltip handler
    canvas.onmousemove = (e) => {
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        const c = Math.floor((mx - labelW) / cellSize);
        const r = Math.floor((my - labelH) / cellSize);
        let tip = canvas.parentElement.querySelector('.tooltip');
        if (!tip) { tip = document.createElement('div'); tip.className = 'tooltip'; canvas.parentElement.style.position = 'relative'; canvas.parentElement.appendChild(tip); }
        if (r >= 0 && r < nRows && c >= 0 && c < nCols) {
            tip.textContent = `${rowLabels[r]} → ${colLabels[c]}: ${weights[r][c].toFixed(4)}`;
            tip.style.left = (mx + 10) + 'px';
            tip.style.top = (my - 20) + 'px';
            tip.style.display = 'block';
        } else {
            tip.style.display = 'none';
        }
    };
    canvas.onmouseleave = () => {
        const tip = canvas.parentElement.querySelector('.tooltip');
        if (tip) tip.style.display = 'none';
    };
}

// =========================================================================
// Render bar chart
// =========================================================================

function renderBarChart(container, values, labels, title) {
    container.innerHTML = '';
    const header = document.createElement('h3');
    header.textContent = title;
    container.appendChild(header);

    const probs = softmax(values);
    const chart = document.createElement('div');
    chart.className = 'bar-chart';
    chart.style.alignItems = 'flex-end';

    const maxProb = Math.max(...probs, 0.01);
    for (let i = 0; i < values.length; i++) {
        if (!isFinite(values[i]) || values[i] < -1e6) continue;
        const bar = document.createElement('div');
        bar.className = 'bar' + (values[i] < 0 ? ' negative' : '');
        const h = Math.max(2, (probs[i] / maxProb) * 100);
        bar.style.height = h + 'px';

        const label = document.createElement('span');
        label.className = 'bar-label';
        label.textContent = labels[i] || i;
        bar.appendChild(label);

        const val = document.createElement('span');
        val.className = 'bar-value';
        val.textContent = (probs[i] * 100).toFixed(1) + '%';
        bar.appendChild(val);

        chart.appendChild(bar);
    }
    container.appendChild(chart);
}

// =========================================================================
// Render embedding similarity matrix
// =========================================================================

function renderEmbeddingSimilarity(canvas, embeddings, labels, title) {
    const n = embeddings.length;
    if (n === 0) return;

    // Compute cosine similarity matrix
    const sims = [];
    for (let i = 0; i < n; i++) {
        sims[i] = [];
        for (let j = 0; j < n; j++) {
            let dot = 0, normA = 0, normB = 0;
            for (let k = 0; k < embeddings[i].length; k++) {
                dot += embeddings[i][k] * embeddings[j][k];
                normA += embeddings[i][k] * embeddings[i][k];
                normB += embeddings[j][k] * embeddings[j][k];
            }
            sims[i][j] = (normA > 0 && normB > 0) ? dot / (Math.sqrt(normA) * Math.sqrt(normB)) : 0;
        }
    }

    renderHeatmap(canvas, sims, labels, labels, title);
}

// =========================================================================
// Build the report
// =========================================================================

function buildReport() {
    const D = DIAG;
    const root = document.getElementById('report');

    // --- Summary metrics ---
    const summary = document.createElement('div');
    summary.className = 'section';
    summary.innerHTML = `<h2>Summary</h2>
        <span class="metric"><span class="label">d_model: </span><span class="value">${D.d_model}</span></span>
        <span class="metric"><span class="label">n_heads: </span><span class="value">${D.n_heads}</span></span>
        <span class="metric"><span class="label">tokens: </span><span class="value">${D.token_labels.length}</span></span>
        <span class="metric"><span class="label">entities: </span><span class="value">${D.entity_labels.length}</span></span>
        <span class="metric"><span class="label">value: </span><span class="value">${D.value.toFixed(4)}</span></span>
        <div style="margin-top:8px">
            <span class="metric"><span class="label">tokens: </span><span class="value" style="font-size:0.85em">${D.token_labels.join(' ')}</span></span>
        </div>
        <div style="margin-top:4px">
            <span class="metric"><span class="label">entities: </span><span class="value" style="font-size:0.85em">${
                D.entity_labels.map((l,i) => `<span class="entity-type-${D.entity_type_ids[i]}">${l}</span>`).join(' ')
            }</span></span>
        </div>`;
    root.appendChild(summary);

    // --- Action Distributions ---
    const actionSec = document.createElement('div');
    actionSec.className = 'section';
    actionSec.innerHTML = '<h2>Action Distributions</h2>';

    const typeChart = document.createElement('div');
    typeChart.className = 'card';
    renderBarChart(typeChart, D.action_type_logits, ACTION_TYPE_NAMES, 'Action Type Probabilities');
    actionSec.appendChild(typeChart);

    const ptrGrid = document.createElement('div');
    ptrGrid.className = 'grid';
    ptrGrid.style.marginTop = '12px';

    // Attack pointer
    const atkCard = document.createElement('div');
    atkCard.className = 'card';
    renderBarChart(atkCard, D.attack_pointer, D.entity_labels, 'Attack Target Pointer');
    ptrGrid.appendChild(atkCard);

    // Move pointer
    const mvCard = document.createElement('div');
    mvCard.className = 'card';
    renderBarChart(mvCard, D.move_pointer, D.entity_labels, 'Move Target Pointer');
    ptrGrid.appendChild(mvCard);

    // Ability pointers
    D.ability_pointers.forEach((ptr, i) => {
        if (!ptr) return;
        const card = document.createElement('div');
        card.className = 'card';
        renderBarChart(card, ptr, D.entity_labels, `Ability ${i} Target`);
        ptrGrid.appendChild(card);
    });

    actionSec.appendChild(ptrGrid);
    root.appendChild(actionSec);

    // --- Transformer Self-Attention ---
    if (D.transformer_attention.length > 0) {
        const attnSec = document.createElement('div');
        attnSec.className = 'section';
        attnSec.innerHTML = '<h2>Ability Transformer Self-Attention</h2>';

        D.transformer_attention.forEach(layerAttn => {
            const layerDiv = document.createElement('div');
            layerDiv.innerHTML = `<h3>Layer ${layerAttn.layer}</h3>`;
            const headGrid = document.createElement('div');
            headGrid.className = 'grid';

            layerAttn.weights.forEach((headWeights, headIdx) => {
                const card = document.createElement('div');
                card.className = 'card';
                const canvas = document.createElement('canvas');
                card.appendChild(canvas);
                headGrid.appendChild(card);
                renderHeatmap(canvas, headWeights, D.token_labels, D.token_labels, `Head ${headIdx}`);
            });

            layerDiv.appendChild(headGrid);
            attnSec.appendChild(layerDiv);
        });
        root.appendChild(attnSec);
    }

    // --- Entity Encoder Attention ---
    if (D.entity_attention.length > 0) {
        const entSec = document.createElement('div');
        entSec.className = 'section';
        entSec.innerHTML = '<h2>Entity Encoder Self-Attention</h2>';

        D.entity_attention.forEach(layerAttn => {
            const layerDiv = document.createElement('div');
            layerDiv.innerHTML = `<h3>Layer ${layerAttn.layer}</h3>`;
            const headGrid = document.createElement('div');
            headGrid.className = 'grid';

            layerAttn.weights.forEach((headWeights, headIdx) => {
                const card = document.createElement('div');
                card.className = 'card';
                const canvas = document.createElement('canvas');
                card.appendChild(canvas);
                headGrid.appendChild(card);
                renderHeatmap(canvas, headWeights, D.entity_labels, D.entity_labels, `Head ${headIdx}`);
            });

            layerDiv.appendChild(headGrid);
            entSec.appendChild(layerDiv);
        });
        root.appendChild(entSec);
    }

    // --- Cross-Attention ---
    if (D.cross_attention.length > 0) {
        const crossSec = document.createElement('div');
        crossSec.className = 'section';
        crossSec.innerHTML = '<h2>Cross-Attention (Ability CLS → Entities)</h2>';

        const crossGrid = document.createElement('div');
        crossGrid.className = 'grid';

        D.cross_attention.forEach(ca => {
            const card = document.createElement('div');
            card.className = 'card';
            const canvas = document.createElement('canvas');
            card.appendChild(canvas);
            crossGrid.appendChild(card);
            // ca.weights is [n_heads][n_entities] — reshape to [n_heads][1_query][n_entities] for heatmap
            const headWeights = ca.weights.map(hw => [hw]);
            const headLabels = ca.weights.map((_, i) => `H${i}`);
            // Or show as [n_heads x n_entities] matrix
            renderHeatmap(canvas, ca.weights, headLabels, D.entity_labels, `Ability ${ca.ability_slot}`);
        });

        crossSec.appendChild(crossGrid);
        root.appendChild(crossSec);
    }

    // --- Entity Embedding Similarity ---
    if (D.entity_embeddings.length > 0) {
        const embSec = document.createElement('div');
        embSec.className = 'section';
        embSec.innerHTML = '<h2>Entity Embedding Similarity (Cosine)</h2>';
        const canvas = document.createElement('canvas');
        const card = document.createElement('div');
        card.className = 'card';
        card.appendChild(canvas);
        embSec.appendChild(card);
        renderEmbeddingSimilarity(canvas, D.entity_embeddings, D.entity_labels, 'Entity Cosine Similarity');
        root.appendChild(embSec);
    }

    // --- CLS Embedding Histogram ---
    if (D.cls_embedding.length > 0) {
        const clsSec = document.createElement('div');
        clsSec.className = 'section';
        clsSec.innerHTML = '<h2>CLS Embedding Distribution</h2>';
        const card = document.createElement('div');
        card.className = 'card';

        // Simple histogram of CLS embedding values
        const canvas = document.createElement('canvas');
        canvas.width = 400;
        canvas.height = 120;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#0d1117';
        ctx.fillRect(0, 0, 400, 120);

        const vals = D.cls_embedding;
        const min = Math.min(...vals);
        const max = Math.max(...vals);
        const nBins = 30;
        const bins = new Array(nBins).fill(0);
        vals.forEach(v => {
            const bin = Math.min(nBins - 1, Math.floor((v - min) / (max - min + 1e-8) * nBins));
            bins[bin]++;
        });
        const maxBin = Math.max(...bins);
        const binW = 400 / nBins;
        bins.forEach((count, i) => {
            const h = maxBin > 0 ? (count / maxBin) * 100 : 0;
            ctx.fillStyle = '#388bfd';
            ctx.fillRect(i * binW, 110 - h, binW - 1, h);
        });
        ctx.fillStyle = '#8b949e';
        ctx.font = '9px monospace';
        ctx.fillText(`min=${min.toFixed(3)} max=${max.toFixed(3)} mean=${(vals.reduce((a,b)=>a+b,0)/vals.length).toFixed(3)}`, 4, 118);

        card.appendChild(canvas);
        clsSec.appendChild(card);
        root.appendChild(clsSec);
    }
}

document.addEventListener('DOMContentLoaded', buildReport);
</script>
<div id="report"></div>
</body>
</html>
"##;
