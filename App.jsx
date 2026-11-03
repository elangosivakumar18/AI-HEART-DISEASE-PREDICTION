import { useCallback, useEffect, useRef, useState } from "react";
import { HeartPulse, Send, X, LoaderCircle, Plus, Eye, EyeOff, ArrowRight, Check, Sparkles, Mail, Lock, User, ShieldCheck, Activity, Trash2 } from "lucide-react";

import logo from "./assets/logo.png";

const API_URL = import.meta.env.VITE_API_URL || import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

/* ------------------------------------------------------------------ */
/* Login state (token kept in localStorage)                            */
/* ------------------------------------------------------------------ */

const AUTH_KEY = "cardio_ai_auth";

function loadAuth() {
  try {
    const saved = JSON.parse(localStorage.getItem(AUTH_KEY));
    return saved && saved.token && saved.user ? saved : null;
  } catch {
    return null;
  }
}

function saveAuth(auth) {
  localStorage.setItem(AUTH_KEY, JSON.stringify(auth));
}

function clearAuth() {
  localStorage.removeItem(AUTH_KEY);
}

function authHeaders() {
  const saved = loadAuth();
  return saved ? { Authorization: `Bearer ${saved.token}` } : {};
}


/* ------------------------------------------------------------------ */
/* Prediction page */
/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/* Data                                                                */
/* ------------------------------------------------------------------ */

const DEFAULTS = {
  age: "45", sex: "0", cp: "2", trestbps: "120", chol: "220", fbs: "0",
  restecg: "0", thalach: "150", exang: "0", oldpeak: "1.0", slope: "1", ca: "0", thal: "3",
};

const FALLBACK = {
  age: 45, sex: 0, cp: 2, trestbps: 120, chol: 200, fbs: 0,
  restecg: 0, thalach: 150, exang: 0, oldpeak: 0, slope: 1, ca: 0, thal: 3,
};

/* page field key -> key expected by your FastAPI /predict endpoint */
const API_KEYS = {
  age: "age", sex: "sex", cp: "chest_pain_type", trestbps: "resting_bp",
  chol: "cholesterol", fbs: "fasting_blood_sugar", restecg: "resting_ecg",
  thalach: "max_heart_rate", exang: "exercise_angina", oldpeak: "oldpeak",
  slope: "st_segment", ca: "major_vessels", thal: "thal",
};

const LABELS = {
  age: "Age", sex: "Sex", cp: "Chest pain type", trestbps: "Resting BP", chol: "Cholesterol",
  fbs: "Fasting sugar", restecg: "Resting ECG", thalach: "Max heart rate", exang: "Exercise angina",
  oldpeak: "ST depression", slope: "ST slope", ca: "Vessels (ca)", thal: "Thal scan",
};

const REF = { age: 45, sex: 0, cp: 3, trestbps: 118, chol: 190, fbs: 0, restecg: 0, thalach: 150, exang: 0, oldpeak: 0, slope: 1, ca: 0, thal: 3 };
const MID = { cp: 2, ca: 1, oldpeak: 1, thal: 6, exang: 0 };

const PRESETS = {
  low: { age: 38, sex: 0, cp: 3, trestbps: 114, chol: 182, fbs: 0, restecg: 0, thalach: 168, exang: 0, oldpeak: 0.0, slope: 1, ca: 0, thal: 3 },
  athlete: { age: 44, sex: 1, cp: 2, trestbps: 126, chol: 215, fbs: 0, restecg: 1, thalach: 154, exang: 0, oldpeak: 0.8, slope: 2, ca: 0, thal: 6 },
  high: { age: 64, sex: 1, cp: 1, trestbps: 164, chol: 288, fbs: 1, restecg: 2, thalach: 106, exang: 1, oldpeak: 3.2, slope: 3, ca: 2, thal: 7 },
};

function randomPreset() {
  return {
    age: Math.floor(Math.random() * 45 + 30),
    sex: Math.random() > 0.5 ? 1 : 0,
    cp: Math.floor(Math.random() * 4) + 1,
    trestbps: Math.floor(Math.random() * 70 + 105),
    chol: Math.floor(Math.random() * 160 + 160),
    fbs: Math.random() > 0.7 ? 1 : 0,
    restecg: Math.floor(Math.random() * 3),
    thalach: Math.floor(Math.random() * 90 + 95),
    exang: Math.random() > 0.6 ? 1 : 0,
    oldpeak: parseFloat((Math.random() * 3.5).toFixed(1)),
    slope: Math.floor(Math.random() * 3) + 1,
    ca: Math.floor(Math.random() * 4),
    thal: [3, 6, 7][Math.floor(Math.random() * 3)],
  };
}

const toStrings = (preset) => Object.fromEntries(Object.entries(preset).map(([k, v]) => [k, String(v)]));

/* ------------------------------------------------------------------ */
/* Live preview formula (heuristic - NOT your trained models)          */
/* ------------------------------------------------------------------ */

function toNumbers(patient) {
  const out = {};
  for (const k in FALLBACK) {
    const v = parseFloat(patient[k]);
    out[k] = Number.isNaN(v) ? FALLBACK[k] : v;
  }
  return out;
}

function riskPct(d) {
  let l = -3.8 + (d.age - 45) * 0.045 + (d.sex === 1 ? 0.75 : -0.25);
  l += d.cp === 1 ? 1.35 : d.cp === 2 ? 0.45 : d.cp === 3 ? -0.35 : 0.2;
  if (d.trestbps > 140) l += (d.trestbps - 140) * 0.022 + 0.3; else if (d.trestbps < 115) l -= 0.15;
  if (d.chol > 240) l += (d.chol - 240) * 0.007 + 0.25;
  if (d.fbs === 1) l += 0.42;
  l += d.restecg === 1 ? 0.35 : d.restecg === 2 ? 0.65 : 0;
  l += (150 - d.thalach) * 0.025;
  if (d.exang === 1) l += 0.95;
  l += d.oldpeak * 0.72;
  l += d.slope === 3 ? 0.75 : d.slope === 2 ? 0.25 : -0.4;
  l += d.ca * 0.85;
  l += d.thal === 7 ? 1.35 : d.thal === 6 ? 0.65 : -0.6;
  return Math.min(Math.max(100 / (1 + Math.exp(-l)), 0.5), 99);
}

function liveDrivers(d) {
  const base = riskPct(d);
  const rows = [];
  for (const k in REF) {
    if (d[k] === REF[k]) continue;
    rows.push({ k, v: d[k], delta: base - riskPct({ ...d, [k]: REF[k] }) });
  }
  return rows
    .filter((r) => Math.abs(r.delta) >= 0.05)
    .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta))
    .slice(0, 8);
}

/* percentage-point contribution versus a mid-risk reference value */
function midDelta(d, k) {
  return riskPct(d) - riskPct({ ...d, [k]: MID[k] });
}

/* ------------------------------------------------------------------ */
/* Themes (ambient colour morphing, exactly as in the HTML)            */
/* ------------------------------------------------------------------ */

const THEMES = {
  high: {
    root: "bg-[#fdf1f2] selection:bg-rose-500",
    orb1: "bg-rose-400/35 blur-[140px]", orb2: "bg-red-400/25 blur-[140px]", orb3: "bg-orange-400/20 blur-[140px]",
    glow: "bg-rose-500/15",
    stressText: "High Stress / CAD", stressClass: "text-rose-400",
    bar: "from-amber-500 to-rose-600",
    miniBadge: "bg-rose-100 text-rose-800", miniText: "High Risk",
  },
  moderate: {
    root: "bg-[#fef8ed] selection:bg-amber-500",
    orb1: "bg-amber-300/35 blur-[140px]", orb2: "bg-orange-300/25 blur-[140px]", orb3: "bg-yellow-300/20 blur-[140px]",
    glow: "bg-amber-400/15",
    stressText: "Moderate Risk", stressClass: "text-amber-400",
    bar: "from-teal-400 to-amber-500",
    miniBadge: "bg-amber-100 text-amber-800", miniText: "Moderate",
  },
  low: {
    root: "bg-[#edf4f8] selection:bg-teal-500",
    orb1: "bg-teal-300/30 blur-[130px]", orb2: "bg-cyan-300/25 blur-[140px]", orb3: "bg-blue-300/20 blur-[140px]",
    glow: "bg-teal-400/15",
    stressText: "Low Risk / Stable", stressClass: "text-emerald-400",
    bar: "from-emerald-400 to-teal-400",
    miniBadge: "bg-emerald-100 text-emerald-800", miniText: "Low Risk",
  },
};

const themeFor = (p) => (p >= 65 ? THEMES.high : p >= 30 ? THEMES.moderate : THEMES.low);
const tierKey = (p) => (p >= 65 ? "high" : p >= 30 ? "moderate" : "low");

const TIER_UI = {
  high: {
    label: "High Risk Identified", labelClass: "text-rose-600",
    box: "bg-rose-50 border-rose-300 text-rose-700",
    badge: "bg-rose-100 text-rose-800 border-rose-300", badgeText: "Critical Risk / Urgent Action",
    circle: "text-rose-500", dot: "text-rose-600",
    headline: (p) => `${p.toFixed(1)}% High / Critical Risk of CAD Detected`,
    sub: "Urgent clinical markers indicate severe CAD probability requiring aggressive intervention.",
    recs: [
      ["Cath Lab Referral:", "Urgent coronary angiography within 48-72h."],
      ["Pharmacotherapy:", "Initiate high-intensity statin & dual antiplatelet therapy."],
    ],
    pill: "bg-rose-100 text-rose-800 border-rose-200",
  },
  moderate: {
    label: "Moderate Clinical Risk", labelClass: "text-amber-600",
    box: "bg-amber-50 border-amber-300 text-amber-700",
    badge: "bg-amber-100 text-amber-800 border-amber-300", badgeText: "Moderate CAD Risk",
    circle: "text-amber-500", dot: "text-amber-600",
    headline: (p) => `${p.toFixed(1)}% Moderate Probability of CAD`,
    sub: "Intermediate risk with physiological ischemia markers requiring outpatient diagnostic triage.",
    recs: [
      ["Imaging Workup:", "Exercise stress echocardiography or myocardial nuclear SPECT within 14-30 days."],
      ["Cardiovascular Risk Reduction:", "Optimize BP (<130/80) and LDL-C target (<70 mg/dl)."],
    ],
    pill: "bg-amber-100 text-amber-800 border-amber-200",
  },
  low: {
    label: "Low Probability / Normal", labelClass: "text-emerald-600",
    box: "bg-emerald-50 border-emerald-300 text-emerald-700",
    badge: "bg-emerald-100 text-emerald-800 border-emerald-300", badgeText: "Low Risk (Normal Baseline)",
    circle: "text-emerald-500", dot: "text-emerald-600",
    headline: (p) => `${p.toFixed(1)}% Low Risk (Normal Baseline)`,
    sub: "Algorithmic screening does not detect evidence of active flow-limiting CAD.",
    recs: [
      ["Outpatient Maintenance:", "Re-evaluate during routine annual physical with fasting lipid panel."],
      ["Lifestyle Prescriptions:", "Mediterranean dietary pattern & 150 min/wk moderate aerobic exercise."],
    ],
    pill: "bg-emerald-100 text-emerald-800 border-emerald-200",
  },
};

const TONE = {
  emerald: "bg-emerald-100 text-emerald-800 border-emerald-200",
  amber: "bg-amber-100 text-amber-800 border-amber-200",
  rose: "bg-rose-100 text-rose-800 border-rose-200",
  teal: "bg-teal-100 text-teal-800 border-teal-200",
  purple: "bg-purple-100 text-purple-800 border-purple-200",
  blue: "bg-blue-100 text-blue-800 border-blue-200",
};

const SEG_SELECTED = {
  teal: "bg-teal-700 text-white border-teal-800 shadow-xs transition",
  amber: "bg-amber-600 text-white border-amber-700 shadow-xs transition",
  rose: "bg-rose-600 text-white border-rose-700 shadow-xs transition",
};
const SEG_IDLE = "bg-white text-slate-700 border-slate-300 hover:bg-slate-100 transition";

/* ------------------------------------------------------------------ */
/* Badge text / colour rules (same rules as the HTML sync functions)   */
/* ------------------------------------------------------------------ */

function badgeFor(key, d) {
  const n = d[key];
  switch (key) {
    case "age":
      return [n < 35 ? `Young (${n} yrs)` : n < 55 ? `Adult (${n} yrs)` : `Senior (${n} yrs)`, "teal"];
    case "sex":
      return n === 1 ? ["Male (1)", "blue"] : ["Female (0)", "purple"];
    case "cp":
      return [{ 1: "1 - Typical Angina", 2: "2 - Atypical Angina", 3: "3 - Non-anginal", 4: "4 - Asymptomatic" }[n] ?? String(n), { 1: "rose", 2: "amber", 3: "emerald", 4: "amber" }[n] ?? "amber"];
    case "trestbps":
      return n < 120 ? [`${n} Normal (<120)`, "emerald"] : n <= 139 ? [`${n} Pre-HTN (120-139)`, "amber"] : [`${n} Hypertensive (140+)`, "rose"];
    case "chol":
      return n < 200 ? [`${n} Desirable (<200)`, "emerald"] : n <= 239 ? [`${n} Borderline (200-239)`, "amber"] : [`${n} High Risk (≥240)`, "rose"];
    case "fbs":
      return n === 1 ? ["> 120 mg/dl (Elevated)", "amber"] : ["≤ 120 mg/dl (Normal)", "emerald"];
    case "restecg":
      return [["0 - Normal", "1 - ST-T Abnormality", "2 - LV Hypertrophy"][n] ?? String(n), ["emerald", "amber", "rose"][n] ?? "emerald"];
    case "thalach":
      return n < 110 ? [`${n} bpm (Low Max HR)`, "rose"] : [`${n} bpm (Healthy Target)`, "emerald"];
    case "exang":
      return n === 1 ? ["1 - Induced Angina", "rose"] : ["0 - No Angina", "emerald"];
    case "oldpeak":
      return n === 0 ? ["0.0 mm (Normal Baseline)", "emerald"] : n < 2 ? [`${n} mm (Mild Ischemia)`, "amber"] : [`${n} mm (Significant ST Dep)`, "rose"];
    case "slope":
      return [{ 1: "1 - Upsloping", 2: "2 - Flat", 3: "3 - Downsloping" }[n] ?? String(n), n === 3 ? "rose" : n === 2 ? "amber" : "emerald"];
    case "ca":
      return [`${n} Vessel${n === 1 ? "" : "s"}`, n === 0 ? "emerald" : "rose"];
    case "thal":
      return [{ 3: "3 - Normal", 6: "6 - Fixed Defect", 7: "7 - Reversible Defect" }[n] ?? String(n), n === 7 ? "rose" : n === 6 ? "amber" : "emerald"];
    default:
      return [String(n), "teal"];
  }
}

/* ------------------------------------------------------------------ */
/* Form definition (labels, ranges, options, hints copied from HTML)   */
/* ------------------------------------------------------------------ */

const FIELDS = [
  { key: "age", label: "1. Age (years)", kind: "range", min: 18, max: 100, step: 1, unit: "yrs", ticks: ["18y", "50y", "100y"] },
  {
    key: "sex", label: "2. Biological Sex", kind: "segment",
    options: [[1, "👨 Male (1)", "teal"], [0, "👩 Female (0)", "teal"]], gap: true,
    hint: "Gender baseline adjustment applied to Framingham factor.",
  },
  {
    key: "cp", label: "3. Chest Pain Type (cp)", kind: "select",
    options: [[1, "1 - Typical Angina (High Ischemia)"], [2, "2 - Atypical Angina"], [3, "3 - Non-anginal Pain"], [4, "4 - Asymptomatic"]],
    cpTicks: true,
  },
  { key: "trestbps", label: "4. Resting Blood Pressure", kind: "range", min: 80, max: 220, step: 1, unit: "mmHg", ticks: ["80", "120 Norm", "140 Pre-HTN", "220 Max"] },
  { key: "chol", label: "5. Serum Cholesterol", kind: "range", min: 120, max: 580, step: 1, unit: "mg/dl", ticks: ["<200 Desirable", "200-239", "≥240 High"] },
  {
    key: "fbs", label: "6. Fasting Blood Sugar", kind: "segment",
    options: [[0, "Normal (≤120)", "teal"], [1, "Elevated (>120)", "amber"]],
    hint: "Indicator of metabolic and diabetic comorbidity.",
  },
  {
    key: "restecg", label: "7. Resting ECG Code", kind: "select",
    options: [[0, "0 - Normal Sinus"], [1, "1 - ST-T Wave Abnormality"], [2, "2 - Left Ventricular Hypertrophy"]],
    hint: "Electrocardiographic baseline wave pattern.",
  },
  { key: "thalach", label: "8. Max Heart Rate (thalach)", kind: "range", min: 60, max: 220, step: 1, unit: "bpm", ticks: ["<110 Low Max HR", "150+ Expected", "220"] },
  {
    key: "exang", label: "9. Exercise Angina (exang)", kind: "segment",
    options: [[0, "No (0)", "teal"], [1, "Yes / Induced (1)", "rose"]],
    hint: "Angina provoked during Bruce treadmill stress protocol.",
  },
  { key: "oldpeak", label: "10. Oldpeak (ST Depression)", kind: "range", min: 0, max: 6.2, step: 0.1, unit: "mm", ticks: ["0.0 Normal", "1.5 Mild", "≥2.5 Severe"] },
  {
    key: "slope", label: "11. Peak ST Slope", kind: "select",
    options: [[1, "1 - Upsloping (Low Risk)"], [2, "2 - Flat (Moderate Risk)"], [3, "3 - Downsloping (High Risk)"]],
    hint: "Slope shape of the exercise ST segment.",
  },
  {
    key: "ca", label: "12. Fluoroscopy Vessels (ca)", kind: "select",
    options: [[0, "0 - No colored vessels (Clear)"], [1, "1 - One major vessel"], [2, "2 - Two major vessels"], [3, "3 - Three major vessels (Severe CAD)"]],
    hint: "Number of calcified coronary vessels colored.",
  },
  {
    key: "thal", label: "13. Thalassemia Defect (thal)", kind: "select",
    options: [[3, "3 - Normal Perfusion"], [6, "6 - Fixed Defect (Previous Infarct)"], [7, "7 - Reversible Defect (Active Ischemia)"]],
    hint: "Myocardial nuclear scintigraphy perfusion scan.",
  },
];

const CARD = "p-4 rounded-xl bg-slate-50/80 border border-slate-200/90 space-y-2.5 transition hover:border-teal-300";
const LABEL = "block text-xs font-bold uppercase tracking-wider text-slate-700";
const NUM_INPUT = "block w-full rounded-lg border-slate-300 pl-3 pr-10 text-slate-900 font-bold text-sm focus:border-teal-500 focus:ring-teal-500 py-2";
const SELECT = "block w-full rounded-lg border-slate-300 py-2 pl-3 pr-8 text-slate-800 text-sm font-semibold focus:border-teal-500 focus:ring-teal-500";

function FieldCard({ field, patient, numbers, onChange }) {
  const [badgeText, tone] = badgeFor(field.key, numbers);
  const value = patient[field.key];
  const id = `input-${field.key}`;

  return (
    <div className={CARD}>
      <div className="flex items-center justify-between">
        <label className={LABEL} htmlFor={field.kind === "segment" ? undefined : id}>{field.label}</label>
        <span className={`text-[11px] font-bold px-2 py-0.5 rounded-md border ${TONE[tone]}`}>{badgeText}</span>
      </div>

      {field.kind === "range" && (
        <>
          <div className="relative rounded-lg shadow-2xs">
            <input
              className={NUM_INPUT}
              id={id}
              name={field.key}
              type="number"
              min={field.min}
              max={field.max}
              step={field.step}
              value={value}
              required
              onChange={(e) => onChange(field.key, e.target.value)}
            />
            <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-3">
              <span className="text-xs text-slate-400 font-mono">{field.unit}</span>
            </div>
          </div>
          <div className="space-y-1">
            <input
              className="w-full h-1.5"
              type="range"
              aria-label={`${field.label} slider`}
              min={field.min}
              max={field.max}
              step={field.step}
              value={value === "" ? field.min : value}
              onChange={(e) => onChange(field.key, e.target.value)}
            />
            <div className="flex justify-between text-[10px] text-slate-400 font-mono">
              {field.ticks.map((t) => <span key={t}>{t}</span>)}
            </div>
          </div>
        </>
      )}

      {field.kind === "segment" && (
        <>
          <div className="grid grid-cols-2 gap-2">
            {field.options.map(([v, text, sel]) => (
              <button
                key={v}
                type="button"
                aria-pressed={String(v) === String(value)}
                onClick={() => onChange(field.key, String(v))}
                className={`py-2 px-3 rounded-lg text-xs font-bold border flex items-center justify-center ${field.gap ? "gap-1 " : ""}${String(v) === String(value) ? SEG_SELECTED[sel] : SEG_IDLE}`}
              >
                {text}
              </button>
            ))}
          </div>
          <p className="text-[11px] text-slate-500">{field.hint}</p>
        </>
      )}

      {field.kind === "select" && (
        <>
          <select className={SELECT} id={id} name={field.key} value={value} onChange={(e) => onChange(field.key, e.target.value)}>
            {field.options.map(([v, text]) => <option key={v} value={v}>{text}</option>)}
          </select>
          {field.cpTicks ? (
            <div className="flex gap-1 text-[10px] text-slate-400 font-mono">
              <span>0: Typical</span><span>•</span><span>1: Atypical</span><span>•</span><span>2: Non-ang</span><span>•</span><span>3: Asymp</span>
            </div>
          ) : (
            <p className="text-[11px] text-slate-500">{field.hint}</p>
          )}
        </>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

function toPct(value) {
  if (value === null || value === undefined || value === "" || typeof value === "boolean") return null;
  const n = Number(value);
  if (!Number.isFinite(n) || n < 0 || n > 100) return null;
  return Math.round(Math.min(100, Math.max(0, n <= 1 ? n * 100 : n)));
}

function download(name, text, type = "text/plain") {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 500);
}

/* ------------------------------------------------------------------ */
/* Risk Drivers Visualizer Component (Ablation Attribution)           */
/* ------------------------------------------------------------------ */

function RiskDriversVisualizer({
  drivers,
  title = "Live Risk Drivers",
  subtitle = "Percentage-point change in the preview estimate versus a healthy reference value for each feature. Updates as you edit.",
  badge = "ablation attribution",
  headerPills = null,
  compact = false,
}) {
  const maxDelta = Math.max(1, ...(drivers || []).map((r) => Math.abs(r.delta)));

  return (
    <div className={`bg-white/95 backdrop-blur-md rounded-2xl shadow-card-elevated border border-slate-200 ${compact ? "p-4" : "p-6"}`}>
      <div className="flex items-start justify-between gap-3 mb-4">
        <div>
          <h3 className={`${compact ? "text-base" : "text-lg"} font-bold text-slate-900`}>{title}</h3>
          <p className="text-xs text-slate-500 mt-0.5">{subtitle}</p>
        </div>
        <div className="flex items-center gap-2 shrink-0 flex-wrap justify-end">
          {headerPills}
          {badge && (
            <span className="text-[10px] font-mono text-teal-700 bg-teal-50 border border-teal-200 px-2.5 py-1 rounded-md shrink-0">
              {badge}
            </span>
          )}
        </div>
      </div>

      <div className="space-y-2.5">
        {!drivers || drivers.length === 0 ? (
          <p className="text-xs text-slate-400 italic py-6 text-center">Every feature matches the healthy reference.</p>
        ) : (
          drivers.map((r) => {
            const up = r.delta > 0;
            const w = Math.min(Math.max((Math.abs(r.delta) / maxDelta) * 50, 2), 50);
            return (
              <div key={r.k} className="flex items-center gap-3 text-xs">
                <div className={`${compact ? "w-32" : "w-36"} shrink-0 text-right truncate`}>
                  <span className="font-bold text-slate-800">{LABELS[r.k] || r.k}</span>{" "}
                  <span className="font-mono text-slate-400 font-medium ml-1">{r.v}</span>
                </div>
                <div className="relative flex-grow h-4 bg-slate-100 rounded-full overflow-hidden flex items-center">
                  <div className="absolute left-1/2 top-0 bottom-0 w-[2px] -translate-x-1/2 bg-slate-300 z-10" />
                  <div
                    className={`absolute h-3.5 inset-y-0.5 rounded-full transition-all duration-300 ${
                      up ? "bg-[#ff5a77]" : "bg-emerald-400"
                    }`}
                    style={{ [up ? "left" : "right"]: "50%", width: `${w}%` }}
                  />
                </div>
                <div
                  className={`w-14 shrink-0 font-mono font-bold text-sm text-right ${
                    up ? "text-[#ff3b5c]" : "text-emerald-600"
                  }`}
                >
                  {up ? "+" : ""}
                  {r.delta.toFixed(1)}
                </div>
              </div>
            );
          })
        )}
      </div>

      <div className="flex justify-between text-[10px] text-slate-400 font-mono mt-3.5 px-1">
        <span>&larr; lowers risk</span>
        <span>raises risk &rarr;</span>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Prediction Record Formatter                                         */
/* ------------------------------------------------------------------ */

function formatHistoryRecord(record) {
  if (!record) return null;
  const p = record.patient_data || {};
  const pred = record.prediction || {};

  const age = String(p.age ?? DEFAULTS.age);
  const sex = String(p.sex ?? DEFAULTS.sex);
  const cp = String(p.cp ?? p.chest_pain_type ?? p["chest pain type"] ?? DEFAULTS.cp);
  const trestbps = String(p.trestbps ?? p.resting_bp ?? p["resting blood pressure"] ?? DEFAULTS.trestbps);
  const chol = String(p.chol ?? p.cholesterol ?? p["serum cholestoral"] ?? DEFAULTS.chol);
  const fbs = String(p.fbs ?? p.fasting_blood_sugar ?? p["fasting blood sugar"] ?? DEFAULTS.fbs);
  const restecg = String(p.restecg ?? p.resting_ecg ?? p["resting electrocardiographic results"] ?? DEFAULTS.restecg);
  const thalach = String(p.thalach ?? p.max_heart_rate ?? p["max heart rate"] ?? DEFAULTS.thalach);
  const exang = String(p.exang ?? p.exercise_angina ?? p["exercise induced angina"] ?? DEFAULTS.exang);
  const oldpeak = String(p.oldpeak ?? DEFAULTS.oldpeak);
  const slope = String(p.slope ?? p.st_segment ?? p["ST segment"] ?? DEFAULTS.slope);
  const ca = String(p.ca ?? p.major_vessels ?? p["major vessels"] ?? DEFAULTS.ca);
  const thal = String(p.thal ?? DEFAULTS.thal);

  const inputs = {
    age, sex, cp, trestbps, chol, fbs,
    restecg, thalach, exang, oldpeak, slope, ca, thal,
  };

  const numbers = toNumbers(inputs);

  const dt = toPct(pred?.decision_tree?.probability);
  const bn = toPct(pred?.bayesian_network?.probability ?? pred?.bayesian_network?.disease_probability);
  let prob = pred?.probability !== undefined ? toPct(pred.probability) : null;
  if (prob === null) {
    if (dt !== null && bn !== null) {
      prob = Math.round((dt + bn) / 2);
    } else {
      prob = dt ?? bn ?? Math.round(riskPct(numbers));
    }
  }

  let dateObj = new Date();
  if (record.created_at) {
    const raw = String(record.created_at);
    const iso = raw.includes("Z") || raw.includes("+") ? raw : raw + "Z";
    const parsed = new Date(iso);
    if (!isNaN(parsed.getTime())) {
      dateObj = parsed;
    }
  }

  const time = dateObj.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  const stamp = dateObj.toLocaleString();

  return {
    id: record.id || null,
    prob,
    dt,
    bn,
    raw: pred,
    numbers,
    inputs,
    time,
    stamp,
  };
}

/* ------------------------------------------------------------------ */
/* Page                                                                */
/* ------------------------------------------------------------------ */

function PredictionPage({ onContext, user, onLogout, onOpenAuth }) {
  const [patient, setPatient] = useState(DEFAULTS);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const storageKey = user?.id ? `cardio_ai_history_${user.id}` : "cardio_ai_history_guest";
  const [history, setHistory] = useState(() => {
    try {
      const cached = JSON.parse(localStorage.getItem(storageKey));
      if (Array.isArray(cached)) return cached;
    } catch {}
    return [];
  });
  const [historyLoading, setHistoryLoading] = useState(false);

  const [toast, setToast] = useState("");
  const [modalOpen, setModalOpen] = useState(false);
  const [modalShown, setModalShown] = useState(false);
  const [shownProb, setShownProb] = useState(0);
  const [historyViewIndex, setHistoryViewIndex] = useState(null);
  const [expandedHistoryIndex, setExpandedHistoryIndex] = useState(null);

  const glowRef = useRef(null);
  const toastTimer = useRef(null);

  const numbers = toNumbers(patient);
  const live = Math.round(riskPct(numbers) * 10) / 10;
  const theme = themeFor(live);
  const drivers = liveDrivers(numbers);

  /* Load user prediction history from backend upon login/mount */
  useEffect(() => {
    let active = true;
    async function loadHistory() {
      const auth = loadAuth();
      if (!auth?.token) return;

      setHistoryLoading(true);
      try {
        const response = await fetch(`${API_URL}/history?limit=100`, {
          headers: authHeaders(),
        });

        if (response.status === 401) {
          onLogout?.();
          return;
        }

        if (!response.ok) throw new Error("Could not load prediction history");

        const data = await response.json();
        if (!active || !data || !Array.isArray(data.history)) return;

        const records = data.history.map(formatHistoryRecord).filter(Boolean);
        // DB returns latest first (DESC). Reverse to ASC order so oldest is #1 and latest is appended.
        const chronHistory = records.reverse();
        setHistory(chronHistory);
        try {
          localStorage.setItem(storageKey, JSON.stringify(chronHistory));
        } catch {}
      } catch (err) {
        console.warn("Could not retrieve prediction history from server:", err);
      } finally {
        if (active) setHistoryLoading(false);
      }
    }

    loadHistory();
    return () => {
      active = false;
    };
  }, [user?.id, storageKey, onLogout]);

  /* Persist history to localStorage whenever history changes */
  useEffect(() => {
    if (history.length > 0) {
      try {
        localStorage.setItem(storageKey, JSON.stringify(history));
      } catch {}
    }
  }, [history, storageKey]);

  /* mouse-follow ambient spotlight */
  useEffect(() => {
    const move = (e) => {
      if (glowRef.current) glowRef.current.style.transform = `translate(${e.clientX}px, ${e.clientY}px)`;
    };
    document.addEventListener("mousemove", move);
    return () => document.removeEventListener("mousemove", move);
  }, []);

  /* Escape closes the results modal */
  useEffect(() => {
    const onKey = (e) => e.key === "Escape" && closeModal();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  /* count-up animation in the modal */
  useEffect(() => {
    if (!modalOpen || !result) return undefined;
    let current = 0;
    const target = result.prob;
    const step = target / 20;
    setShownProb(0);
    const timer = setInterval(() => {
      current += step;
      if (current >= target) {
        current = target;
        clearInterval(timer);
      }
      setShownProb(current);
    }, 20);
    return () => clearInterval(timer);
  }, [modalOpen, result]);

  /* give the chatbot a plain-text summary of the current result */
  useEffect(() => {
    if (!onContext) return;
    onContext(
      result
        ? `Please explain this prediction in simple terms. Inputs: ${Object.keys(API_KEYS).map((k) => `${API_KEYS[k]}=${result.inputs[k]}`).join(", ")}. ` +
            `Decision Tree: ${result.dt ?? "n/a"}%, Bayesian Network: ${result.bn ?? "n/a"}%.`
        : "Please explain in simple terms what the 13 patient input fields mean."
    );
  }, [result, onContext]);

  function notify(message) {
    setToast(message);
    clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(""), 2400);
  }

  function setField(key, value) {
    setPatient((p) => ({ ...p, [key]: value }));
  }

  function applyPreset(type) {
    setPatient(toStrings(type === "random" ? randomPreset() : PRESETS[type]));
  }

  function openModal() {
    setModalOpen(true);
    setTimeout(() => setModalShown(true), 10);
  }

  function closeModal() {
    setModalShown(false);
    setTimeout(() => setModalOpen(false), 300);
  }

  async function runPrediction() {
    setError("");
    setLoading(true);

    const body = Object.fromEntries(Object.keys(API_KEYS).map((k) => [API_KEYS[k], Number(patient[k])]));

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify(body),
      });
      const data = await response.json().catch(() => ({}));

      if (response.status === 401 && loadAuth()) {
        onLogout?.();
        throw new Error("Your session expired. Please log in again.");
      }

      if (!response.ok) {
        const detail = Array.isArray(data.detail)
          ? data.detail.map((i) => `${Array.isArray(i.loc) ? i.loc.join(".") : ""}: ${i.msg}`).join("; ")
          : data.detail;
        throw new Error(detail || `Prediction failed (${response.status}).`);
      }

      const prediction = data.prediction ?? data.result ?? data;
      const dt = toPct(prediction?.decision_tree?.probability);
      const bn = toPct(prediction?.bayesian_network?.probability);
      const prob = dt !== null && bn !== null ? Math.round((dt + bn) / 2) : dt ?? bn;

      if (prob === null) {
        throw new Error("The API response has no numeric decision_tree.probability or bayesian_network.probability.");
      }

      const run = {
        id: data.history_id || null,
        prob, dt, bn, raw: prediction,
        numbers: toNumbers(patient),
        inputs: { ...patient },
        time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
        stamp: new Date().toLocaleString(),
      };

      setResult(run);
      setHistory((h) => [...h, run]);
      /* Fallback save if not saved automatically via /predict */
      if (!data.history_id && loadAuth()) {
        fetch(`${API_URL}/history`, {
          method: "POST",
          headers: { "Content-Type": "application/json", ...authHeaders() },
          body: JSON.stringify({ patient_data: body, prediction_data: prediction }),
        }).then((historyResponse) => {
          if (historyResponse.status === 401) onLogout?.();
        }).catch((historyError) => {
          console.warn("Cardio AI history was not saved:", historyError);
        });
      }
      openModal();
    } catch (err) {
      console.error("Cardio AI prediction error:", err);
      setError(
        err instanceof TypeError
          ? `Cannot reach the prediction API at ${API_URL}. Start the backend with ".venv/bin/uvicorn api:app --reload --host 127.0.0.1 --port 8000" and try again.`
          : err.message || "Could not complete the prediction request."
      );
    } finally {
      setLoading(false);
    }
  }

  function exportReport() {
    if (!result) return notify("Run a prediction first");
    const dr = liveDrivers(result.numbers).slice(0, 5);
    const text =
      `CARDIO AI - RISK REPORT\nGenerated: ${new Date().toLocaleString()}\n\n` +
      `Combined probability: ${result.prob}% (${TIER_UI[tierKey(result.prob)].label})\n` +
      `Decision Tree: ${result.dt ?? "n/a"}%\nBayesian Network: ${result.bn ?? "n/a"}%\n\nInputs\n` +
      Object.keys(API_KEYS).map((k) => `  ${LABELS[k]}: ${result.inputs[k]}`).join("\n") +
      `\n\nTop preview drivers (percentage points vs healthy reference)\n` +
      dr.map((r) => `  ${LABELS[r.k]}: ${r.delta > 0 ? "+" : ""}${r.delta.toFixed(1)}`).join("\n") +
      `\n\nEducational model output only. Not a medical diagnosis.\n`;
    download("cardio-ai-report.txt", text);
    notify("Report downloaded");
  }

  function exportHistoryCsv() {
    if (!history.length) return notify("Run a prediction first");
    const keys = Object.keys(API_KEYS);
    const rows = history.map((h, i) => [i + 1, h.time, h.prob, h.dt ?? "", h.bn ?? "", ...keys.map((k) => h.inputs[k])].join(","));
    download("cardio-ai-history.csv", [["run", "time", "combined", "decision_tree", "bayesian_network", ...keys.map((k) => API_KEYS[k])].join(","), ...rows].join("\n"), "text/csv");
  }

  async function clearHistory() {
    setHistory([]);
    setHistoryViewIndex(null);
    setExpandedHistoryIndex(null);
    try {
      localStorage.removeItem(storageKey);
    } catch {}

    const auth = loadAuth();
    if (auth?.token) {
      try {
        await fetch(`${API_URL}/history`, {
          method: "DELETE",
          headers: authHeaders(),
        });
      } catch (err) {
        console.warn("Could not clear server history:", err);
      }
    }
    notify("Prediction history cleared");
  }

  /* ---- derived result UI ---- */
  const tier = result ? tierKey(result.prob) : "low";
  const ui = TIER_UI[tier];
  const rn = result?.numbers;

  let factor1 = "", factor1Sub = "", factor2 = "", factor2Sub = "", recPrimary = "";
  if (result) {
    if (tier === "high") {
      factor1 = rn.ca > 0 ? `${rn.ca} Fluoroscopy Vessel(s) Occluded` : "Severe ST Depression (Oldpeak ≥ 2.0)";
      factor1Sub = "Largest upward driver in the live preview";
      factor2 = rn.exang === 1 ? "Exercise Induced Angina Present" : `Severe SBP (${rn.trestbps} mm Hg)`;
      factor2Sub = "Significant sign of myocardial oxygen deficit";
      recPrimary = "Urgent Cath Lab Referral / Coronary Angiogram";
    } else if (tier === "moderate") {
      factor1 = rn.thal === 7 ? "Reversible Defect (Thal = 7)" : `Borderline Cholesterol (${rn.chol} mg/dl)`;
      factor1Sub = "Moderate upward contribution in the live preview";
      factor2 = `ST Depression Oldpeak: ${rn.oldpeak} mm`;
      factor2Sub = "Exercise stress electrocardiographic abnormality";
      recPrimary = "Exercise Stress Echocardiography / SPECT";
    } else {
      factor1 = "Hemodynamic markers within normal limits";
      factor1Sub = "No fluoroscopic or nuclear perfusion defects";
      factor2 = "Absence of exercise-induced ischemia";
      factor2Sub = "Normotensive profile & target heart rate achieved";
      recPrimary = "Routine Outpatient Wellness / Annual Screen";
    }
  }

  const shapList = [];
  if (result) {
    const fmt = (v) => `${v >= 0 ? "+" : "-"}${Math.abs(v).toFixed(1)}%`;
    const add = (name, k) => { const v = midDelta(rn, k); shapList.push({ name, impact: fmt(v), isRisk: v >= 0 }); };
    add(rn.thal === 7 ? "Reversible Defect (Thal = 7)" : rn.thal === 6 ? "Fixed Defect (Thal = 6)" : "Normal Perfusion (Thal = 3)", "thal");
    add(rn.ca > 0 ? `${rn.ca} Calcified Vessel(s) (ca = ${rn.ca})` : "Zero Vessels Occluded (ca = 0)", "ca");
    add(rn.oldpeak >= 2 ? `Severe ST Depression (${rn.oldpeak} mm)` : rn.oldpeak > 0 ? `Mild ST Depression (${rn.oldpeak} mm)` : "Normal ST Baseline (Oldpeak 0.0)", "oldpeak");
    if (rn.cp === 1) add("Typical Angina (cp = 1)", "cp");
    else if (rn.cp === 3) add("Non-anginal Pain (cp = 3)", "cp");
    if (rn.exang === 1) add("Exercise Angina (exang = 1)", "exang");
  }

  const sparkPts = history.map((h, i) => {
    const x = history.length === 1 ? 100 : (i / (history.length - 1)) * 190 + 5;
    const y = 45 - (h.prob / 100) * 40;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });

  return (
    <div className={`min-h-screen flex flex-col font-sans text-slate-800 antialiased ${theme.root} selection:text-white relative overflow-x-hidden transition-colors duration-700`}>
      {/* Ambient dynamic aurora background */}
      <div aria-hidden="true" className="fixed inset-0 pointer-events-none z-0 overflow-hidden">
        <div ref={glowRef} className={`absolute w-[450px] h-[450px] rounded-full ${theme.glow} blur-[110px] transition-transform duration-75 ease-out -translate-x-1/2 -translate-y-1/2`} />
        <div className={`orb-1 aurora-blur absolute -top-40 -left-40 w-[600px] h-[600px] rounded-full ${theme.orb1}`} />
        <div className={`orb-2 aurora-blur absolute top-1/3 -right-40 w-[550px] h-[550px] rounded-full ${theme.orb2}`} />
        <div className={`orb-3 aurora-blur absolute -bottom-40 left-1/4 w-[650px] h-[650px] rounded-full ${theme.orb3}`} />
      </div>

      {/* Header */}
      <header className="bg-[#0f293d]/95 backdrop-blur-md text-white sticky top-0 z-40 shadow-lg border-b border-cyan-900/40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3 flex items-center justify-between gap-4">
          <div className="flex items-center space-x-3.5">
            <div className="w-14 h-14 rounded-xl bg-white border border-white/20 shadow-md p-0 flex items-center justify-center shrink-0 backdrop-blur-sm overflow-hidden">
              <img alt="Cardio AI Logo" className="w-full h-full object-contain scale-110" src={logo} />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-lg sm:text-xl font-black tracking-tight text-white leading-none">Cardio AI</h1>
                <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-teal-500/25 text-teal-300 border border-teal-400/40 leading-normal inline-flex items-center">Clinical ML</span>
              </div>
              <p className="text-xs text-slate-300 hidden sm:block mt-0.5">Heart disease prediction &amp; risk stratification dashboard</p>
            </div>
          </div>

          <div className="flex items-center gap-3 sm:gap-6">
            <div className="hidden md:flex items-center gap-3 bg-white/10 px-3.5 py-1.5 rounded-xl border border-white/10 backdrop-blur-sm">
              <div className="text-right">
                <div className="text-[9px] uppercase tracking-wider font-bold text-slate-300">Live Stress Meter</div>
                <div className={`font-bold ${theme.stressClass} text-xs`}>{theme.stressText}</div>
              </div>
              <div className="w-24 bg-slate-700/80 rounded-full h-2 overflow-hidden border border-slate-600/60 p-[1px]">
                <div className={`h-full bg-gradient-to-r ${theme.bar} rounded-full transition-all duration-300`} style={{ width: `${Math.min(live, 100)}%` }} />
              </div>
              <span className="font-mono text-xs font-bold text-white">{live.toFixed(0)}%</span>
            </div>

            {user && (
              <div className="flex items-center gap-2.5 pl-3 sm:pl-4 border-l border-white/15">
                {user.is_anonymous ? (
                  <>
                    <span className="hidden sm:inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-bold bg-amber-400/20 text-amber-200 border border-amber-300/30">
                      <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
                      Guest Trial
                    </span>
                    <button
                      type="button"
                      onClick={() => onOpenAuth?.("login")}
                      className="px-3 py-1.5 rounded-lg text-xs font-bold text-white bg-white/15 hover:bg-white/25 border border-white/20 transition cursor-pointer"
                    >
                      Log in
                    </button>
                    <button
                      type="button"
                      onClick={() => onOpenAuth?.("register")}
                      className="px-3 py-1.5 rounded-lg text-xs font-bold text-slate-900 bg-teal-300 hover:bg-teal-200 shadow-sm transition cursor-pointer"
                    >
                      Create account
                    </button>
                    <button
                      type="button"
                      onClick={onLogout}
                      className="px-2.5 py-1.5 rounded-lg text-xs font-medium text-slate-300 hover:text-white transition cursor-pointer"
                      title="Exit session"
                    >
                      Exit
                    </button>
                  </>
                ) : (
                  <>
                    <div className="w-9 h-9 rounded-full bg-teal-500/25 border border-teal-400/40 text-teal-200 font-bold flex items-center justify-center text-sm uppercase shrink-0">
                      {(user.full_name || user.name || user.email || "?").trim().charAt(0)}
                    </div>
                    <div className="hidden sm:block text-left leading-tight">
                      <div className="text-xs font-bold text-white max-w-[150px] truncate">{user.full_name || user.name || user.email}</div>
                      <div className="text-[10px] text-slate-300 max-w-[150px] truncate">{user.email}</div>
                    </div>
                    <button
                      type="button"
                      onClick={onLogout}
                      className="px-3 py-1.5 rounded-lg text-xs font-semibold text-slate-100 bg-white/10 hover:bg-white/20 border border-white/10 transition cursor-pointer"
                    >
                      Log out
                    </button>
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="relative z-10 flex-grow py-7 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto w-full">
        {/* Title */}
        <div className="flex flex-col md:flex-row md:items-center justify-between pb-5 gap-4">
          <div>
            <div className="flex items-center gap-2.5">
              <h2 className="text-2xl sm:text-3xl font-black text-slate-900 tracking-tight">Patient assessment</h2>
              <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-teal-100 text-teal-800 border border-teal-200">
                <span className="w-2 h-2 rounded-full bg-teal-500 animate-pulse" />
                Dynamic Reactive Engine
              </span>
            </div>
            <p className="text-xs sm:text-sm text-slate-600 mt-0.5">Dual-control interactive inputs &amp; real-time algorithmic biomarker risk inference</p>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-500">Dataset Mapping:</span>
            <span className="inline-flex items-center px-3 py-1 rounded-lg text-xs font-semibold bg-white/90 text-teal-800 border border-teal-200 shadow-sm">UCI Cleveland Standard (13 Features)</span>
          </div>
        </div>

        {/* Quick presets */}
        <div className="mb-5 bg-white/90 backdrop-blur-md rounded-2xl p-3 sm:p-4 border border-slate-200/90 shadow-sm flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <svg className="w-4 h-4 text-teal-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path d="M13 10V3L4 14h7v7l9-11h-7z" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" />
            </svg>
            <span className="text-xs font-bold text-slate-700">Quick Presets:</span>
          </div>
          <div className="flex items-center flex-wrap gap-2 w-full sm:w-auto">
            <button type="button" onClick={() => applyPreset("low")} className="px-3 py-1.5 rounded-xl text-xs font-bold bg-emerald-50 hover:bg-emerald-100 text-emerald-800 border border-emerald-300 transition transform hover:-translate-y-0.5 flex items-center gap-1.5 cursor-pointer">
              <span className="w-2 h-2 rounded-full bg-emerald-500" /> Normal Baseline (Low Risk)
            </button>
            <button type="button" onClick={() => applyPreset("athlete")} className="px-3 py-1.5 rounded-xl text-xs font-bold bg-cyan-50 hover:bg-cyan-100 text-cyan-800 border border-cyan-300 transition transform hover:-translate-y-0.5 flex items-center gap-1.5 cursor-pointer">
              <span className="w-2 h-2 rounded-full bg-cyan-500" /> Borderline Athlete
            </button>
            <button type="button" onClick={() => applyPreset("high")} className="px-3 py-1.5 rounded-xl text-xs font-bold bg-rose-50 hover:bg-rose-100 text-rose-800 border border-rose-300 transition transform hover:-translate-y-0.5 flex items-center gap-1.5 cursor-pointer">
              <span className="w-2 h-2 rounded-full bg-rose-500 animate-pulse" /> Acute CAD / Ischemic Case
            </button>
            <button type="button" onClick={() => applyPreset("random")} className="px-3 py-1.5 rounded-xl text-xs font-bold bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-300 transition transform hover:-translate-y-0.5 flex items-center gap-1.5 cursor-pointer">
              🎲 Randomize
            </button>
          </div>
        </div>

        {/* Form card */}
        <div className="bg-white/95 backdrop-blur-md rounded-2xl shadow-card-elevated border border-slate-200 overflow-hidden transition-all duration-300">
          <div className="px-6 py-4 bg-gradient-to-r from-teal-50/90 via-cyan-50/50 to-transparent border-b border-slate-100 flex items-center justify-between flex-wrap gap-2">
            <div className="flex items-center space-x-2.5">
              <div className="p-2 rounded-xl bg-teal-100 text-teal-800 shadow-inner">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path d="M13 10V3L4 14h7v7l9-11h-7z" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.2" />
                </svg>
              </div>
              <div>
                <h3 className="text-base font-bold text-slate-900">Patient Clinical Biomarkers (13 Features)</h3>
                <p className="text-xs text-slate-500">Dual range sliders + typed inputs synchronize directly with live risk weights</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold text-slate-500">Live background reactive sync</span>
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-ping" />
            </div>
          </div>

          <form className="p-6 md:p-8 space-y-7" onSubmit={(e) => { e.preventDefault(); runPrediction(); }}>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {FIELDS.map((field) => (
                <FieldCard key={field.key} field={field} patient={patient} numbers={numbers} onChange={setField} />
              ))}
            </div>

            <div className="pt-3 border-t border-slate-100 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-slate-500">
              <p className="italic text-center sm:text-left">
                Coded fields must match the preprocessing categories of your trained prediction models.
              </p>
              <span className="inline-flex items-center gap-1.5 text-[11px] font-mono text-teal-700 bg-teal-50 px-2.5 py-1 rounded-md border border-teal-200 shrink-0">
                <span className="w-1.5 h-1.5 rounded-full bg-teal-600" />
                Live Preview Drivers Active
              </span>
            </div>

            <div className="flex flex-wrap items-center justify-between gap-4 pt-1">
              <div className="flex items-center gap-3">
                <button
                  type="submit"
                  disabled={loading}
                  className="inline-flex items-center justify-center px-7 py-3.5 rounded-xl bg-gradient-to-r from-[#0e5c6a] to-[#0a4652] hover:from-[#0b4d59] hover:to-[#083842] active:scale-[0.98] text-white font-bold text-sm shadow-lg hover:shadow-glow-teal transition gap-2.5 cursor-pointer disabled:opacity-70 disabled:cursor-wait"
                >
                  <svg className={`w-5 h-5 text-teal-300 ${loading ? "animate-pulse" : ""}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path d="M3 12h3.5l2-6 3.5 12 3-8 2 5 2-3H21" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.2" />
                  </svg>
                  <span>{loading ? "Running models..." : "Run prediction"}</span>
                </button>
                <button
                  type="button"
                  disabled={loading}
                  onClick={() => applyPreset("low")}
                  className="inline-flex items-center justify-center px-5 py-3.5 rounded-xl bg-white hover:bg-slate-50 text-slate-700 font-semibold text-sm border border-slate-300 shadow-sm transition gap-2 cursor-pointer disabled:opacity-60"
                >
                  <svg className="w-4 h-4 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" />
                  </svg>
                  <span>Reset form</span>
                </button>
              </div>

              <div className="flex items-center gap-3 px-4 py-2 bg-slate-50 rounded-xl border border-slate-200">
                <span className="text-xs font-semibold text-slate-500">Live Estimate:</span>
                <div className="flex items-center gap-2">
                  <span className="font-mono font-bold text-sm text-slate-800">{live.toFixed(1)}% CAD Risk</span>
                  <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${theme.miniBadge}`}>{theme.miniText}</span>
                </div>
              </div>
            </div>
            <p className="text-[11px] text-slate-400 -mt-3">The live estimate is a quick heuristic preview, not the output of your trained models. Press Run prediction for the model results.</p>
          </form>
        </div>

        {error && (
          <div role="alert" className="mt-6 p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-sm">
            <strong className="block">Unable to complete prediction</strong>
            <span className="text-xs break-words">{error}</span>
          </div>
        )}

        {/* Results panel */}
        {result && (
          <div className="mt-8 bg-white/95 backdrop-blur-md rounded-2xl shadow-card-elevated border border-slate-200 p-6 md:p-8 transition-all">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 pb-6 border-b border-slate-100">
              <div>
                <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-800 mb-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-500" />
                  Dynamic Inference Result
                </div>
                <h3 className="text-2xl font-bold text-slate-900">Diagnostic Assessment Summary</h3>
                <p className="text-xs sm:text-sm text-slate-500">Computed by the Decision Tree and Bayesian Network models (average of both outputs)</p>
              </div>
              <div className="flex items-center gap-4">
                <div className="text-right">
                  <div className="text-xs uppercase font-bold tracking-wider text-slate-400">Heart Disease Risk</div>
                  <div className={`text-2xl font-extrabold ${ui.labelClass}`}>{ui.label}</div>
                </div>
                <div className={`w-16 h-16 rounded-2xl border-2 flex flex-col items-center justify-center font-bold shadow-inner ${ui.box}`}>
                  <span className="text-lg">{result.prob.toFixed(0)}%</span>
                  <span className="text-[10px] uppercase tracking-tighter">Prob.</span>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-6">
              <div className="bg-slate-50/80 rounded-xl p-4 border border-slate-200">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-500">Primary Contributing Factor</span>
                <p className="text-sm sm:text-base font-semibold text-slate-800 mt-1">{factor1}</p>
                <p className="text-xs text-slate-400 mt-0.5">{factor1Sub}</p>
              </div>
              <div className="bg-slate-50/80 rounded-xl p-4 border border-slate-200">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-500">Secondary Contributing Factor</span>
                <p className="text-sm sm:text-base font-semibold text-slate-800 mt-1">{factor2}</p>
                <p className="text-xs text-slate-400 mt-0.5">{factor2Sub}</p>
              </div>
              <div className="bg-slate-50/80 rounded-xl p-4 border border-slate-200">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-500">Recommended Next Step</span>
                <p className="text-sm sm:text-base font-semibold text-slate-800 mt-1">{recPrimary}</p>
                <p className="text-xs text-slate-400 mt-0.5">Recommended within clinical timeframe</p>
              </div>
            </div>

            <div className="flex flex-wrap items-center justify-between gap-3 pt-5">
              <p className="text-[11px] text-slate-400">Educational model output, not a medical diagnosis.</p>
              <button type="button" onClick={openModal} className="px-4 py-2 text-xs font-bold text-teal-800 bg-teal-50 hover:bg-teal-100 border border-teal-200 rounded-lg transition cursor-pointer">
                Open detailed report
              </button>
            </div>
          </div>
        )}

        {/* Insights */}
        <section className="mt-8 grid grid-cols-1 lg:grid-cols-5 gap-6">
          <div className="lg:col-span-3">
            <RiskDriversVisualizer
              drivers={
                historyViewIndex !== null && history[historyViewIndex]
                  ? liveDrivers(history[historyViewIndex].numbers)
                  : drivers
              }
              title={
                historyViewIndex !== null && history[historyViewIndex]
                  ? `Run #${historyViewIndex + 1} Risk Drivers (${history[historyViewIndex].prob.toFixed(1)}%)`
                  : "Live Risk Drivers"
              }
              subtitle={
                historyViewIndex !== null && history[historyViewIndex]
                  ? `Percentage-point change in estimate versus a healthy reference for Run #${historyViewIndex + 1} (${history[historyViewIndex].time}).`
                  : "Percentage-point change in the preview estimate versus a healthy reference value for each feature. Updates as you edit."
              }
              headerPills={
                history.length > 0 ? (
                  <div className="flex items-center gap-1.5 flex-wrap">
                    <button
                      type="button"
                      onClick={() => setHistoryViewIndex(null)}
                      className={`px-2.5 py-1 text-[11px] font-bold rounded-lg border transition cursor-pointer ${
                        historyViewIndex === null
                          ? "bg-teal-700 text-white border-teal-800 shadow-xs"
                          : "bg-slate-100 text-slate-600 border-slate-200 hover:bg-slate-200"
                      }`}
                    >
                      ● Live Edit
                    </button>
                    {history.map((h, i) => (
                      <button
                        key={i}
                        type="button"
                        onClick={() => setHistoryViewIndex(i)}
                        className={`px-2 py-1 text-[11px] font-mono font-bold rounded-lg border transition cursor-pointer ${
                          historyViewIndex === i
                            ? "bg-teal-700 text-white border-teal-800 shadow-xs"
                            : "bg-slate-100 text-slate-600 border-slate-200 hover:bg-slate-200"
                        }`}
                        title={`View risk drivers for Run #${i + 1}`}
                      >
                        #{i + 1} ({h.prob.toFixed(0)}%)
                      </button>
                    ))}
                  </div>
                ) : null
              }
            />
          </div>

          <div className="lg:col-span-2 bg-white/95 backdrop-blur-md rounded-2xl shadow-card-elevated border border-slate-200 p-6 flex flex-col">
            <div className="flex items-start justify-between gap-3 mb-3">
              <div>
                <h3 className="text-lg font-bold text-slate-900">Prediction History</h3>
                <p className="text-xs text-slate-500">
                  {history.length > 0
                    ? `${history.length} saved run${history.length > 1 ? "s" : ""} on account. Click Risk Drivers to inspect.`
                    : "Each run is saved to your account with full risk drivers."}
                </p>
              </div>
              <div className="flex gap-1.5 shrink-0">
                <button type="button" onClick={exportHistoryCsv} className="text-[11px] font-semibold px-2 py-1 rounded-md border border-slate-300 bg-white hover:bg-slate-50 cursor-pointer">CSV</button>
                <button type="button" onClick={clearHistory} className="text-[11px] font-semibold px-2 py-1 rounded-md border border-slate-300 bg-white hover:bg-slate-50 cursor-pointer">Clear</button>
              </div>
            </div>
            <svg className="w-full h-14 mb-2" preserveAspectRatio="none" viewBox="0 0 200 50" aria-hidden="true">
              {history.length > 0 && (
                <>
                  <line x1="0" x2="200" y1={45 - 0.3 * 40} y2={45 - 0.3 * 40} stroke="#fcd34d" strokeDasharray="3 3" />
                  <line x1="0" x2="200" y1={45 - 0.65 * 40} y2={45 - 0.65 * 40} stroke="#fda4af" strokeDasharray="3 3" />
                  <polyline fill="none" stroke="#0f766e" strokeWidth="2" points={sparkPts.join(" ")} />
                  {sparkPts.map((p, i) => <circle key={i} cx={p.split(",")[0]} cy={p.split(",")[1]} r="2.5" fill="#0f766e" />)}
                </>
              )}
            </svg>
            <div className="space-y-2 overflow-y-auto max-h-[30rem] flex-grow pr-1">
              {historyLoading && history.length === 0 ? (
                <div className="flex items-center justify-center py-8 text-slate-400 text-xs gap-2">
                  <LoaderCircle size={16} className="spin text-teal-600" />
                  <span>Loading saved predictions...</span>
                </div>
              ) : history.length === 0 ? (
                <p className="text-xs text-slate-400 italic py-6 text-center">No runs yet. Press Run prediction.</p>
              ) : (
                history.map((h, i) => ({ h, i })).reverse().map(({ h, i }) => (
                  <div
                    key={i}
                    className={`rounded-xl border transition ${
                      expandedHistoryIndex === i || historyViewIndex === i
                        ? "border-teal-500 bg-white shadow-sm ring-1 ring-teal-200"
                        : "border-slate-200 bg-slate-50/80 hover:border-slate-300 hover:bg-white"
                    }`}
                  >
                    <div className="p-3 flex items-center justify-between gap-2 flex-wrap sm:flex-nowrap">
                      <div className="flex items-center gap-2 min-w-0">
                        <span className="font-mono text-slate-700 font-bold text-xs shrink-0">#{i + 1}</span>
                        <span className="text-slate-300">&middot;</span>
                        <span className="text-slate-500 font-mono text-[11px] shrink-0">{h.time}</span>
                        <span className="text-slate-700 truncate text-xs font-medium">
                          {h.numbers.age}y {h.numbers.sex ? "M" : "F"}, cp{h.numbers.cp}, ca{h.numbers.ca}
                        </span>
                      </div>
                      <div className="flex items-center gap-1.5 shrink-0">
                        <span className={`font-mono font-bold px-2 py-0.5 rounded text-xs ${TIER_UI[tierKey(h.prob)].pill.split(" border")[0]}`}>
                          {h.prob.toFixed(1)}%
                        </span>
                        <button
                          type="button"
                          onClick={() => {
                            setExpandedHistoryIndex(expandedHistoryIndex === i ? null : i);
                            setHistoryViewIndex(i);
                          }}
                          className={`px-2 py-1 text-[11px] font-bold rounded-lg border transition cursor-pointer ${
                            expandedHistoryIndex === i
                              ? "bg-teal-700 text-white border-teal-800 shadow-xs"
                              : "bg-white text-teal-800 border-teal-300 hover:bg-teal-50"
                          }`}
                        >
                          {expandedHistoryIndex === i ? "Hide" : "Risk Drivers"}
                        </button>
                        <button
                          type="button"
                          onClick={() => { setPatient({ ...h.inputs }); notify(`Restored run #${i + 1}`); window.scrollTo({ top: 0, behavior: "smooth" }); }}
                          className="px-2 py-1 text-[11px] font-semibold rounded-lg border border-slate-200 bg-white text-slate-600 hover:bg-slate-100 transition cursor-pointer"
                        >
                          Restore
                        </button>
                      </div>
                    </div>
                    {expandedHistoryIndex === i && (
                      <div className="border-t border-slate-100 p-2.5 bg-slate-50/50">
                        <RiskDriversVisualizer
                          drivers={liveDrivers(h.numbers)}
                          title={`Prediction #${i + 1} Risk Drivers`}
                          subtitle={`Ablation attribution for ${h.stamp} (${h.prob.toFixed(1)}% risk)`}
                          compact={true}
                        />
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>
        </section>
      </main>

      {/* Toast */}
      {toast && (
        <div className="fixed bottom-24 left-1/2 -translate-x-1/2 z-[60] pointer-events-none">
          <div className="mb-2 px-4 py-2 rounded-xl bg-slate-900 text-white text-xs font-semibold shadow-xl">{toast}</div>
        </div>
      )}

      {/* Results modal */}
      {modalOpen && result && (
        <div aria-modal="true" role="dialog" className={`fixed inset-0 z-50 overflow-y-auto transition-opacity duration-300 ${modalShown ? "opacity-100" : "opacity-0"}`}>
          <div className="fixed inset-0 bg-slate-900/65 backdrop-blur-md transition-opacity" onClick={closeModal} />
          <div className="flex min-h-full items-center justify-center p-3 sm:p-6 text-center">
            <div className="relative transform overflow-hidden rounded-2xl bg-white text-left shadow-2xl transition-all sm:my-8 sm:w-full sm:max-w-3xl border border-slate-200">
              <div className="px-6 py-5 bg-gradient-to-r from-[#0d2538] via-[#13354b] to-[#0f4457] text-white flex items-center justify-between border-b border-teal-800/40">
                <div className="flex items-center space-x-3">
                  <div className="p-2.5 rounded-xl bg-teal-500/20 border border-teal-400/30 text-teal-300 shadow-inner">
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" />
                    </svg>
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-white tracking-tight">Cardiovascular Risk Stratification Result</h3>
                    <p className="text-xs text-cyan-200/80">Inference completed • {result.stamp} • Models: Decision Tree + Bayesian Network</p>
                  </div>
                </div>
                <button type="button" aria-label="Close modal" onClick={closeModal} className="p-2 rounded-lg text-slate-300 hover:text-white hover:bg-white/10 transition cursor-pointer">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path d="M6 18L18 6M6 6l12 12" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" />
                  </svg>
                </button>
              </div>

              <div className="p-6 sm:p-7 space-y-6 max-h-[calc(85vh-130px)] overflow-y-auto">
                {/* Overall score card */}
                <div className="rounded-2xl border p-5 sm:p-6 bg-slate-50/70 border-slate-200">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-5">
                    <div className="flex items-center space-x-5">
                      <div className="relative w-24 h-24 shrink-0 flex items-center justify-center">
                        <svg className="w-24 h-24 -rotate-90 transform" viewBox="0 0 36 36">
                          <path className="text-slate-200" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="currentColor" strokeWidth="3.2" />
                          <path className={`${ui.circle} transition-all duration-1000 ease-out`} d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="currentColor" strokeDasharray={`${Math.min(result.prob, 100)}, 100`} strokeLinecap="round" strokeWidth="3.5" />
                        </svg>
                        <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
                          <span className="text-xl font-black tracking-tight text-slate-900 leading-none">{shownProb.toFixed(1)}%</span>
                          <span className="text-[9px] font-bold text-slate-400 uppercase tracking-wider mt-0.5">Prob</span>
                        </div>
                      </div>
                      <div>
                        <div className="flex items-center gap-2 mb-1.5 flex-wrap">
                          <span className={`inline-flex items-center px-3 py-0.5 rounded-full text-xs font-bold border gap-1.5 ${ui.badge}`}>
                            <span className="w-2 h-2 rounded-full bg-amber-500 animate-pulse" />
                            <span>{ui.badgeText}</span>
                          </span>
                          <span className="text-xs text-slate-500 font-mono">
                            {[result.dt !== null && `Decision Tree: ${result.dt}%`, result.bn !== null && `Bayesian Network: ${result.bn}%`].filter(Boolean).join(" • ")}
                          </span>
                        </div>
                        <h4 className="text-lg font-bold text-slate-900">{ui.headline(result.prob)}</h4>
                        <p className="text-xs text-slate-500 mt-0.5">{ui.sub}</p>
                      </div>
                    </div>
                  </div>

                  <div className="mt-5 pt-4 border-t border-slate-200/80">
                    <div className="flex justify-between text-[11px] font-bold mb-1.5">
                      <span className="text-emerald-700">Low Risk (&lt;30%)</span>
                      <span className="text-amber-700">Borderline / Moderate (30–65%)</span>
                      <span className="text-rose-700">Critical / High (&gt;65%)</span>
                    </div>
                    <div className="w-full h-3 bg-slate-200 rounded-full overflow-hidden flex relative shadow-inner">
                      <div className="w-[30%] bg-emerald-400" />
                      <div className="w-[35%] bg-amber-400" />
                      <div className="w-[35%] bg-rose-500" />
                      <div className="absolute top-0 bottom-0 w-2 bg-slate-900 rounded-full shadow-md transition-all duration-700 ring-2 ring-white" style={{ left: `${Math.min(Math.max(result.prob, 2), 97)}%` }} />
                    </div>
                  </div>
                </div>

                {/* Driver breakdown */}
                <div>
                  <RiskDriversVisualizer
                    drivers={liveDrivers(result.numbers)}
                    title="Prediction Risk Drivers"
                    subtitle="Percentage-point change versus healthy reference for this patient's prediction."
                  />
                </div>

                {/* Recommendations */}
                <div className="bg-gradient-to-br from-teal-50/70 to-cyan-50/50 rounded-xl p-5 border border-teal-200/90 shadow-2xs">
                  <h5 className="text-sm font-bold text-teal-950 flex items-center gap-2 mb-3">
                    <svg className="w-4 h-4 text-teal-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" />
                    </svg>
                    Recommended Clinical Action Items
                  </h5>
                  <ul className="space-y-2.5 text-xs text-slate-700">
                    {ui.recs.map(([title, text]) => (
                      <li key={title} className="flex items-start gap-2">
                        <span className={`${ui.dot} font-bold`}>●</span>
                        <span><strong>{title}</strong> {text}</span>
                      </li>
                    ))}
                  </ul>
                  <p className="text-[10px] text-slate-500 mt-3">Educational model output only. These items are not medical advice or a diagnosis.</p>
                </div>
              </div>

              <div className="px-6 py-4 bg-slate-50 border-t border-slate-200 flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <button type="button" onClick={exportReport} className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-semibold text-slate-700 bg-white hover:bg-slate-100 border border-slate-300 rounded-lg shadow-2xs transition cursor-pointer">
                    📄 Download Report
                  </button>
                  <button type="button" onClick={() => notify("Demo only: connect your EHR/FHIR endpoint to enable this")} className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-semibold text-teal-700 bg-teal-50 hover:bg-teal-100 border border-teal-200 rounded-lg transition cursor-pointer">
                    🏥 Send to EHR
                  </button>
                </div>
                <button type="button" onClick={closeModal} className="px-5 py-2 text-xs font-bold text-white bg-slate-800 hover:bg-slate-900 rounded-lg shadow-sm transition active:scale-95 cursor-pointer">
                  Done / Dismiss
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Chatbot widget */
/* ------------------------------------------------------------------ */

const REPORT_TYPES = [".pdf", ".txt", ".csv", ".png", ".jpg", ".jpeg"];
const REPORT_MAX_MB = 5;
const REPORT_DEFAULT_QUESTION = "Please analyse this report and summarise the key findings in simple terms.";

const CHAT_TABS = [
  ["chat", "💬", "Ask AI"],
  ["facts", "❤️", "Heart Facts & Anatomy"],
  ["about", "🛡️", "About Cardio AI"],
];

const FACTS = [
  {
    tone: "teal", icon: "⚡", title: "Oldpeak & ST-Depression",
    text: "During exercise stress testing, ischemia can impair ventricular repolarization, causing horizontal or downsloping ST depression. An oldpeak above about 1.5 mm is a strong marker of ischemia.",
    ask: "How does ST-depression (Oldpeak) indicate coronary artery disease?",
  },
  {
    tone: "rose", icon: "🔬", title: "Nuclear SPECT (Thal Perfusion)",
    text: "Thallium-201 scintigraphy shows myocardial blood flow. Thal 1 = normal perfusion, Thal 2 = fixed defect (prior scar or infarct), Thal 3 = reversible defect (viable but ischemic tissue).",
    ask: "What is the difference between Thalassemia 1, 2, and 3 scans?",
  },
  {
    tone: "amber", icon: "🏃", title: "Max HR Formula (220 - Age)",
    text: "Maximum heart rate decreases with age. Being unable to reach about 85% of the age-predicted maximum (chronotropic incompetence) is linked to higher cardiovascular risk.",
    ask: "How does age affect Maximum Heart Rate (thalach)?",
  },
  {
    tone: "indigo", icon: "🩻", title: "Fluoroscopy Colored Vessels (ca)",
    text: "Fluoroscopy can show calcification in the major coronary arteries (LAD, LCx, RCA). Seeing 1-3 calcified vessels is associated with a higher likelihood of significant coronary artery disease.",
    ask: "How does Coronary Fluoroscopy (ca) detect arterial occlusion?",
  },
  {
    tone: "emerald", icon: "💓", title: "Daily Cardiac Workload",
    text: "The adult heart beats roughly 100,000 times a day, pumping about 7,500 litres (2,000 gallons) of blood through a vascular network some 60,000 miles long.",
    ask: "How much blood does the heart pump each day?",
  },
];

const ABOUT_SPECS = [
  ["Prediction Models", "Decision Tree + Bayesian Network", "Combined score = average of both"],
  ["Input Features", "13 Clinical Biomarkers", "Coded using your model's categories"],
  ["Live Preview", "Heuristic Risk Drivers", "Quick estimate, not the trained models"],
  ["Intended Use", "Educational Project", "Not a medical device"],
];

const CHIPS = [
  ["🧾 Explain my result", "explain"],
  ["🫀 What is oldpeak?", "What is oldpeak (ST depression) and why does it matter?"],
  ["🩺 Lower risk", "Which lifestyle factors are generally linked with lower heart disease risk?"],
];

const DEFAULT_WELCOME = [
  {
    role: "assistant",
    text: "Hello! I'm Cardio AI Assistant. How can I help you today?",
  },
];

function ChatbotWidget({ getContext, user }) {
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [tab, setTab] = useState("chat");
  const [attachment, setAttachment] = useState(null);

  const chatStorageKey = user?.id ? `cardio_ai_chat_${user.id}` : "cardio_ai_chat_guest";

  const [messages, setMessages] = useState(() => {
    try {
      const saved = localStorage.getItem(chatStorageKey);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed) && parsed.length > 0) return parsed;
      }
    } catch {}
    return DEFAULT_WELCOME;
  });

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    let active = true;
    const auth = loadAuth();
    if (auth?.token && user && !user.is_anonymous) {
      fetch(`${API_URL}/chat/history?limit=100`, { headers: authHeaders() })
        .then((res) => (res.ok ? res.json() : null))
        .then((data) => {
          if (!active || !data?.messages) return;
          if (Array.isArray(data.messages) && data.messages.length > 0) {
            const formatted = data.messages.map((m) => ({ role: m.role, text: m.text }));
            setMessages(formatted);
            try {
              localStorage.setItem(chatStorageKey, JSON.stringify(formatted));
            } catch {}
          }
        })
        .catch((err) => console.warn("Could not sync chat history from server:", err));
    }
    return () => {
      active = false;
    };
  }, [user?.id, chatStorageKey]);

  useEffect(() => {
    try {
      localStorage.setItem(chatStorageKey, JSON.stringify(messages));
    } catch {}
  }, [messages, chatStorageKey]);

  async function clearChatHistory() {
    if (!window.confirm("Are you sure you want to clear your chat history?")) return;
    setMessages(DEFAULT_WELCOME);
    try {
      localStorage.removeItem(chatStorageKey);
    } catch {}
    const auth = loadAuth();
    if (auth?.token && user && !user.is_anonymous) {
      try {
        await fetch(`${API_URL}/chat/history`, {
          method: "DELETE",
          headers: authHeaders(),
        });
      } catch (err) {
        console.warn("Could not clear server chat history:", err);
      }
    }
  }

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, busy, open]);

  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open]);

  useEffect(() => {
    const onKey = (e) => e.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  function addAssistantMessage(text) {
    setMessages((previous) => [...previous, { role: "assistant", text }]);
  }

  function pickReport(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    const extension = file.name.includes(".") ? `.${file.name.split(".").pop().toLowerCase()}` : "";

    if (!REPORT_TYPES.includes(extension)) {
      addAssistantMessage(`That file type isn't supported. Please attach one of: ${REPORT_TYPES.join(", ")}.`);
      event.target.value = "";
      return;
    }

    if (file.size > REPORT_MAX_MB * 1024 * 1024) {
      addAssistantMessage(`That file is larger than ${REPORT_MAX_MB} MB. Please attach a smaller report.`);
      event.target.value = "";
      return;
    }

    setAttachment(file);
    event.target.value = "";
    inputRef.current?.focus();
  }

  async function send(question, file = null) {
    const text = question.trim() || (file ? REPORT_DEFAULT_QUESTION : "");
    if (!text || busy) return;

    const history = messages.map(({ role, text: content }) => ({ role, content }));

    setMessages((previous) => [
      ...previous,
      { role: "user", text: file ? `📎 ${file.name}\n${text}` : text },
    ]);
    setInput("");
    setAttachment(null);
    setBusy(true);

    try {
      let response;

      if (file) {
        /* report upload: multipart form, analysed by POST /chat/report */
        const form = new FormData();
        form.append("file", file);
        form.append("message", text);
        form.append("history", JSON.stringify(history));

        response = await fetch(`${API_URL}/chat/report`, { method: "POST", headers: authHeaders(), body: form });
      } else {
        response = await fetch(`${API_URL}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json", ...authHeaders() },
          body: JSON.stringify({ message: text, history }),
        });
      }

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : `Chat request failed (${response.status}).`
        );
      }

      const answer = data.answer ?? data.response;

      if (typeof answer !== "string" || !answer.trim()) {
        throw new Error("The chat service returned no answer.");
      }

      addAssistantMessage(answer);
    } catch (error) {
      console.error("Cardio AI chat error:", error);
      addAssistantMessage(
        file
          ? `I couldn't analyse that report. ${error.message}`
          : "I couldn't reach the chat service. Please check that " +
              "FastAPI is running and the /chat endpoint is available."
      );
    } finally {
      setBusy(false);
    }
  }

  function sendMessage(event) {
    event.preventDefault();
    send(input, attachment);
  }

  function askFact(question) {
    setTab("chat");
    send(question);
  }

  return (
    <>
      {open && (
        <section
          className="chat-window"
          role="dialog"
          aria-label="Cardio AI Assistant"
          aria-modal="false"
        >
          <header className="chat-header">
            <div className="chat-title">
              <span className="chat-status-dot" />
              <div>
                <h2>Cardio AI Assistant</h2>
                <p>Heart-health educational support</p>
              </div>
            </div>
            <div className="chat-header-actions">
              <button
                className="chat-clear-btn"
                type="button"
                onClick={clearChatHistory}
                aria-label="Clear chat history"
                title="Clear conversation history"
              >
                <Trash2 size={16} />
              </button>
              <button
                className="chat-close"
                type="button"
                onClick={() => setOpen(false)}
                aria-label="Close chat"
                title="Close chat"
              >
                <X size={21} />
              </button>
            </div>
          </header>

          <div className="chat-tabs" role="tablist" aria-label="Cardio AI sections">
            {CHAT_TABS.map(([id, icon, label]) => (
              <button
                key={id}
                type="button"
                role="tab"
                aria-selected={tab === id}
                className={`chat-tab ${tab === id ? "active" : ""}`}
                onClick={() => setTab(id)}
              >
                <span aria-hidden="true">{icon}</span> {label}
              </button>
            ))}
          </div>

          {tab === "chat" && (
            <div className="chat-panel chat-panel-chat">
            <div className="chat-welcome">
              <p>How can I help you today?</p>
              <span>Ask follow-up questions to continue your conversation.</span>
            </div>

            <div className="chat-messages" aria-live="polite">
              {messages.map((message, index) => (
                <div key={`${index}-${message.role}`} className={`chat-message ${message.role}`}>
                  {message.text}
                </div>
              ))}

              {busy && (
                <div className="chat-message assistant typing">
                  <LoaderCircle size={16} className="spin" />
                  Thinking...
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            <div className="chat-chips">
              {CHIPS.map(([label, text]) => (
                <button
                  key={label}
                  type="button"
                  disabled={busy}
                  onClick={() => send(text === "explain" ? getContext() : text)}
                >
                  {label}
                </button>
              ))}
            </div>

            {attachment && (
              <div className="chat-attachment">
                <span aria-hidden="true">📎</span>
                <span className="chat-attachment-name">{attachment.name}</span>
                <small>{Math.max(1, Math.round(attachment.size / 1024))} KB</small>
                <button type="button" onClick={() => setAttachment(null)} aria-label="Remove attachment" title="Remove attachment">
                  <X size={14} />
                </button>
              </div>
            )}

            <form className="chat-form" onSubmit={sendMessage}>
              <input
                ref={fileInputRef}
                type="file"
                accept={REPORT_TYPES.join(",")}
                onChange={pickReport}
                hidden
              />
              <button
                type="button"
                className="chat-attach"
                onClick={() => fileInputRef.current?.click()}
                disabled={busy}
                aria-label="Attach a report"
                title={`Attach a report (${REPORT_TYPES.join(", ")}, max ${REPORT_MAX_MB} MB)`}
              >
                <Plus size={20} />
              </button>
              <input
                ref={inputRef}
                value={input}
                onChange={(event) => setInput(event.target.value)}
                placeholder={attachment ? "Ask about this report (optional)..." : "Message Cardio AI..."}
                aria-label="Your message"
                maxLength={2000}
                disabled={busy}
              />
              <button
                type="submit"
                disabled={(!input.trim() && !attachment) || busy}
                aria-label="Send message"
                title="Send message"
              >
                {busy ? <LoaderCircle size={19} className="spin" /> : <Send size={19} />}
              </button>
            </form>

            <p className="chat-disclaimer">
              Educational information only, not a medical diagnosis.
            </p>
            </div>
          )}

          {tab === "facts" && (
            <div className="chat-panel chat-scroll">
              <div className="facts-banner">
                <h4>❤️ Clinical Cardiology &amp; Heart Anatomy</h4>
                <p>
                  Key anatomical and physiological ideas behind the 13 input features.
                  Click any card to ask Cardio AI about it.
                </p>
              </div>

              {FACTS.map((fact) => (
                <button
                  key={fact.title}
                  type="button"
                  className={`fact-card tone-${fact.tone}`}
                  onClick={() => askFact(fact.ask)}
                  disabled={busy}
                >
                  <span className="fact-head">
                    <span>{fact.icon} {fact.title}</span>
                    <span className="fact-link">Ask AI →</span>
                  </span>
                  <span className="fact-text">{fact.text}</span>
                </button>
              ))}
            </div>
          )}

          {tab === "about" && (
            <div className="chat-panel chat-scroll">
              <div className="about-card">
                <h4>About Cardio AI Platform</h4>
                <span className="about-sub">Heart disease risk prediction</span>
                <p>
                  Cardio AI estimates heart disease risk from 13 clinical inputs using a
                  Decision Tree and a Bayesian Network, and shows both scores together
                  with a combined result.
                </p>
              </div>

              <div className="spec-grid">
                {ABOUT_SPECS.map(([label, value, note]) => (
                  <div className="spec-card" key={label}>
                    <span>{label}</span>
                    <strong>{value}</strong>
                    <em>{note}</em>
                  </div>
                ))}
              </div>

              <div className="safety-card">
                <strong>🛡️ Ethical AI &amp; Clinical Safety</strong>
                <p>
                  Cardio AI is an educational project. It does not replace the judgment of a
                  qualified clinician and must not be used for diagnosis or treatment decisions.
                </p>
              </div>

              <button type="button" className="about-ask" onClick={() => askFact("How does Cardio AI estimate heart disease risk?")} disabled={busy}>
                Ask Cardio AI about this →
              </button>
            </div>
          )}
        </section>
      )}

      <button
        className={`heart-fab ${open ? "active" : ""}`}
        type="button"
        onClick={() => setOpen((previous) => !previous)}
        aria-label={open ? "Close Cardio AI" : "Open Cardio AI"}
        aria-expanded={open}
        title={open ? "Close assistant" : "Chat with Cardio AI"}
      >
        {open ? <X size={27} /> : <HeartPulse className="heart-beat-icon" size={35} />}
      </button>
    </>
  );
}

/* ------------------------------------------------------------------ */
/* Login / sign-up page                                                */
/* ------------------------------------------------------------------ */

const AUTH_INPUT =
  "block w-full rounded-xl border border-slate-300 bg-slate-50/50 py-2.5 px-3.5 text-slate-900 text-sm font-medium transition-all duration-200 hover:border-slate-400 focus:bg-white focus:border-teal-600 focus:ring-4 focus:ring-teal-500/15 focus:outline-none";

function getPasswordStrength(password) {
  if (!password) return { score: 0, label: "", color: "bg-slate-200", text: "text-slate-400" };
  let score = 0;
  if (password.length >= 8) score++;
  if (/[0-9]/.test(password)) score++;
  if (/[a-z]/.test(password) && /[A-Z]/.test(password)) score++;
  if (/[^a-zA-Z0-9]/.test(password)) score++;

  if (score <= 1) return { score: 1, label: "Weak", color: "bg-rose-500", text: "text-rose-600" };
  if (score === 2) return { score: 2, label: "Fair", color: "bg-amber-500", text: "text-amber-600" };
  if (score === 3) return { score: 3, label: "Good", color: "bg-teal-500", text: "text-teal-600" };
  return { score: 4, label: "Strong", color: "bg-emerald-500", text: "text-emerald-600" };
}

function AuthPage({ onAuth, onClose, initialMode = "login" }) {
  const [mode, setMode] = useState(initialMode);
  const [form, setForm] = useState({ name: "", email: "", password: "", confirm: "" });
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [loading, setLoading] = useState(false);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });

  useEffect(() => {
    if (initialMode) {
      setMode(initialMode);
      setError("");
      setSuccess("");
    }
  }, [initialMode]);

  const isRegister = mode === "register";
  const setField = (key) => (event) => setForm((f) => ({ ...f, [key]: event.target.value }));

  const isValidEmail = /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.email.trim());
  const pwdStrength = getPasswordStrength(form.password);
  const isConfirmMatching = form.confirm.length > 0 && form.password === form.confirm;
  const isConfirmMismatch = form.confirm.length > 0 && form.password !== form.confirm;

  function switchMode(next) {
    setMode(next);
    setError("");
    setSuccess("");
  }

  function handleMouseMove(e) {
    const rect = e.currentTarget.getBoundingClientRect();
    setMousePos({ x: e.clientX - rect.left, y: e.clientY - rect.top });
  }

  async function startFreeTrial() {
    setError("");
    setSuccess("");
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/auth/free-trial`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.detail || "Unable to start the free trial.");
      }
      onAuth({
        token: data.access_token,
        user: data.user,
      });
      onClose?.();
    } catch (err) {
      console.error("Cardio AI free trial error:", err);
      const isNetworkError = err instanceof TypeError || err.message?.includes("fetch");
      setError(
        isNetworkError
          ? `Cannot reach backend server at ${API_URL}. Please ensure the backend is running.`
          : err.message || "Unable to start the free trial."
      );
    } finally {
      setLoading(false);
    }
  }

  function validate() {
    if (isRegister && !form.name.trim()) return "Please enter your name.";
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.email.trim())) return "Please enter a valid email address.";
    if (form.password.length < 8) return "Password must be at least 8 characters.";
    if (isRegister && form.password !== form.confirm) return "The two passwords do not match.";
    return "";
  }

  async function submit(event) {
    event.preventDefault();
    setError("");
    setSuccess("");

    const problem = validate();
    if (problem) {
      setError(problem);
      return;
    }

    setLoading(true);

    try {
      const email = form.email.trim();

      if (isRegister) {
        const res = await fetch(`${API_URL}/auth/register`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            email,
            password: form.password,
            full_name: form.name.trim() || "User",
          }),
        });

        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          throw new Error(data.detail || "Registration failed. Please try again.");
        }

        // Automatic authentication into dashboard upon registration
        if (data.access_token && data.user) {
          onAuth({
            token: data.access_token,
            user: data.user,
          });
          onClose?.();
          return;
        }

        // Switch to login tab, prefill email, clear passwords, and show success message
        setMode("login");
        setForm((prev) => ({ ...prev, password: "", confirm: "" }));
        setSuccess("Account created successfully! Please log in with your credentials.");
        return;
      }

      // Log in flow
      const res = await fetch(`${API_URL}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email,
          password: form.password,
        }),
      });

      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.detail || "Invalid email or password.");
      }

      onAuth({
        token: data.access_token,
        user: data.user,
      });
      onClose?.();
    } catch (err) {
      console.error("Cardio AI auth error:", err);
      const isNetworkError = err instanceof TypeError || err.message?.includes("fetch");
      setError(
        isNetworkError
          ? `Cannot reach backend server at ${API_URL}. Please start the backend with ".venv/bin/uvicorn api:app --reload --host 127.0.0.1 --port 8000" and try again.`
          : err.message || "Authentication failed. Please try again."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className={onClose ? "w-full max-w-4xl relative overflow-hidden font-sans text-slate-800 antialiased my-auto" : "min-h-screen flex items-center justify-center p-4 sm:p-6 lg:p-8 bg-[#edf4f8] relative overflow-hidden font-sans text-slate-800 antialiased"}>
      {/* Ambient background orbs */}
      {!onClose && (
        <div aria-hidden="true" className="fixed inset-0 pointer-events-none overflow-hidden">
          <div className="orb-1 aurora-blur absolute -top-40 -left-40 w-[600px] h-[600px] rounded-full bg-teal-300/35 blur-[130px]" />
          <div className="orb-2 aurora-blur absolute top-1/3 -right-40 w-[550px] h-[550px] rounded-full bg-cyan-300/30 blur-[140px]" />
          <div className="orb-3 aurora-blur absolute -bottom-40 left-1/4 w-[650px] h-[650px] rounded-full bg-blue-300/25 blur-[140px]" />
        </div>
      )}

      {/* Main interactive card container */}
      <div
        onMouseMove={handleMouseMove}
        className="group relative z-10 w-full max-w-4xl grid grid-cols-1 lg:grid-cols-5 bg-white/95 backdrop-blur-xl rounded-3xl shadow-card-elevated border border-slate-200/90 overflow-hidden transition-all duration-300"
      >
        {onClose && (
          <button
            type="button"
            onClick={onClose}
            aria-label="Close authentication modal"
            className="absolute top-4 right-4 z-40 w-9 h-9 rounded-full bg-slate-100 hover:bg-slate-200 text-slate-500 hover:text-slate-800 flex items-center justify-center transition cursor-pointer shadow-xs"
          >
            <X size={18} />
          </button>
        )}
        {/* Dynamic mouse spotlight */}
        <div
          className="pointer-events-none absolute -inset-px rounded-3xl opacity-0 group-hover:opacity-100 transition-opacity duration-300 z-30"
          style={{
            background: `radial-gradient(550px circle at ${mousePos.x}px ${mousePos.y}px, rgba(20, 184, 166, 0.08), transparent 40%)`,
          }}
        />

        {/* ------------------------------------------------------------ */}
        {/* Left Brand Panel                                             */}
        {/* ------------------------------------------------------------ */}
        <div className="lg:col-span-2 bg-gradient-to-br from-[#0a1e2e] via-[#0d2f44] to-[#0a3848] text-white p-7 sm:p-8 flex flex-col justify-between gap-6 relative overflow-hidden">
          {/* Subtle ambient lighting */}
          <div className="pointer-events-none absolute -bottom-24 -right-24 w-72 h-72 bg-teal-500/15 rounded-full blur-3xl group-hover:bg-teal-500/25 transition-all duration-700" />
          <div className="pointer-events-none absolute -top-20 -left-20 w-60 h-60 bg-cyan-500/10 rounded-full blur-2xl" />

          {/* Brand header */}
          <div className="relative z-10">
            <div className="flex items-center gap-3">
              <div className="w-14 h-14 rounded-2xl bg-white/10 backdrop-blur-md p-2 border border-white/20 shadow-xl flex items-center justify-center transform transition-transform duration-300 group-hover:scale-105">
                <img alt="Cardio AI Logo" className="w-full h-full object-contain scale-110 drop-shadow-sm" src={logo} />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h1 className="text-2xl font-black tracking-tight text-white">Cardio AI</h1>
                  <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold tracking-wide bg-teal-400/20 text-teal-200 border border-teal-300/30">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                    v2.5
                  </span>
                </div>
                <p className="text-xs text-cyan-200/80 font-medium">Dual-AI Diagnostic Platform</p>
              </div>
            </div>

            <p className="mt-3.5 text-xs text-slate-300 leading-relaxed">
              AI-powered heart disease risk assessment &amp; multi-model predictive stratification.
            </p>

            {/* Central High-Tech Telemetry Widget */}
            <div className="mt-5 rounded-2xl bg-white/[0.07] border border-white/15 p-3.5 backdrop-blur-md space-y-2.5 shadow-inner">
              <div className="flex items-center justify-between text-[11px] font-semibold text-slate-200">
                <span className="inline-flex items-center gap-1.5 text-rose-300">
                  <HeartPulse size={14} className="animate-pulse text-rose-400" />
                  <span>74 BPM</span>
                </span>
                <span className="inline-flex items-center gap-1 text-[10px] text-emerald-300 bg-emerald-500/20 px-2 py-0.5 rounded-full border border-emerald-400/30 font-bold">
                  ● Normal Sinus
                </span>
              </div>

              {/* Animated ECG wave */}
              <div className="h-7 w-full overflow-hidden flex items-center">
                <svg className="w-full h-7 text-teal-300/90" viewBox="0 0 200 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M0 12h40l4-7 6 15 5-11 4 4 5-1h40l4-8 5 16 6-11 3 3h40l4-7 5 15 5-11 3 3h20" />
                </svg>
              </div>

              {/* Live metrics strip */}
              <div className="grid grid-cols-2 gap-2 pt-1 border-t border-white/10 text-[10px]">
                <div className="bg-white/5 rounded-lg p-1.5 text-center">
                  <p className="text-slate-400">DT Test Accuracy</p>
                  <p className="font-extrabold text-white">77.8% Accurate</p>
                </div>
                <div className="bg-white/5 rounded-lg p-1.5 text-center">
                  <p className="text-slate-400">Inference Engine</p>
                  <p className="font-extrabold text-teal-300">Bayesian + DT</p>
                </div>
              </div>
            </div>
          </div>

          {/* Interactive Feature Cards */}
          <div className="space-y-2 relative z-10">
            <div className="p-2.5 rounded-xl bg-white/[0.04] hover:bg-white/[0.1] border border-white/10 transition-all duration-200 flex items-start gap-2.5 cursor-default">
              <div className="w-7 h-7 rounded-lg bg-teal-500/20 border border-teal-400/30 flex items-center justify-center text-teal-300 shrink-0 mt-0.5">
                <Activity size={14} />
              </div>
              <div className="text-xs">
                <p className="font-bold text-white">13 Biometric Parameters</p>
                <p className="text-[11px] text-slate-300 leading-snug">Continuous &amp; discrete patient feature mapping.</p>
              </div>
            </div>

            <div className="p-2.5 rounded-xl bg-white/[0.04] hover:bg-white/[0.1] border border-white/10 transition-all duration-200 flex items-start gap-2.5 cursor-default">
              <div className="w-7 h-7 rounded-lg bg-cyan-500/20 border border-cyan-400/30 flex items-center justify-center text-cyan-300 shrink-0 mt-0.5">
                <Sparkles size={14} />
              </div>
              <div className="text-xs">
                <p className="font-bold text-white">Explainable Risk Drivers</p>
                <p className="text-[11px] text-slate-300 leading-snug">Visual feature ablation attribution breakdown.</p>
              </div>
            </div>
          </div>


        </div>

        {/* ------------------------------------------------------------ */}
        {/* Right Form Panel                                             */}
        {/* ------------------------------------------------------------ */}
        <div className="lg:col-span-3 p-7 sm:p-9 flex flex-col justify-between">
          <div>
            {/* Mode switch tabs */}
            <div className="flex items-center justify-between gap-2">
              <div className="inline-flex rounded-xl bg-slate-100 p-1 text-xs font-bold" role="tablist">
                {[["login", "Log in"], ["register", "Create account"]].map(([id, label]) => (
                  <button
                    key={id}
                    type="button"
                    role="tab"
                    aria-selected={mode === id}
                    onClick={() => switchMode(id)}
                    className={`px-4 py-2 rounded-lg transition-all duration-200 cursor-pointer ${
                      mode === id
                        ? "bg-white text-teal-900 shadow-sm font-extrabold"
                        : "text-slate-500 hover:text-slate-800 hover:bg-white/50"
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>

            {/* Header text */}
            <div className="mt-5">
              <h2 className="text-2xl font-black text-slate-900 tracking-tight">
                {isRegister ? "Create Account" : "Welcome back"}
              </h2>
              <p className="text-xs sm:text-sm text-slate-500 mt-1">
                {isRegister
                  ? "Register your account to access heart disease risk assessments."
                  : "Enter your credentials to access your dashboard."}
              </p>
            </div>

            {/* Form */}
            <form className="mt-5 space-y-3.5" onSubmit={submit} noValidate>
              {/* Name field (register only) */}
              {isRegister && (
                <div>
                  <label className="block text-[11px] font-bold uppercase tracking-wider text-slate-700 mb-1" htmlFor="auth-name">
                    Full name
                  </label>
                  <div className="relative">
                    <div className="pointer-events-none absolute inset-y-0 left-0 pl-3.5 flex items-center text-slate-400">
                      <User size={16} />
                    </div>
                    <input
                      id="auth-name"
                      className={`${AUTH_INPUT} pl-10`}
                      type="text"
                      autoComplete="name"
                      value={form.name}
                      onChange={setField("name")}
                      placeholder="Alex Morgan"
                    />
                  </div>
                </div>
              )}

              {/* Email field */}
              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="block text-[11px] font-bold uppercase tracking-wider text-slate-700" htmlFor="auth-email">
                    Email address
                  </label>
                  {isValidEmail && (
                    <span className="inline-flex items-center gap-1 text-[11px] font-bold text-emerald-600">
                      <Check size={12} /> Valid format
                    </span>
                  )}
                </div>
                <div className="relative">
                  <div className="pointer-events-none absolute inset-y-0 left-0 pl-3.5 flex items-center text-slate-400">
                    <Mail size={16} />
                  </div>
                  <input
                    id="auth-email"
                    className={`${AUTH_INPUT} pl-10 ${isValidEmail ? "border-emerald-400 focus:border-emerald-500 focus:ring-emerald-500/15" : ""}`}
                    type="email"
                    autoComplete="email"
                    value={form.email}
                    onChange={setField("email")}
                    placeholder="user@example.com"
                  />
                </div>
              </div>

              {/* Password field */}
              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="block text-[11px] font-bold uppercase tracking-wider text-slate-700" htmlFor="auth-password">
                    Password
                  </label>
                  {isRegister && pwdStrength.score > 0 && (
                    <span className={`text-[11px] font-bold ${pwdStrength.text}`}>
                      Strength: {pwdStrength.label}
                    </span>
                  )}
                </div>
                <div className="relative">
                  <div className="pointer-events-none absolute inset-y-0 left-0 pl-3.5 flex items-center text-slate-400">
                    <Lock size={16} />
                  </div>
                  <input
                    id="auth-password"
                    className={`${AUTH_INPUT} pl-10 pr-11`}
                    type={showPassword ? "text" : "password"}
                    autoComplete={isRegister ? "new-password" : "current-password"}
                    value={form.password}
                    onChange={setField("password")}
                    placeholder={isRegister ? "At least 8 characters" : "Enter your password"}
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword((v) => !v)}
                    aria-label={showPassword ? "Hide password" : "Show password"}
                    className="absolute inset-y-0 right-0 px-3.5 text-slate-400 hover:text-slate-700 transition cursor-pointer flex items-center"
                  >
                    {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                  </button>
                </div>

                {/* Password strength progress bar */}
                {isRegister && form.password.length > 0 && (
                  <div className="mt-2 space-y-1">
                    <div className="grid grid-cols-4 gap-1.5 h-1.5">
                      {[1, 2, 3, 4].map((step) => (
                        <div
                          key={step}
                          className={`rounded-full transition-all duration-300 ${
                            step <= pwdStrength.score ? pwdStrength.color : "bg-slate-200"
                          }`}
                        />
                      ))}
                    </div>
                    <p className="text-[10px] text-slate-400">
                      Use 8+ characters with uppercase letters, numbers, and symbols.
                    </p>
                  </div>
                )}
              </div>

              {/* Confirm password (register only) */}
              {isRegister && (
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <label className="block text-[11px] font-bold uppercase tracking-wider text-slate-700" htmlFor="auth-confirm">
                      Confirm password
                    </label>
                    {isConfirmMatching && (
                      <span className="inline-flex items-center gap-1 text-[11px] font-bold text-emerald-600">
                        <Check size={12} /> Passwords match
                      </span>
                    )}
                    {isConfirmMismatch && (
                      <span className="text-[11px] font-medium text-rose-500">
                        Passwords do not match
                      </span>
                    )}
                  </div>
                  <div className="relative">
                    <div className="pointer-events-none absolute inset-y-0 left-0 pl-3.5 flex items-center text-slate-400">
                      <Lock size={16} />
                    </div>
                    <input
                      id="auth-confirm"
                      className={`${AUTH_INPUT} pl-10 ${
                        isConfirmMatching
                          ? "border-emerald-400 focus:border-emerald-500 focus:ring-emerald-500/15"
                          : isConfirmMismatch
                          ? "border-rose-300 focus:border-rose-400 focus:ring-rose-400/15"
                          : ""
                      }`}
                      type={showPassword ? "text" : "password"}
                      autoComplete="new-password"
                      value={form.confirm}
                      onChange={setField("confirm")}
                      placeholder="Re-enter password"
                    />
                  </div>
                </div>
              )}

              {/* Success banner */}
              {success && (
                <div role="status" className="p-3.5 rounded-xl bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs font-semibold flex items-start gap-2.5 shadow-2xs">
                  <Check size={16} className="text-emerald-600 shrink-0 mt-0.5" />
                  <p className="font-extrabold text-emerald-900 leading-snug">{success}</p>
                </div>
              )}

              {/* Error banner */}
              {error && (
                <div role="alert" className="p-3 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs font-medium">
                  {error}
                </div>
              )}

              {/* Primary action button */}
              <button
                type="submit"
                disabled={loading}
                className="group w-full inline-flex items-center justify-center gap-2 px-6 py-3.5 rounded-xl bg-gradient-to-r from-[#0d4f5c] via-[#0f5c6b] to-[#0a424e] hover:from-[#0b444f] hover:to-[#08353f] active:scale-[0.98] text-white font-extrabold text-sm shadow-md hover:shadow-glow-teal transition-all duration-200 cursor-pointer disabled:opacity-70 disabled:cursor-wait mt-2"
              >
                {loading ? (
                  <LoaderCircle size={18} className="spin" />
                ) : (
                  <>
                    <span>{isRegister ? "Create Account" : "Log In"}</span>
                    <ArrowRight size={17} className="transition-transform duration-200 group-hover:translate-x-1.5" />
                  </>
                )}
                {loading && <span>Authenticating...</span>}
              </button>
            </form>

            {/* Divider */}
            <div className="relative my-4">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-slate-200" />
              </div>
              <div className="relative flex justify-center text-[10px] font-bold uppercase tracking-wider text-slate-400">
                <span className="bg-white px-2.5">or explore without an account</span>
              </div>
            </div>

            {/* Instant Free Trial button */}
            <button
              type="button"
              onClick={startFreeTrial}
              disabled={loading}
              className="group w-full inline-flex items-center justify-between p-3 rounded-xl border border-teal-200 bg-gradient-to-r from-teal-50/80 to-emerald-50/60 hover:from-teal-100/90 hover:to-emerald-100/80 active:scale-[0.98] transition-all duration-200 shadow-2xs hover:border-teal-300 disabled:opacity-60 disabled:cursor-wait cursor-pointer"
            >
              <div className="flex items-center gap-2.5 text-left">
                <div className="w-8 h-8 rounded-lg bg-teal-500/15 border border-teal-400/30 flex items-center justify-center text-teal-700 shrink-0">
                  <HeartPulse size={17} className="text-teal-600 transition-transform duration-200 group-hover:scale-115" />
                </div>
                <div>
                  <p className="text-xs font-extrabold text-teal-900 leading-tight">Start Instant Free Trial</p>
                  <p className="text-[10px] text-teal-700/80">Anonymous guest session • Full prediction access</p>
                </div>
              </div>
              <ArrowRight size={15} className="text-teal-700 transition-transform duration-200 group-hover:translate-x-1 shrink-0 mr-1" />
            </button>

            {/* Mode switch helper link */}
            <p className="mt-3.5 text-center text-xs text-slate-500">
              {isRegister ? "Already registered?" : "New to Cardio AI?"}{" "}
              <button
                type="button"
                onClick={() => switchMode(isRegister ? "login" : "register")}
                className="font-bold text-teal-700 hover:text-teal-900 transition-colors cursor-pointer underline underline-offset-2"
              >
                {isRegister ? "Log in here" : "Create an account"}
              </button>
            </p>
          </div>

          {/* Educational disclaimer */}
          <p className="mt-5 text-[10px] text-slate-400 leading-relaxed text-center">
            Cardio AI is an educational diagnostic research tool and not a certified medical device.
          </p>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* App                                                                 */
/* ------------------------------------------------------------------ */

export default function App() {
  const [auth, setAuth] = useState(() => loadAuth());
  const [checking, setChecking] = useState(() => Boolean(loadAuth()));
  const [authModal, setAuthModal] = useState(null); // null | "login" | "register"

  const chatContext = useRef("Please explain in simple terms what the 13 patient input fields mean.");

  const handleContext = useCallback((text) => {
    chatContext.current = text;
  }, []);

  const handleAuth = useCallback((next) => {
    saveAuth(next);
    setAuth(next);
    setAuthModal(null);
  }, []);

  const handleLogout = useCallback(() => {
    const saved = loadAuth();
    if (saved?.token) {
      fetch(`${API_URL}/auth/logout`, {
        method: "POST",
        headers: { Authorization: `Bearer ${saved.token}` },
      }).catch(() => {});
    }
    clearAuth();
    setAuth(null);
    setAuthModal(null);
  }, []);

  /* Restore and verify session from Python backend on load */
  useEffect(() => {
    let active = true;
    const saved = loadAuth();

    if (!saved?.token) {
      setChecking(false);
      return undefined;
    }

    fetch(`${API_URL}/auth/me`, {
      headers: { Authorization: `Bearer ${saved.token}` },
    })
      .then((res) => {
        if (!res.ok) throw new Error("Session expired");
        return res.json();
      })
      .then((data) => {
        if (!active) return;
        if (data?.user) {
          const next = {
            token: saved.token,
            user: data.user,
          };
          saveAuth(next);
          setAuth(next);
        } else {
          clearAuth();
          setAuth(null);
        }
      })
      .catch(() => {
        if (!active) return;
        clearAuth();
        setAuth(null);
      })
      .finally(() => {
        if (active) setChecking(false);
      });

    return () => {
      active = false;
    };
  }, []);

  if (checking) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#edf4f8] text-teal-700">
        <LoaderCircle size={32} className="spin" />
      </div>
    );
  }

  if (!auth) {
    return <AuthPage onAuth={handleAuth} initialMode="login" />;
  }

  return (
    <>
      <PredictionPage
        onContext={handleContext}
        user={auth.user}
        onLogout={handleLogout}
        onOpenAuth={(mode) => setAuthModal(mode || "login")}
      />
      <ChatbotWidget getContext={() => chatContext.current} user={auth.user} />

      {authModal && (
        <div
          role="dialog"
          aria-modal="true"
          className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-6 bg-slate-900/65 backdrop-blur-md overflow-y-auto"
        >
          <AuthPage
            initialMode={authModal}
            onClose={() => setAuthModal(null)}
            onAuth={handleAuth}
          />
        </div>
      )}
    </>
  );
}
