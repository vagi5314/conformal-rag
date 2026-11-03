/* api.js — fetch wrapper for the conformal-rag REST API (same origin) */

const API = (() => {
  async function request(method, path, body) {
    const options = { method, headers: {} };
    if (body !== undefined) {
      options.headers["Content-Type"] = "application/json";
      options.body = JSON.stringify(body);
    }
    const res = await fetch(path, options);
    if (!res.ok) {
      let detail = `${res.status} ${res.statusText}`;
      try {
        const err = await res.json();
        if (err.detail) {
          detail = typeof err.detail === "string"
            ? err.detail
            : JSON.stringify(err.detail);
        }
      } catch (_) { /* non-JSON error body */ }
      throw new Error(detail);
    }
    return res.json();
  }

  const get = (path) => request("GET", path);
  const post = (path, body) => request("POST", path, body);

  return {
    health: () => get("/health"),
    getConfig: () => get("/config"),
    setConfig: (alpha, online) => post("/config", { alpha, online }),
    reset: () => post("/reset"),

    samples: () => get("/samples"),
    index: (documents) => post("/index", { documents }),
    documents: () => get("/documents"),
    retrieve: (query, finalK = 5) => post("/retrieve", { query, final_k: finalK }),

    predict: (query, context, response, domain) =>
      post("/predict", { query, context, response, domain }),
    calibrate: (data, domain = "default") => post("/calibrate", { data, domain }),
    calibration: () => get("/calibration"),
    record: (query, context, response, actualHallucination) =>
      post("/record", {
        query, context, response,
        actual_hallucination: actualHallucination,
      }),
    metrics: () => get("/metrics"),
    online: () => get("/online"),

    listDomains: () => get("/domains"),
    registerDomain: (name, alpha) => post("/domains", { name, alpha }),
    calibrateDomain: (name, data) => post(`/domains/${encodeURIComponent(name)}/calibrate`, { data }),
    domainDrift: (name) => get(`/domains/${encodeURIComponent(name)}/drift`),
  };
})();
