/* Thin wrapper over the REST API. Every failure surfaces as an Error with a
   message that is safe to show to a person. */

const API = (() => {
  async function request(path, options) {
    let res;
    try {
      res = await fetch(path, options);
    } catch (err) {
      throw new Error("Cannot reach the server. Check that the demo is still running.");
    }

    if (!res.ok) {
      let detail = `Request failed with status ${res.status}.`;
      try {
        const body = await res.json();
        if (body && body.detail) {
          detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
        }
      } catch (err) {
        /* keep the status message */
      }
      throw new Error(detail);
    }

    return res.json();
  }

  const get = (path) => request(path);
  const post = (path, body) =>
    request(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  /* Multipart upload: the browser sets the Content-Type boundary itself. */
  const upload = (path, file) => {
    const body = new FormData();
    body.append("file", file);
    return request(path, { method: "POST", body });
  };

  return { get, post, upload };
})();
