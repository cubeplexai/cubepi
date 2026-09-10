// Cloudflare Pages "advanced mode" Worker.
//
// The site moved from cubepi.pages.dev → cubepi.ai → cubeloop.dev. The Pages
// project is still named `cubepi`, so the production pages.dev hostname remains
// cubepi.pages.dev (preview URLs are <hash>.cubepi.pages.dev). This 301s that
// hostname — and cubeloop.pages.dev if the project is later renamed — to the
// matching cubeloop.dev path. Search Console Change of Address requires a
// single hop; chaining through cubepi.ai fails Google's 301 check.
//
// Why _worker.js (advanced mode) and not functions/_middleware.js:
//   The site is deployed via Direct Upload (cloudflare/pages-action uploads
//   website/build). Cloudflare compiles a _worker.js that sits IN the uploaded
//   output directory; a functions/ directory, by contrast, must live at the
//   project root where Wrangler runs — placing it under static/ (so it lands in
//   build/functions/) makes Cloudflare serve it as a static file, not a Function.
//   Docusaurus copies static/_worker.js to build/_worker.js, the deployed dir.
//
// Notes:
//   - Matches the production hostname exactly, so preview deployments
//     (e.g. <hash>.cubepi.pages.dev) fall through and stay browsable.
//   - Everything else is forwarded to env.ASSETS.fetch, which serves the static
//     site exactly as Pages would (trailing-slash handling, 404.html, etc.).
export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (
      url.hostname === "cubepi.pages.dev" ||
      url.hostname === "cubeloop.pages.dev"
    ) {
      url.hostname = "cubeloop.dev";
      url.protocol = "https:";
      url.port = "";
      return Response.redirect(url.toString(), 301);
    }
    return env.ASSETS.fetch(request);
  },
};
