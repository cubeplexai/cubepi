// Cloudflare Pages middleware.
//
// The site moved from the default *.pages.dev hostname to the custom domain
// cubepi.ai. This 301-redirects the production pages.dev hostname to the
// canonical domain so the migration sends search engines one clean signal
// (every old URL maps 1:1 to its new path) instead of leaving the old domain
// serving duplicate content.
//
// Notes:
//   - Matches the production hostname exactly, so preview deployments
//     (e.g. <hash>.cubepi.pages.dev) keep working for review.
//   - Runs on every request; cubepi.ai and previews fall through to
//     context.next(), so there is no redirect loop.
//   - Lives in static/ so Docusaurus copies it to build/functions/, which is
//     the directory uploaded to Cloudflare Pages (see .github/workflows/docs.yml).
export async function onRequest(context) {
  const url = new URL(context.request.url);
  if (url.hostname === "cubepi.pages.dev") {
    url.hostname = "cubepi.ai";
    url.protocol = "https:";
    url.port = "";
    return Response.redirect(url.toString(), 301);
  }
  return context.next();
}
