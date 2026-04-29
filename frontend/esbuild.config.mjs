import esbuild from "esbuild";
import { copyFile, mkdir, cp } from "node:fs/promises";

const watch = process.argv.includes("--watch");

const entries = {
  "background/service-worker": "src/background/service-worker.ts",
  "content/index": "src/content/index.tsx",
  "popup/index": "src/popup/index.tsx",
  "injected/inject": "src/injected/inject.ts",
};

async function copyStatic() {
  await mkdir("dist/popup", { recursive: true });
  await mkdir("dist/icons", { recursive: true });
  await copyFile("public/manifest.json", "dist/manifest.json");
  await copyFile("src/popup/index.html", "dist/popup/index.html");
  await cp("public/icons", "dist/icons", { recursive: true }).catch(() => {});
}

const ctx = await esbuild.context({
  entryPoints: entries,
  bundle: true,
  outdir: "dist",
  format: "esm",
  target: "es2020",
  jsx: "automatic",
  loader: { ".css": "css" },
  sourcemap: watch ? "inline" : false,
  minify: !watch,
  logLevel: "info",
});

await copyStatic();

if (watch) {
  await ctx.watch();
  console.log("esbuild watching for changes...");
} else {
  await ctx.rebuild();
  await ctx.dispose();
  console.log("Build complete: dist/");
}
