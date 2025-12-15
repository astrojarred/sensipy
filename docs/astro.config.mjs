// @ts-check
import {defineConfig} from "astro/config";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import starlight from "@astrojs/starlight";

// https://astro.build/config
export default defineConfig({
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },
  integrations: [
    starlight({
      title: "sensipy",
      head: [
        {
          tag: "link",
          attrs: {
            rel: "stylesheet",
            href: "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css",
            integrity: "sha384-nB0miv6/jRmo5UMMR1wu3Gz6NLsoTkbqJghGIsx//Rlm+ZU03BU6SQNC66uf4l5+",
            crossorigin: "anonymous",
          },
        },
      ],
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/astrojarred/sensipy",
        },
      ],
      sidebar: [
        {
          label: "Getting Started",
          items: [
            {label: "Overview", slug: "getting_started/overview"},
            {label: "Installation", slug: "getting_started/installation"},
            {label: "Setup", slug: "getting_started/setup"},
          ],
        },
        {
          label: "Working with sensipy",
          items: [
            {label: "Overview", slug: "working_with_sensipy/overview"},
            {label: "Working with IRFs", slug: "working_with_sensipy/irfs"},
            {label: "Spectral Models", slug: "working_with_sensipy/spectral_models"},
            {label: "EBL Models", slug: "working_with_sensipy/ebl_models"},
            {label: "Sensitivity Calculations", slug: "working_with_sensipy/sensitivity"},
            {label: "Simulating Observations", slug: "working_with_sensipy/exposure"},
            {label: "Followup Analysis", slug: "working_with_sensipy/followups"},
          ],
        },
        {
          label: "API Reference",
          items: [
            {label: "Overview", slug: "reference"},
            {label: "source", slug: "reference/source"},
            {label: "sensitivity", slug: "reference/sensitivity"},
            {label: "followup", slug: "reference/followup"},
            {label: "ctaoirf", slug: "reference/ctaoirf"},
          ],
        },
      ],
    }),
  ],
});
