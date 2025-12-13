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
            {label: "IRFs", slug: "getting_started/irfs"},
            {label: "Spectral Models", slug: "getting_started/spectral_models"},
            {label: "EBL Models", slug: "getting_started/ebl_models"},
            {
              label: "Sensitivity Calculations",
              slug: "getting_started/sensitivity",
            },
            {label: "Exposure Calculations", slug: "getting_started/exposure"},
            {label: "Followup Calculations", slug: "getting_started/followups"},
          ],
        },
        {
          label: "Tutorials",
          items: [
            {label: "Overview", slug: "tutorials"},
            {label: "Basic Workflow", slug: "tutorials/basic_workflow"},
            {label: "Loading IRFs", slug: "tutorials/loading_irfs"},
            {
              label: "Calculating Sensitivity",
              slug: "tutorials/calculating_sensitivity",
            },
            {
              label: "Simulating Observations",
              slug: "tutorials/simulating_observations",
            },
            {label: "Followup Analysis", slug: "tutorials/followup_analysis"},
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
