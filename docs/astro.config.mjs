// @ts-check
import {defineConfig} from "astro/config";
import starlight from "@astrojs/starlight";

// https://astro.build/config
export default defineConfig({
  integrations: [
    starlight({
      title: "sensipy",
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
