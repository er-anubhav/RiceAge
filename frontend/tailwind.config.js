/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        lab: {
          bg: "#f4f6f8",
          card: "#ffffff",
          border: "#e2e8f0",
          header: "#1e3a8a", // Dark lab blue for headers
          navy: {
            DEFAULT: "#0f172a",
            light: "#1e293b",
            dark: "#0b1120",
            hover: "#1e293b",
            active: "#2563eb"
          },
          teal: {
            DEFAULT: "#0d9488",
            light: "#f0fdfa",
            border: "#99f6e4",
            text: "#0f766e"
          },
          coral: {
            DEFAULT: "#e11d48",
            light: "#fff1f2",
            border: "#fecdd3",
            text: "#be123c"
          },
          blue: {
            DEFAULT: "#2563eb",
            light: "#eff6ff",
            border: "#bfdbfe",
            text: "#1d4ed8"
          },
          orange: {
            DEFAULT: "#ea580c",
            light: "#fff7ed",
            border: "#fed7aa",
            text: "#c2410c"
          },
          pass: {
            DEFAULT: "#16a34a",
            bg: "#dcfce7",
            border: "#bbf7d0",
            text: "#15803d"
          },
          warning: {
            DEFAULT: "#d97706",
            bg: "#fef3c7",
            border: "#fde68a",
            text: "#b45309"
          },
          substandard: {
            DEFAULT: "#dc2626",
            bg: "#fee2e2",
            border: "#fca5a5",
            text: "#b91c1c"
          }
        }
      },
      fontFamily: {
        mono: [
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Monaco",
          "Consolas",
          "Liberation Mono",
          "Courier New",
          "monospace"
        ]
      }
    },
  },
  plugins: [],
}
