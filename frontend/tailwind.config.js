/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#0B0D13',
        surface: '#12151F',
        'surface-elevated': '#1A1E2C',
        'surface-hover': '#24293C',
        accent: '#D4AF37', // Refined champagne gold
        'accent-hover': '#E5C048',
        brand: '#E50914', // Classic cinema crimson
        'brand-dark': '#B81D24',
        muted: '#8E95A5',
        'muted-dark': '#555D6E',
        border: 'rgba(255, 255, 255, 0.08)',
        'border-focus': 'rgba(255, 255, 255, 0.2)',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        display: ['Outfit', 'Inter', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
