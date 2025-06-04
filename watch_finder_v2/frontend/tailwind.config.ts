import type { Config } from "tailwindcss";
import tailwindcssAnimate from "tailwindcss-animate";

export default {
	darkMode: ["class"],
	content: [
		"./pages/**/*.{ts,tsx}",
		"./components/**/*.{ts,tsx}",
		"./app/**/*.{ts,tsx}",
		"./src/**/*.{ts,tsx}",
	],
	prefix: "",
	theme: {
		container: {
			center: true,
			padding: '2rem',
			screens: {
				'2xl': '1400px'
			}
		},
		extend: {
			colors: {
				border: 'hsl(var(--border))',
				input: 'hsl(var(--input))',
				ring: 'hsl(var(--ring))',
				background: 'hsl(var(--background))',
				foreground: 'hsl(var(--foreground))',
				primary: {
					DEFAULT: 'hsl(var(--primary))',
					foreground: 'hsl(var(--primary-foreground))'
				},
				secondary: {
					DEFAULT: 'hsl(var(--secondary))',
					foreground: 'hsl(var(--secondary-foreground))'
				},
				destructive: {
					DEFAULT: 'hsl(var(--destructive))',
					foreground: 'hsl(var(--destructive-foreground))'
				},
				muted: {
					DEFAULT: 'hsl(var(--muted))',
					foreground: 'hsl(var(--muted-foreground))'
				},
				accent: {
					DEFAULT: 'hsl(var(--accent))',
					foreground: 'hsl(var(--accent-foreground))'
				},
				popover: {
					DEFAULT: 'hsl(var(--popover))',
					foreground: 'hsl(var(--popover-foreground))'
				},
				card: {
					DEFAULT: 'hsl(var(--card))',
					foreground: 'hsl(var(--card-foreground))'
				},
				sidebar: {
					DEFAULT: 'hsl(var(--sidebar-background))',
					foreground: 'hsl(var(--sidebar-foreground))',
					primary: 'hsl(var(--sidebar-primary))',
					'primary-foreground': 'hsl(var(--sidebar-primary-foreground))',
					accent: 'hsl(var(--sidebar-accent))',
					'accent-foreground': 'hsl(var(--sidebar-accent-foreground))',
					border: 'hsl(var(--sidebar-border))',
					ring: 'hsl(var(--sidebar-ring))'
				}
			},
			borderRadius: {
				lg: 'var(--radius)',
				md: 'calc(var(--radius) - 2px)',
				sm: 'calc(var(--radius) - 4px)'
			},
			keyframes: {
				'accordion-down': {
					from: {
						height: '0'
					},
					to: {
						height: 'var(--radix-accordion-content-height)'
					}
				},
				'accordion-up': {
					from: {
						height: 'var(--radix-accordion-content-height)'
					},
					to: {
						height: '0'
					}
				},
				'scale-in': {
					'0%': {
						transform: 'scale(0.95)',
						opacity: '0'
					},
					'100%': {
						transform: 'scale(1)',
						opacity: '1'
					}
				}
			},
			animation: {
				'accordion-down': 'accordion-down 0.2s ease-out',
				'accordion-up': 'accordion-up 0.2s ease-out',
				'scale-in': 'scale-in 0.2s ease-out'
			},
			height: {
				'screen-dynamic': 'calc(var(--viewport-height, 100vh) - var(--safe-area-inset-top, 0px) - var(--safe-area-inset-bottom, 0px))',
				'dvh': '100dvh',
				'card-max': 'min(600px, calc(100vh - 200px))',
				'card-small': 'calc(100vh - 120px)',
				'card-tiny': 'calc(100vh - 100px)',
			},
			minHeight: {
				'screen-dynamic': 'calc(var(--viewport-height, 100vh) - var(--safe-area-inset-top, 0px) - var(--safe-area-inset-bottom, 0px))',
				'dvh': '100dvh',
				'card-responsive': 'min(400px, calc(100vh - 200px))',
				'card-small': 'calc(100vh - 120px)',
				'card-tiny': 'calc(100vh - 100px)',
			},
			maxHeight: {
				'screen-dynamic': 'calc(var(--viewport-height, 100vh) - var(--safe-area-inset-top, 0px) - var(--safe-area-inset-bottom, 0px))',
				'dvh': '100dvh',
				'card-responsive': 'min(600px, calc(100vh - 200px))',
				'card-small': 'calc(100vh - 120px)',
				'card-tiny': 'calc(100vh - 100px)',
			},
			spacing: {
				'safe-top': 'var(--safe-area-inset-top, 0px)',
				'safe-bottom': 'var(--safe-area-inset-bottom, 0px)',
				'safe-left': 'var(--safe-area-inset-left, 0px)',
				'safe-right': 'var(--safe-area-inset-right, 0px)',
			},
		}
	},
	plugins: [tailwindcssAnimate],
} satisfies Config;
