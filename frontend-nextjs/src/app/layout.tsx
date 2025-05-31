import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { AppProvider } from "@/contexts/AppContext";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Watch Finder - AI-Powered Discovery",
  description: "Discover your perfect timepiece with AI-powered visual similarity matching from 4,994+ watches",
  keywords: "watches, AI, discovery, timepieces, luxury watches, watch finder",
  authors: [{ name: "Watch Finder Team" }],
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <AppProvider>
          {children}
        </AppProvider>
      </body>
    </html>
  );
}
