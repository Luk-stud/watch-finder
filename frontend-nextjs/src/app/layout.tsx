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
  manifest: "/manifest.json",
  appleWebApp: {
    capable: true,
    statusBarStyle: "black-translucent",
    title: "Watch Finder",
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
  viewportFit: "cover",
  themeColor: "#558a86",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="overflow-hidden">
        <AppProvider>
          {children}
        </AppProvider>
      </body>
    </html>
  );
}
