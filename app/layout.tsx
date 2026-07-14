import type { Metadata } from "next";
import { Fraunces, Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

const fraunces = Fraunces({
  variable: "--font-fraunces",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "When Nothing Happens",
  description:
    "A layperson-friendly exhibit for a one-shot 5.6-Sol research artifact extending prediction-market volatility models.",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
  openGraph: {
    title: "When Nothing Happens",
    description:
      "Prediction-market stillness as a forecast target, with verification caveats visible.",
    images: ["/hazard-reliability.png"],
  },
  twitter: {
    card: "summary_large_image",
    title: "When Nothing Happens",
    description:
      "A plain-English web exhibit for a one-shot prediction-market volatility extension.",
    images: ["/hazard-reliability.png"],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.variable} ${fraunces.variable}`}>{children}</body>
    </html>
  );
}
