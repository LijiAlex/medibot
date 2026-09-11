import type { Metadata } from "next";
import { Archivo, Source_Serif_4 } from "next/font/google";
import "./tokens.css";

/* Two families with different jobs. Archivo carries the interface: flat, slightly
 * condensed forms in the vein of ward signage, with tabular numerals for doses. Source
 * Serif sets the answer body only, because every answer is an extract from a printed
 * clinical document and should read like the page it came from. */
const archivo = Archivo({ subsets: ["latin"], weight: ["400", "500", "600"], variable: "--font-sans" });
const serif = Source_Serif_4({ subsets: ["latin"], weight: ["400", "600"], variable: "--font-serif" });

export const metadata: Metadata = {
  title: "MediBot",
  description: "Ask about MediAssist's documents and operations, within what your role may read.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${archivo.variable} ${serif.variable}`}>
      <body>{children}</body>
    </html>
  );
}
