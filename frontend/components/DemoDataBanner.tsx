'use client'

import { Info } from 'lucide-react'

/**
 * Labels dashboard surfaces that render sample data.
 *
 * These pages demonstrate the monitoring and registry interfaces. The model
 * versions, experiment results, and metrics they display are illustrative
 * values, not measurements from a trained model or a live deployment.
 */
export default function DemoDataBanner() {
  return (
    <div
      role="note"
      className="mb-6 flex items-start gap-3 rounded-lg border border-amber-500/40 bg-amber-500/10 px-4 py-3"
    >
      <Info className="mt-0.5 h-5 w-5 flex-shrink-0 text-amber-500" aria-hidden="true" />
      <p className="text-sm leading-relaxed text-amber-200">
        <span className="font-semibold">Demo data.</span> This page illustrates the interface.
        Model versions, experiment results, and metrics shown here are sample values, not
        measurements from a trained model or a live deployment.
      </p>
    </div>
  )
}
