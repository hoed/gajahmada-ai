/**
 * GajahMada — Vercel Web Analytics Integration (@vercel/analytics)
 * Supports both React (<Analytics /> from "@vercel/analytics/react")
 * and Vanilla HTML/FastAPI Serverless execution on Vercel.
 */

export function Analytics() {
  if (typeof window !== 'undefined') {
    initVercelAnalytics();
  }
  return null;
}

export function initVercelAnalytics() {
  if (typeof window === 'undefined') return;

  // Initialize Vercel Analytics queue (same mechanism as @vercel/analytics/react)
  window.va =
    window.va ||
    function () {
      (window.vaq = window.vaq || []).push(arguments);
    };

  if (!document.querySelector('script[src="/_vercel/insights/script.js"]')) {
    const script = document.createElement('script');
    script.src = '/_vercel/insights/script.js';
    script.defer = true;
    script.dataset.sdkn = '@vercel/analytics/react';
    script.dataset.sdkv = '1.5.0';
    document.head.appendChild(script);
  }
}

export function trackVercelEvent(eventName, properties = {}) {
  if (typeof window !== 'undefined' && typeof window.va === 'function') {
    window.va('event', { name: eventName, data: properties });
  }
}

if (typeof window !== 'undefined') {
  initVercelAnalytics();
}
