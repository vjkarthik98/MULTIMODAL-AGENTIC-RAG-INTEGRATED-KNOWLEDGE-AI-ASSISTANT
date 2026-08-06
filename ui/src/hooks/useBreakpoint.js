import { useCallback, useSyncExternalStore } from 'react'

/**
 * Media-query hooks — the single source of truth for responsive behaviour in JS.
 *
 * Replaces the previous `useIsMobile` (a lone `max-width: 560px` check). That
 * breakpoint is why tablets rendered broken: ChatPage treated anything ≥560px as
 * "desktop" and laid the sidebar out inline at 320px, so a 768px iPad had ~450px
 * left for the conversation.
 *
 * Two rules this module exists to enforce:
 *
 * 1. BREAKPOINTS MATCH TAILWIND. The numbers below are Tailwind's own (sm 640,
 *    lg 1024), so a JS decision and a `sm:`/`lg:` class can never disagree about
 *    where a layout changes. Previously the repo had three competing systems —
 *    560px here, 1024px in App.css (dead Vite scaffold, now deleted), and
 *    Tailwind's defaults in the handful of prefixed classes.
 *
 * 2. ASK THE QUESTION YOU ACTUALLY MEAN. Width is the wrong proxy for most
 *    responsive decisions. `useHasHover` is about input capability, not size;
 *    a component sized against a fixed-width container should query that width,
 *    not the viewport's. Overloading one `isMobile` flag for all three was what
 *    made the old behaviour wrong in more than one place at once.
 */

/**
 * Subscribe to any media query.
 *
 * Built on `useSyncExternalStore` rather than useState+useEffect. matchMedia is
 * precisely what that API is for — an external store React doesn't own — and it
 * avoids the failure mode the useState version has: the initial state is read
 * during render, so a viewport change between render and effect leaves a stale
 * value on screen until something else re-renders. Papering over that with a
 * `setMatches` inside the effect works, but causes a second render pass on
 * every mount (and trips react-hooks/set-state-in-effect). useSyncExternalStore
 * reads the current value at render time, every time, with no extra pass.
 *
 * The third argument is the server snapshot: `false` during SSR/prerender,
 * since there is no viewport to measure.
 */
export function useMediaQuery(query) {
  const subscribe = useCallback(
    onChange => {
      const mq = window.matchMedia(query)
      mq.addEventListener('change', onChange)
      return () => mq.removeEventListener('change', onChange)
    },
    [query]
  )

  return useSyncExternalStore(
    subscribe,
    () => window.matchMedia(query).matches,
    () => false
  )
}

/**
 * True below Tailwind's `lg` (1024px) — i.e. phones AND tablets.
 *
 * This is the "is there room for a permanent inline sidebar?" question. Below
 * 1024px the answer is no, and the sidebar becomes an overlay drawer so the
 * conversation gets the full width. iPad portrait (768px) is deliberately on the
 * drawer side of this line; iPad landscape (1024px) gets the inline sidebar.
 */
export function useIsCompact() {
  return useMediaQuery('(max-width: 1023px)')
}

/** True below Tailwind's `sm` (640px) — phone-sized only. */
export function useIsPhone() {
  return useMediaQuery('(max-width: 639px)')
}

/**
 * True when the device has a real hovering pointer (mouse/trackpad).
 *
 * For anything currently revealed on `:hover` — message action rows, etc. On a
 * touch device those controls can never be revealed, so they must be shown
 * permanently. Width is a bad stand-in for this: a 900px tablet and a 1400px
 * touchscreen laptop both fail to hover while reading as "desktop" by width,
 * which is exactly how the old 560px check left actions unreachable on tablets.
 */
export function useHasHover() {
  return useMediaQuery('(hover: hover) and (pointer: fine)')
}
