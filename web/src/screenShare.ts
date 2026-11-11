import { useSyncExternalStore } from 'react'
import { api, post } from './api'

/**
 * Captures one browser tab/window the user picks and streams its frames to ``POST /api/v1/camera/frame``.
 * Frames are read with MediaStreamTrackProcessor (promise-driven, not timer-driven) so a hidden app tab is not throttled.
 */

// Not in lib.dom yet (Chrome/Edge only).
declare class MediaStreamTrackProcessor {
  constructor(init: { track: MediaStreamTrack; maxBufferSize?: number })
  readable: ReadableStream<VideoFrame>
}

export type SharePhase = 'idle' | 'ready' | 'starting' | 'sending' | 'ended' | 'error'
export type ShareState = {
  phase: SharePhase; alertTracks: string[]; framesSent: number; sendFps: number | null; surfaceLabel: string | null; error: string | null
}
type FrameReply = { alert_track_ids?: string[]; frames_processed?: number }

const MAX_SIDE = 1280
const JPEG_QUALITY = 0.8
const RETRY_MS = 200
const STARTING_GIVE_UP_MS = 60_000
const MAX_FAILURES = 5
const UNSUPPORTED = 'Trình duyệt này chưa hỗ trợ quét tab. Hãy dùng Chrome hoặc Edge.'
const IDLE: ShareState = { phase: 'idle', alertTracks: [], framesSent: 0, sendFps: null, surfaceLabel: null, error: null }

let state: ShareState = IDLE
let stream: MediaStream | null = null
let runId = 0
const listeners = new Set<() => void>()

function set(patch: Partial<ShareState>) {
  state = { ...state, ...patch }
  listeners.forEach((listener) => listener())
}
const releaseStream = () => { stream?.getTracks().forEach((track) => track.stop()); stream = null }

export const getShareState = () => state
export function subscribeShare(listener: () => void) {
  listeners.add(listener)
  return () => { listeners.delete(listener) }
}
export const useScreenShare = () => useSyncExternalStore(subscribeShare, getShareState)

export const screenShareSupported = () => typeof navigator !== 'undefined' && !!navigator.mediaDevices?.getDisplayMedia
  && 'MediaStreamTrackProcessor' in globalThis && typeof OffscreenCanvas !== 'undefined'

/** Opens the browser's tab/window picker; the app's own tab is excluded so the preview cannot capture itself. */
export async function pickSurface(): Promise<void> {
  if (!screenShareSupported()) { set({ phase: 'error', error: UNSUPPORTED }); return }
  runId++
  releaseStream()
  try {
    const media = await navigator.mediaDevices.getDisplayMedia({
      video: { frameRate: 15 }, audio: false, selfBrowserSurface: 'exclude',
    } as DisplayMediaStreamOptions)
    const track = media.getVideoTracks()[0]
    stream = media
    track.addEventListener('ended', () => onTrackEnded(media))
    set({ ...IDLE, phase: 'ready', surfaceLabel: track.label || null })
  } catch (reason) {
    const name = (reason as { name?: string } | null)?.name
    set({ ...IDLE, error: name === 'NotAllowedError' ? 'Bạn chưa chọn tab nào để quét.' : `Không chọn được tab để quét (${name ?? 'lỗi không rõ'}).` })
  }
}

/** Start sending frames; call it before the camera ``start`` request resolves (the server accepts frames while the model loads). */
export function beginSending(fps: number): void {
  const track = stream?.getVideoTracks()[0]
  if (!track) return
  const id = ++runId
  set({ phase: 'starting', alertTracks: [], framesSent: 0, sendFps: null, error: null })
  void sendLoop(id, track, fps)
}

/** Stop capturing locally (does not call the API). */
export function stopShare(): void {
  runId++
  releaseStream()
  set({ ...IDLE })
}

function fail(id: number, message: string) {
  if (id !== runId) return
  runId++
  releaseStream()
  set({ phase: 'error', alertTracks: [], sendFps: null, error: message })
}

function onTrackEnded(media: MediaStream) {
  if (stream !== media) return
  const active = state.phase === 'starting' || state.phase === 'sending'
  runId++
  stream = null
  if (active) {
    post('/api/v1/camera/stop').catch(() => undefined)
    set({ phase: 'ended', alertTracks: [], sendFps: null, error: 'Đã dừng chia sẻ tab; phiên camera đã được dừng.' })
  } else {
    set({ ...IDLE, error: 'Đã dừng chia sẻ tab.' })
  }
}

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms))

async function sendLoop(id: number, track: MediaStreamTrack, fps: number) {
  const reader = new MediaStreamTrackProcessor({ track, maxBufferSize: 1 }).readable.getReader()
  const minGapUs = 1_000_000 / fps
  // Chrome sends no frames while the captured tab is static (paused, ended, unchanged page); after this long
  // without a new frame the last one is sent again so the server still sees a live source.
  const idleMs = Math.max(250, 2000 / fps)
  const began = performance.now()
  const sentAt: number[] = []
  let lastUs = -Infinity, canvas: OffscreenCanvas | null = null, context: OffscreenCanvasRenderingContext2D | null = null
  let lastBlob: Blob | null = null, pending: Promise<ReadableStreamReadResult<VideoFrame>> | null = null
  let failures = 0, sent = 0, dead = false

  const encode = async (frame: VideoFrame): Promise<Blob | null> => {
    if (frame.timestamp - lastUs < minGapUs) return null
    lastUs = frame.timestamp
    const width = frame.displayWidth, height = frame.displayHeight
    if (!canvas) {
      const scale = Math.min(1, MAX_SIDE / Math.max(width, height))
      canvas = new OffscreenCanvas(Math.max(2, Math.round(width * scale)), Math.max(2, Math.round(height * scale)))
      context = canvas.getContext('2d')
    }
    if (!context) { dead = true; fail(id, 'Không tạo được vùng vẽ để nén khung hình.'); return null }
    // Fixed canvas size: a resized source window is letterboxed, never sent at a different size.
    const fit = Math.min(canvas.width / width, canvas.height / height)
    const drawW = width * fit, drawH = height * fit
    context.fillStyle = '#000'
    context.fillRect(0, 0, canvas.width, canvas.height)
    context.drawImage(frame, (canvas.width - drawW) / 2, (canvas.height - drawH) / 2, drawW, drawH)
    return canvas.convertToBlob({ type: 'image/jpeg', quality: JPEG_QUALITY })
  }

  /** Uploads one frame; returns false when the loop must end. */
  const send = async (blob: Blob): Promise<boolean> => {
    let response: Response | null = null
    try {
      response = await fetch(api('/api/v1/camera/frame'), { method: 'POST', body: blob, headers: { 'Content-Type': 'image/jpeg' }, cache: 'no-store' })
    } catch { response = null }
    if (id !== runId) return false
    if (response?.ok) {
      failures = 0
      const reply = (await response.json().catch(() => ({}))) as FrameReply
      const now = performance.now()
      sentAt.push(now)
      while (sentAt.length > 1 && now - sentAt[0] > 2000) sentAt.shift()
      sent++
      const span = (now - sentAt[0]) / 1000
      set({ phase: 'sending', framesSent: sent, alertTracks: reply.alert_track_ids ?? [],
        sendFps: sentAt.length > 1 && span > 0 ? Math.round(((sentAt.length - 1) / span) * 10) / 10 : state.sendFps })
    } else if (response?.status === 409 && state.phase === 'starting') {
      if (performance.now() - began > STARTING_GIVE_UP_MS) { fail(id, 'Server không nhận khung hình (phiên camera chưa bắt đầu).'); return false }
      await sleep(RETRY_MS)
    } else if (response?.status === 409) {
      fail(id, 'Phiên camera trên server đã dừng.'); return false
    } else if (++failures >= MAX_FAILURES) {
      fail(id, 'Không gửi được khung hình lên server nhiều lần liên tiếp.'); return false
    }
    return true
  }

  try {
    while (id === runId && !dead) {
      pending ??= reader.read()
      const next = await Promise.race([pending, sleep(idleMs).then(() => null)])
      let blob: Blob | null
      if (next === null) {
        blob = lastBlob
      } else {
        pending = null
        if (next.done || !next.value) break
        try { blob = await encode(next.value) } finally { next.value.close() }
        if (blob) lastBlob = blob
      }
      if (!blob || id !== runId) continue
      if (!(await send(blob))) break
    }
  } finally {
    reader.cancel().catch(() => undefined)
    pending?.then((result) => result.value?.close(), () => undefined)
  }
}
