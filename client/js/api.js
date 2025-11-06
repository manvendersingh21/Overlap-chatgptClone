// Minimal API module: streaming POST to backend conversation endpoint.
// Exports streamConversation(payload, onChunk, signal) -> returns final accumulated text.

export async function streamConversation(payload, onChunk, signal) {
  const url = '/backend-api/v2/conversation';

  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'text/event-stream'
    },
    body: JSON.stringify(payload),
    signal
  });

  if (!res.ok) {
    // attempt to read response body for better error messages
    const body = await res.text().catch(() => '');
    throw new Error(`Request failed: ${res.status} ${res.statusText}${body ? ' - ' + body : ''}`);
  }

  if (!res.body) {
    throw new Error('Response has no body stream');
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let finalText = '';

  // We'll parse Server-Sent Events (SSE) framed as one or more 'data: ...' lines
  // separated by a blank line (\n\n). The server emits JSON payloads in
  // each data: event in the form {"text": "..."}.
  let buffer = '';

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });

      // Basic protection: detect common HTML/CF challenge responses and convert to readable text
      if (chunk.includes('<form id="challenge-form"') || chunk.includes('<title>Attention Required</title>')) {
        const msg = 'Error: Cloudflare/edge returned an HTML challenge. Refresh the page or check the server.';
        finalText += msg;
        try { if (typeof onChunk === 'function') onChunk(msg); } catch (e) { /* ignore */ }
        continue;
      }

      buffer += chunk;

      // Process complete SSE events (separated by \n\n)
      while (true) {
        const idx = buffer.indexOf('\n\n');
        if (idx === -1) break;
        const rawEvent = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);

        // Extract data: lines (may be multiple) and concatenate their payloads
        const lines = rawEvent.split(/\r?\n/);
        let dataPayload = '';
        for (const line of lines) {
          if (line.startsWith('data:')) {
            dataPayload += line.slice(5).trim();
          }
        }

        if (!dataPayload) continue;

        // Try parsing JSON payloads emitted by the server: {"text":"..."}
        let text = dataPayload;
        try {
          const parsed = JSON.parse(dataPayload);
          if (parsed && typeof parsed.text === 'string') text = parsed.text;
        } catch (e) {
          // not JSON — keep raw payload
        }

        finalText += text;
        try { if (typeof onChunk === 'function') onChunk(text); } catch (e) { /* ignore */ }
      }
    }
  } catch (err) {
    // Propagate AbortError to allow callers to detect cancellation
    throw err;
  } finally {
    try { reader.releaseLock(); } catch (e) { /* ignore */ }
  }

  // if any leftover buffer contains text (no trailing \n\n), try to process it
  if (buffer) {
    let text = buffer;
    try {
      const parsed = JSON.parse(buffer);
      if (parsed && typeof parsed.text === 'string') text = parsed.text;
    } catch (e) { /* ignore */ }
    finalText += text;
    try { if (typeof onChunk === 'function') onChunk(text); } catch (e) { /* ignore */ }
  }

  return finalText;
}