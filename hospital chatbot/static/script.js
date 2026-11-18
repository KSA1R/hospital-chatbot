const chatWindow = document.getElementById("chatWindow");
const composerForm = document.getElementById("composerForm");
const messageInput = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");
const newChatBtn = document.getElementById("newChatBtn");
const emergencyBanner = document.getElementById("emergencyBanner");
const statusDot = document.getElementById("statusDot");
const composerFooter = document.querySelector(".composer");

let awaitingReply = false;

// ============================================================
// VOICE LAYER (STT + TTS) - browser-only, no backend changes.
//
// This is purely an additive layer on top of the existing text flow:
// the mic transcribes speech into the SAME messageInput/sendMessage
// path a typed message already goes through, and TTS just reads out
// whatever addMessage() already renders for the assistant. Nothing
// about /api/chat, /api/greeting, or /api/reset changes - voice never
// bypasses the real request path, so anything that works by typing
// works identically by voice.
//
// Uses the Web Speech API (SpeechRecognition for STT, speechSynthesis
// for TTS). Support is inconsistent across browsers (solid in Chrome/
// Edge, absent in Firefox, partial in Safari) - so everything below
// feature-detects first and simply hides itself if unsupported,
// rather than breaking the existing text chat.
// ============================================================

const SpeechRecognitionImpl = window.SpeechRecognition || window.webkitSpeechRecognition;
const speechSupported = {
  stt: !!SpeechRecognitionImpl,
  tts: "speechSynthesis" in window,
};

let voiceOutputEnabled = false;   // only true while a voice session is active
let voiceModeActive = false;      // the actual session flag the mic button toggles
let recognition = null;
let listening = false;            // recognition is capturing audio right now
let suppressAutoSendOnStop = false; // true only when WE call recognition.stop() to end the session
let lastInputWasVoice = false;    // gates auto-relisten so typing doesn't reopen the mic

const micBtn = document.getElementById("micBtn");
const voiceStatus = document.getElementById("voiceStatus");

if (micBtn) micBtn.disabled = !speechSupported.stt;

// Everything mic-related routes status text through here instead of
// console.warn, so "is it even hearing me" has a real, visible answer
// at every stage - starting, live transcript, error, or nothing heard -
// rather than silence being the only feedback (which is what made this
// feel broken in the first place).
let voiceStatusTimeout = null;

function setVoiceStatus(text, type) {
  clearTimeout(voiceStatusTimeout);
  if (!voiceStatus) return;
  if (!text) {
    voiceStatus.classList.add("hidden");
    return;
  }
  voiceStatus.textContent = text;
  voiceStatus.className = "voice-status" + (type ? ` ${type}` : "");
  // Error/info messages auto-clear so they don't linger forever; the
  // "Listening..." state is cleared explicitly by onend/onresult instead.
  if (type === "error" || type === "info") {
    voiceStatusTimeout = setTimeout(() => setVoiceStatus(null), 4000);
  }
}

// --- TTS (speak replies) ---------------------------------------------

function cleanForSpeech(text) {
  // Booking receipts / review tables are formatted for reading on
  // screen (colons, indentation, line breaks) - trim that down so it
  // doesn't come out as an awkward wall of punctuation when spoken.
  return text
    .replace(/\n+/g, ". ")
    .replace(/[#*_>-]+/g, " ")
    .replace(/\s{2,}/g, " ")
    .trim();
}

// After the bot finishes speaking, automatically reopen the mic so a
// voice session is a real back-and-forth conversation, not "press mic
// once per turn". Only does this if: the session is still on, the last
// thing the patient did was actually speak (not type - typing a message
// mid-session shouldn't yank the mic open again), and nothing else
// (emergency, closing, awaiting a reply) has disabled the composer since.
function scheduleRelisten() {
  if (!voiceModeActive || !lastInputWasVoice) return;
  setTimeout(() => {
    if (!voiceModeActive || listening || awaitingReply || micBtn.disabled) return;
    setVoiceStatus("Listening…", "listening");
    try {
      recognition.start();
    } catch (err) {
      // Already starting from somewhere else - ignore, nothing to recover.
    }
  }, 400);
}

function speak(text) {
  if (!speechSupported.tts || !voiceOutputEnabled || !text) {
    // Nothing will actually play, so there's no utterance "onend" to
    // chain off of - relisten right away instead of hanging forever.
    scheduleRelisten();
    return;
  }
  // Never let the bot's own voice get picked back up by the mic.
  if (listening) recognition.stop();
  window.speechSynthesis.cancel();   // don't overlap with a previous reply

  // Chrome has a long-standing bug where calling speak() in the same
  // tick right after cancel() silently drops the utterance - no error,
  // no audio, nothing. Pushing the actual speak() to the next tick
  // reliably avoids it.
  setTimeout(() => {
    const utterance = new SpeechSynthesisUtterance(cleanForSpeech(text));
    utterance.rate = 1.0;
    utterance.onstart = () => setVoiceStatus("🔊 Speaking…", "listening");
    utterance.onend = () => {
      setVoiceStatus(null);
      scheduleRelisten();
    };
    utterance.onerror = (event) => {
      console.warn("[voice] speech synthesis error:", event.error);
      setVoiceStatus("Voice output error.", "error");
      scheduleRelisten();
    };
    window.speechSynthesis.resume();   // some browsers leave the engine "paused" after cancel()
    window.speechSynthesis.speak(utterance);
  }, 50);
}

// --- STT (speak your message) + voice session control -----------------

if (speechSupported.stt) {
  recognition = new SpeechRecognitionImpl();
  recognition.continuous = false;      // one utterance per turn, auto-relisten handles the next one
  recognition.interimResults = true;   // show live transcript while speaking
  recognition.lang = "en-US";

  recognition.onstart = () => {
    listening = true;
    micBtn.classList.add("active");
    window.speechSynthesis.cancel();   // don't transcribe the bot talking over the mic opening
    setVoiceStatus("Listening…", "listening");
  };

  recognition.onresult = (event) => {
    let transcript = "";
    for (let i = 0; i < event.results.length; i++) {
      transcript += event.results[i][0].transcript;
    }
    messageInput.value = transcript;
    autoResizeTextarea();
    // Mirror what's being heard into the status line too - if you're
    // watching the mic button instead of the text box, this is the
    // proof it's actually picking up your voice as you talk.
    setVoiceStatus(transcript ? `Hearing: "${transcript}"` : "Listening…", "listening");
  };

  recognition.onerror = (event) => {
    listening = false;
    micBtn.classList.remove("active");
    const messages = {
      "not-allowed": "Microphone permission denied - check your browser's site settings.",
      "service-not-allowed": "Microphone permission denied - check your browser's site settings.",
      "no-speech": "Didn't hear anything.",
      "audio-capture": "No microphone found.",
      "network": "Voice recognition network error.",
      "aborted": null,   // we stopped it ourselves - not a real error, stay silent
    };
    const fatal = ["not-allowed", "service-not-allowed", "audio-capture"];
    if (fatal.includes(event.error)) {
      // Voice genuinely can't work right now - end the session instead
      // of retrying forever against a mic that isn't there / isn't allowed.
      endVoiceSession();
    }
    const message = event.error in messages ? messages[event.error] : "Voice input error.";
    if (message) setVoiceStatus(message, "error");
  };

  recognition.onend = () => {
    listening = false;
    micBtn.classList.remove("active");

    if (suppressAutoSendOnStop) {
      // We stopped this ourselves (ending the session) - don't send
      // whatever partial transcript happened to be sitting in the box.
      suppressAutoSendOnStop = false;
      return;
    }

    // Auto-send the transcribed message once the patient stops talking,
    // same as submitting a typed message - but only if something was
    // actually heard, so an empty/failed capture doesn't submit a blank
    // message.
    const text = messageInput.value.trim();
    if (text && !awaitingReply) {
      messageInput.value = "";
      autoResizeTextarea();
      setVoiceStatus(null);   // the transcript is about to appear as a real chat bubble, so drop the preview
      lastInputWasVoice = true;
      sendMessage(text);      // speak() will call scheduleRelisten() once the reply is spoken
    } else if (!text) {
      // Nothing captured this round. If the session is still on, keep
      // the conversation hands-free by listening again rather than
      // stopping and making the patient tap the mic a second time.
      if (voiceModeActive) {
        setVoiceStatus("Didn't catch that - listening again…", "info");
        lastInputWasVoice = true;
        scheduleRelisten();
      } else {
        setVoiceStatus("Didn't catch that - tap the mic and try again.", "info");
      }
    }
  };

  function startVoiceSession() {
    voiceModeActive = true;
    voiceOutputEnabled = true;
    lastInputWasVoice = true;
    micBtn.classList.add("session");
    micBtn.setAttribute("aria-pressed", "true");
    micBtn.title = "Stop voice conversation";
    messageInput.value = "";
    setVoiceStatus("Listening…", "listening");
    try {
      recognition.start();
    } catch (err) {
      setVoiceStatus("Voice input error - try again.", "error");
    }
  }

  function endVoiceSession() {
    voiceModeActive = false;
    voiceOutputEnabled = false;
    micBtn.classList.remove("session");
    micBtn.setAttribute("aria-pressed", "false");
    micBtn.title = "Start voice conversation";
    window.speechSynthesis.cancel();   // stop the bot mid-sentence if it's talking
    if (listening) {
      suppressAutoSendOnStop = true;
      recognition.stop();
    }
    messageInput.value = "";
    autoResizeTextarea();
    setVoiceStatus(null);
  }

  // Safari (and some Chrome configs) only let speechSynthesis actually
  // produce audio if the FIRST speak() call on the page happened
  // synchronously inside a user gesture (a click). Every real speak()
  // call we make later happens inside async callbacks - recognition's
  // onend, then an awaited fetch, then a setTimeout - which don't count
  // as a gesture on their own, so the browser can silently swallow them.
  // Priming here, directly in the click handler, with a silent no-op
  // utterance satisfies that requirement once and unlocks normal
  // speak() calls for the rest of the session.
  function primeSpeechSynthesis() {
    if (!speechSupported.tts) return;
    const primer = new SpeechSynthesisUtterance(" ");
    primer.volume = 0;
    window.speechSynthesis.speak(primer);
  }

  micBtn.addEventListener("click", () => {
    if (awaitingReply) return;
    primeSpeechSynthesis();
    if (voiceModeActive) {
      endVoiceSession();
    } else {
      startVoiceSession();
    }
  });
}

function scrollToBottom() {
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function addMessage(role, text) {
  const row = document.createElement("div");
  row.className = `msg-row ${role}`;

  if (role === "assistant") {
    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = "H";
    row.appendChild(avatar);
  }

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";
  bubble.textContent = text;
  row.appendChild(bubble);

  chatWindow.appendChild(row);
  scrollToBottom();

  if (role === "assistant") {
    speak(text);
  }

  return bubble;
}

function addTypingIndicator() {
  const row = document.createElement("div");
  row.className = "msg-row assistant";
  row.id = "typingRow";

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = "H";
  row.appendChild(avatar);

  const dots = document.createElement("div");
  dots.className = "typing-dots";
  dots.innerHTML = "<span></span><span></span><span></span>";
  row.appendChild(dots);

  chatWindow.appendChild(row);
  scrollToBottom();
}

function removeTypingIndicator() {
  const row = document.getElementById("typingRow");
  if (row) row.remove();
}

function setComposerEnabled(enabled) {
  messageInput.disabled = !enabled;
  sendBtn.disabled = !enabled;
  composerFooter.classList.toggle("disabled", !enabled);
  if (speechSupported.stt) {
    micBtn.disabled = !enabled;
    if (!enabled) {
      // Conversation is over (emergency/closing) - end the voice session
      // outright rather than leaving it half-on. Note this does NOT
      // cancel speech synthesis: if the bot's final message (e.g. the
      // emergency notice) is already queued to be spoken, it should
      // still play - only future listening/relistening stops.
      voiceModeActive = false;
      voiceOutputEnabled = false;
      micBtn.classList.remove("session");
      micBtn.setAttribute("aria-pressed", "false");
      if (listening) {
        suppressAutoSendOnStop = true;
        recognition.stop();
      }
      setVoiceStatus(null);
    }
  }
}

function applyStatus(payload) {
  if (payload.is_emergency) {
    emergencyBanner.classList.remove("hidden");
    statusDot.classList.add("emergency");
  }
  if (payload.is_emergency || payload.is_closing) {
    setComposerEnabled(false);
  }
}

function autoResizeTextarea() {
  messageInput.style.height = "auto";
  messageInput.style.height = Math.min(messageInput.scrollHeight, 160) + "px";
}

// Shared by both the very first page load AND "New conversation" -
// hits whichever endpoint is passed in, shows the typing indicator
// while waiting, and renders the opening line + status the same way
// either time, so the two flows can't drift out of sync with each other.
async function fetchOpeningLine(url, options) {
  addTypingIndicator();
  try {
    const res = await fetch(url, options);
    const data = await res.json();
    removeTypingIndicator();
    addMessage("assistant", data.reply);
    applyStatus(data);
  } catch (err) {
    removeTypingIndicator();
    addMessage("assistant", "Sorry, I couldn't connect to the server. Is app.py running?");
  }
}

function loadGreeting() {
  // First load of the page: no prior session exists yet, so just ask
  // for the greeting on whatever session app.py already has ready.
  return fetchOpeningLine("/api/greeting", { method: "GET" });
}

function startNewConversation() {
  // "New conversation": must hit /api/reset, NOT /api/greeting.
  // /api/greeting only calls handle_input("") on the EXISTING session
  // object - it never creates a new ConversationSession(). That's what
  // caused the old bug where clicking "New conversation" after a chat
  // had ended showed the closing message again ("Take care...") instead
  // of a real greeting - the backend session was still sitting in the
  // CLOSING state from the previous conversation, and this button
  // wasn't actually resetting it. /api/reset creates a brand-new
  // session first, so this now genuinely starts over.
  return fetchOpeningLine("/api/reset", { method: "POST" });
}

async function sendMessage(text) {
  addMessage("user", text);
  addTypingIndicator();
  awaitingReply = true;
  setComposerEnabled(false);

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    const data = await res.json();
    removeTypingIndicator();

    if (data.error) {
      addMessage("assistant", "Something went wrong on that last message - please try again.");
    } else {
      addMessage("assistant", data.reply);
      applyStatus(data);
    }
  } catch (err) {
    removeTypingIndicator();
    addMessage("assistant", "Sorry, I couldn't reach the server. Please check app.py is still running.");
  } finally {
    awaitingReply = false;
    // Only re-enable if the conversation isn't over/emergency (applyStatus already disables those cases)
    if (!emergencyBanner || emergencyBanner.classList.contains("hidden")) {
      setComposerEnabled(true);
      messageInput.focus();
    }
  }
}

composerForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = messageInput.value.trim();
  if (!text || awaitingReply) return;
  messageInput.value = "";
  autoResizeTextarea();
  // Typed manually - even mid voice-session, don't auto-reopen the mic
  // after this reply (TTS still plays if the session is on, it just
  // won't yank the mic back open on its own).
  lastInputWasVoice = false;
  sendMessage(text);
});

messageInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    composerForm.requestSubmit();
  }
});

messageInput.addEventListener("input", autoResizeTextarea);

newChatBtn.addEventListener("click", async () => {
  chatWindow.innerHTML = "";
  emergencyBanner.classList.add("hidden");
  statusDot.classList.remove("emergency");
  setComposerEnabled(true);
  await startNewConversation();
});

loadGreeting();
