"""
main.py

Text-only CLI entry point. Run this to talk to the bot in your terminal
and test the state machine before adding voice.

TODO (voice, added later, without touching state_machine.py):
    user_text = stt.transcribe(audio_input)
    reply_text = session.handle_input(user_text)
    tts.speak(reply_text)
"""

from state_machine import ConversationSession, State


def main():
    session = ConversationSession()
    print("=== Hospital Assistant (text mode) ===")
    print("Type 'quit' to exit.\n")

    # Kick off the greeting without waiting for user input first
    print(f"Bot: {session.handle_input('')}")

    while True:
        user_text = input("You: ").strip()
        # Tolerate trailing punctuation ("quit.", "quit!") and accept "exit"
        # too - but this deliberately does NOT match "quit" as just a
        # PREFIX of a longer sentence, since a patient's real message
        # could legitimately start with that word.
        if user_text.rstrip(".!? ").lower() in ("quit", "exit"):
            break

        reply = session.handle_input(user_text)
        print(f"Bot: {reply}")

        if session.state == State.EMERGENCY:
            # Hard stop - no further "You:" prompt. A flagged emergency
            # ends the session outright rather than staying open.
            print("[Session ended.]")
            break

        if session.state == State.CLOSING:
            break


if __name__ == "__main__":
    main()
