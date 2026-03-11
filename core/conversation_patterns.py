"""
Human-like Interview Conversation Patterns
Implements check-ins, progressive silence, and natural variety
"""

import random
import logging
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ConversationTracker:
    """Tracks conversation state to avoid repetition"""
    last_checkin_idx: int = -1
    last_silence_1_idx: int = -1
    last_silence_2_idx: int = -1
    last_encouragement_idx: int = -1
    last_offtopic_idx: int = -1
    recent_checkins: List[str] = field(default_factory=list)
    
    def get_unique_phrase(self, phrases: List[str], last_idx: int) -> Tuple[str, int]:
        """Get a phrase that wasn't recently used"""
        if len(phrases) == 1:
            return phrases[0], 0
        
        idx = random.randint(0, len(phrases) - 1)
        attempts = 0
        while idx == last_idx and attempts < 5:
            idx = random.randint(0, len(phrases) - 1)
            attempts += 1
        
        return phrases[idx], idx


# ═══════════════════════════════════════════════════════════════
# CHECK-IN PHRASES (After user answers)
# ═══════════════════════════════════════════════════════════════

# After STRONG answer → Ask if they want to add more
CHECKIN_AFTER_STRONG = [
    "Good explanation! Want to share any specific challenges you faced with that?",
    "Nice! Anything else you'd like to add about this?",
    "Great! Is there more detail you want to share on this?",
    "Well explained! Ready to move on, or want to elaborate?",
    "That's comprehensive! Anything else?",
    "Excellent! Should we continue to the next topic?",
    "Good point! Any other points you want to cover here?",
    "Right! Want to add anything else?",
    "Perfect! Ready for the next one?",
    "Good! Anything more on this topic?",
]

# After WEAK answer → Encourage elaboration OR offer to skip
CHECKIN_AFTER_WEAK = [
    "I see. Can you give me a specific example of how you did that, or should we try a different question?",
    "Okay. Want to explain that in more detail, or shall we move on?",
    "Alright. Could you walk me through the actual steps you took?",
    "Got it. Is there a real situation you can describe, or should I ask something else?",
    "I understand. Want to share what tools you used, or try a different topic?",
    "Fair enough. Can you elaborate on that, or would you prefer a different question?",
    "Okay. Any specific details you can add, or should we skip this one?",
]

# After PARTIAL answer → Dig deeper
CHECKIN_FOLLOWUP = [
    "Interesting! What happened next?",
    "I see. How did that turn out?",
    "Okay. What was the result?",
    "Right. And then what did you do?",
    "Got it. Anything else about that?",
    "Alright. Is there more to that story?",
    "Good. What did you do after that?",
]


# ═══════════════════════════════════════════════════════════════
# SILENCE PROMPTS (Progressive - gets more helpful)
# ═══════════════════════════════════════════════════════════════

# 1st silence (gentle)
SILENCE_PROMPT_GENTLE = [
    "Take your time, no rush at all.",
    "It's okay, think it through.",
    "No pressure, answer when ready.",
    "I'm here whenever you're ready.",
    "No hurry, just let me know when you're ready.",
    "Take a moment to think.",
]

# 2nd silence (offer help)
SILENCE_PROMPT_HELPFUL = [
    "Still thinking? That's completely fine. Want me to repeat the question?",
    "No worries at all. Can I help? Should I rephrase this?",
    "Take your time. Would you like to skip this one?",
    "It's okay. Need more time, or try a different question?",
    "No problem. Want me to repeat it, or move on?",
]

# 3rd+ silence (move on)
SILENCE_PROMPT_MOVEDN = [
    "Let me try a different question.",
    "No problem, here's another one.",
    "That's okay, let's switch topics.",
    "Alright, I'll ask you something else.",
]


# ═══════════════════════════════════════════════════════════════
# OTHER RESPONSES
# ═══════════════════════════════════════════════════════════════

# When user says "I don't know"
CANT_ANSWER_RESPONSES = [
    "That's okay! Let me ask you something different.",
    "No problem at all! Here's another question.",
    "It's fine! Let's try a different one.",
    "No worries! Let me change the topic.",
    "That's alright! Moving to something else.",
]

# When user asks to skip
SKIP_RESPONSES = [
    "Sure! Let's move on.",
    "No problem, next one.",
    "Of course! Here's another.",
    "Got it, moving forward.",
]

# Encouragement for good technical answers
TECHNICAL_ENCOURAGEMENT = [
    "Good explanation.",
    "Well explained.",
    "Good answer.",
    "Right, good.",
    "That's correct.",
    "Good point.",
    "Nice, you know this well.",
    "Okay, good.",
    "That makes sense.",
    "Good understanding.",
    "Right.",
    "Yes, that's correct.",
    "Good, I can see you understand this.",
]

# Off-topic redirects (progressive)
OFFTOPIC_GENTLE = [
    "I think you might not be aware of that, let me ask you something different.",
    "No worries, let me move on to a different question.",
    "That's okay, I'll ask you something else instead.",
    "I see, let me try a different topic.",
    "Alright, let's switch to another question.",
    "No problem at all, let me ask you something you might be more familiar with.",
]


# ═══════════════════════════════════════════════════════════════
# MAIN FUNCTION: Determine what to say based on answer quality
# ═══════════════════════════════════════════════════════════════

def get_response_for_quality(
    quality: str,
    stage: str,
    tracker: ConversationTracker,
    silence_count: int = 0,
) -> Tuple[str, str]:
    """
    Returns (response_text, action_type)
    
    action_type can be:
    - "checkin_strong" - Ask if they want to add more or move on
    - "checkin_weak" - Ask if they want to elaborate or skip
    - "checkin_followup" - Dig deeper into partial answer
    - "silence_prompt" - Progressive silence handling
    - "cant_answer" - User said "I don't know"
    - "skip" - User wants to skip
    - "repeat" - User wants question repeated
    - "generate_next" - Generate new question immediately
    """
    
    # ── REPEAT ──
    if quality == "repeat":
        return "", "repeat"  # Empty text - will be filled with pure question
    
    # ── SKIP ──
    if quality == "skip":
        phrase, idx = tracker.get_unique_phrase(SKIP_RESPONSES, -1)
        return phrase, "skip"
    
    # ── SILENCE (Progressive) ──
    if quality == "silence":
        if silence_count == 0:
            phrase, tracker.last_silence_1_idx = tracker.get_unique_phrase(
                SILENCE_PROMPT_GENTLE, tracker.last_silence_1_idx
            )
            return phrase, "silence_prompt"
        elif silence_count == 1:
            phrase, tracker.last_silence_2_idx = tracker.get_unique_phrase(
                SILENCE_PROMPT_HELPFUL, tracker.last_silence_2_idx
            )
            return phrase, "silence_prompt"
        else:
            phrase, _ = tracker.get_unique_phrase(SILENCE_PROMPT_MOVEDN, -1)
            return phrase, "generate_next"  # Auto-generate new question
    
    # ── CAN'T ANSWER ──
    if quality == "cant_answer":
        phrase, _ = tracker.get_unique_phrase(CANT_ANSWER_RESPONSES, -1)
        return phrase, "cant_answer"
    
    # ── OFF-TOPIC ──
    if quality == "off_topic":
        phrase, tracker.last_offtopic_idx = tracker.get_unique_phrase(
            OFFTOPIC_GENTLE, tracker.last_offtopic_idx
        )
        return phrase, "off_topic"
    
    # ── GIBBERISH ──
    if quality == "gibberish":
        return "I'm sorry, I didn't catch that clearly. Could you please repeat your answer?", "gibberish"
    
    # ── STRONG ANSWER ──
    if quality == "strong":
        phrase, tracker.last_checkin_idx = tracker.get_unique_phrase(
            CHECKIN_AFTER_STRONG, tracker.last_checkin_idx
        )
        tracker.recent_checkins.append(phrase)
        if len(tracker.recent_checkins) > 3:
            tracker.recent_checkins = tracker.recent_checkins[-3:]
        return phrase, "checkin_strong"
    
    # ── WEAK ANSWER ──
    if quality == "weak":
        phrase, tracker.last_checkin_idx = tracker.get_unique_phrase(
            CHECKIN_AFTER_WEAK, tracker.last_checkin_idx
        )
        tracker.recent_checkins.append(phrase)
        if len(tracker.recent_checkins) > 3:
            tracker.recent_checkins = tracker.recent_checkins[-3:]
        return phrase, "checkin_weak"
    
    # ── NEUTRAL/PARTIAL ──
    # For neutral answers, use followup check-in
    phrase, _ = tracker.get_unique_phrase(CHECKIN_FOLLOWUP, -1)
    return phrase, "checkin_followup"


def get_encouragement(tracker: ConversationTracker) -> str:
    """Get varied encouragement for good technical answers"""
    phrase, tracker.last_encouragement_idx = tracker.get_unique_phrase(
        TECHNICAL_ENCOURAGEMENT, tracker.last_encouragement_idx
    )
    return phrase


# ═══════════════════════════════════════════════════════════════
# HELPER: Detect user intent from check-in responses
# ═══════════════════════════════════════════════════════════════

def detect_checkin_response(user_response: str) -> str:
    """
    Detect what user wants after a check-in.
    
    Returns:
    - "add_more" - User wants to elaborate
    - "move_on" - User is done, wants next question
    - "normal" - User gave a normal answer (not a meta-response)
    """
    r = user_response.lower().strip()
    
    # ══════════════════════════════════════════════════════════════
    # MOVE ON SIGNALS - Comprehensive list for voice recognition
    # ══════════════════════════════════════════════════════════════
    move_on_phrases = [
        # ── Basic affirmations (short yes/ok) ──
        "yes", "yeah", "yep", "yup", "sure", "okay", "ok", "alright", "all right",
        "fine", "good", "right", "correct", "absolutely",
        
        # ── Done/Finished ──
        "done", "finished", "complete", "completed", "i'm done", "im done",
        "that's done", "thats done", "all done", "i'm finished", "im finished",
        "i've finished", "ive finished", "finished with that",
        
        # ── That's all/everything ──
        "that's all", "thats all", "that's it", "thats it", "that is all",
        "that is it", "nothing else", "nothing more", "no more", "nothing",
        "that's everything", "thats everything", "that's all i have",
        "thats all i have", "that's all for now", "thats all for now",
        
        # ── Nothing to add ──
        "nothing to add", "nothing else to add", "no more to add",
        "i have nothing to add", "i don't have anything to add",
        "i dont have anything to add", "nothing further",
        
        # ── Move forward ──
        "next", "next question", "next one", "next one please",
        "move on", "let's move on", "lets move on", "we can move on",
        "continue", "let's continue", "lets continue", "we can continue",
        "go ahead", "go on", "proceed", "carry on", "keep going",
        
        # ── Combinations ──
        "yes next", "yeah next", "yep next", "sure next", "ok next", "okay next",
        "yes let's move on", "yeah lets move on", "yes continue", "yeah continue",
        "alright next", "alright move on", "fine next", "good next",
        
        # ── Ready to continue ──
        "ready", "i'm ready", "im ready", "ready to move on", "ready for next",
        "ready for the next one", "ready to continue", "we can go ahead",
        
        # ── That covers it ──
        "that covers it", "that covers everything", "covered it all",
        "that's covered", "thats covered", "covered", "we covered it",
        "i've covered it", "ive covered it", "i think we covered it",
        
        # ── Agreement phrases ──
        "i think so", "i guess so", "i believe so", "probably",
        "i think that's it", "i guess that's it", "i believe that's it",
        "i think that's all", "i guess that's all", "i believe that's all",
        "seems like it", "looks like it", "yeah i think so",
        
        # ── Polite versions ──
        "shall we move on", "can we move on", "should we move on",
        "shall we continue", "can we continue", "should we continue",
        "we should move on", "we should continue", "let us move on",
        "let us continue",
        
        # ── No elaboration needed ──
        "no need", "no need to elaborate", "no need for more",
        "not really", "nope", "nah", "no thanks", "no thank you",
        "i don't think so", "i dont think so", "don't think so",
        "dont think so", "no i'm good", "no im good",
        
        # ── Complete thought expressions ──
        "that's all i wanted to say", "thats all i wanted to say",
        "that's all i have to say", "thats all i have to say",
        "i've said everything", "ive said everything",
        "that's everything i know", "thats everything i know",
        "that's all i know", "thats all i know",
        "i think that's everything", "i think thats everything",
        
        # ── Indian English patterns (common in India) ──
        "that's all only", "thats all only", "only this much",
        "this much only", "bas that's all", "bas thats all", "bas",
        "nothing more sir", "nothing else sir", "that's it sir",
        "thats it sir", "no sir", "yes sir next", "ok sir",
        
        # ── Casual/conversational ──
        "yup that's all", "yup thats all", "yep that's it", "yep thats it",
        "yeah that's everything", "yeah thats everything",
        "nope nothing else", "nah nothing else", "nope nothing more",
        "nah nothing more", "that'll do", "that will do",
        "that should do", "that's enough", "thats enough",
        
        # ── Transcription variations (STT might mishear) ──
        "that is all", "that is it", "next please", "next one please",
        "move to next", "go to next", "skip to next", "continue please",
        "proceed please", "let us proceed", "lets proceed",
        
        # ── Short variants ──
        "all", "enough", "sufficient", "good enough", "that'll work",
        "that will work", "works for me", "sounds good",
    ]
    
    # Check if response contains any move-on phrase
    if any(phrase in r for phrase in move_on_phrases):
        return "move_on"
    
    # ── Exact ultra-short responses that mean "done" ──
    exact_done = [
        "no", "nope", "nah", "done", "yes", "yeah", "yep", "yup",
        "ok", "okay", "sure", "nothing", "all", "next", "continue",
        "right", "correct", "good", "fine", "alright",
    ]
    
    if r in exact_done:
        return "move_on"
    
    # ══════════════════════════════════════════════════════════════
    # ADD MORE SIGNALS - User wants to elaborate
    # ══════════════════════════════════════════════════════════════
    add_more_phrases = [
        "let me add", "let me explain", "let me elaborate", "let me tell you",
        "i want to add", "i'd like to add", "id like to add", "i should mention",
        "i forgot to mention", "oh and", "also", "and also", "one more thing",
        "actually", "wait", "hold on", "let me think", "oh yes",
        "i remember", "i forgot", "another thing",
    ]
    
    if any(phrase in r for phrase in add_more_phrases):
        return "add_more"
    
    # ══════════════════════════════════════════════════════════════
    # OTHERWISE - Treat as normal response (user is elaborating)
    # ══════════════════════════════════════════════════════════════
    # If user said something substantial (>5 words), they're probably elaborating
    word_count = len(r.split())
    if word_count >= 5:
        return "normal"
    
    # Short responses that aren't move-on signals = probably elaborating
    return "normal"