#!/usr/bin/env python3
"""Generate the labeled evaluation set.

PROVENANCE: every item is SYNTHETIC, hand-written for this repo in the style of
Quora answers (clean expertise, link-farm spam, personal attacks, AI-slop).
No real user content is included. Labels are assigned by construction.
"""
import json
import random

random.seed(7)

CLEAN = [
    "In my experience as a structural engineer, the usual cause is uneven soil settlement. A qualified inspector should check the foundation before any cosmetic repair.",
    "The short answer is compound interest. Starting at 25 instead of 35 roughly doubles your retirement balance at the same monthly contribution, because the extra decade does the heavy lifting.",
    "Python's GIL means threads share one interpreter lock, so CPU-bound work scales poorly with threading. Use multiprocessing or native extensions for CPU-bound tasks; threads are fine for I/O.",
    "When I moved to Toronto, the biggest surprise was winter clothing cost. Budget around $800 for a proper parka, boots, and layers your first year.",
    "Photosynthesis has two stages: the light reactions in the thylakoid membranes produce ATP and NADPH, and the Calvin cycle in the stroma fixes CO2 into sugars.",
    "I fixed this exact bug last year. The race condition came from a shared counter updated outside the mutex. Moving the increment inside the lock made it deterministic.",
    "A practical way to learn linear algebra is 3Blue1Brown's series for intuition, then Strang's MIT course for rigor, doing every exercise by hand.",
    "The difference between a Roth and a traditional IRA is when you pay tax: Roth contributions are taxed now and grow tax-free; traditional contributions are deductible now and taxed at withdrawal.",
    "For a first marathon, most coaches suggest a 16-week plan with three runs a week and one long run, increasing weekly distance by no more than 10 percent.",
    "SQLite is fine for a single-writer app. Once you need concurrent writes from multiple services, move to a client-server database like PostgreSQL.",
    "I interviewed at five startups last spring. Every one asked about past projects in depth; nobody asked brain teasers. Prepare two stories with measurable outcomes.",
    "The reason airplane windows are round is stress concentration: square corners focus stress and led to metal fatigue failures in the early jet age.",
]

SPAM = [
    "Make $5000 a week from home! Click here now to learn the secret banks don't want you to know http://bit.ly/x1 http://bit.ly/x2",
    "Buy cheap followers and grow your account fast. Visit our link now for a discount on all packages http://spam.example/deal",
    "Free crypto giveaway! Earn $200 today, just send your wallet address. WhatsApp +1234567890 for instant payout",
    "This worked for me, check my profile link for free pills at a huge discount, buy now while stocks last",
    "CLICK NOW to get free money from the government program they are hiding. Visit http://scam.example and http://scam2.example today",
    "I make $9000 monthly trading crypto. Click the link in my bio to join my free signals group now",
    "Discount meds online, no prescription needed, cheap pills free shipping. Buy now at http://pills.example",
    "Earn $50 per survey, guaranteed payout. Click here now and visit our sponsor link to start today",
]

HARASS = [
    "Anyone who believes this is a complete idiot. You people are too stupid to understand basic economics.",
    "Shut up. You have no idea what you are talking about and everyone in this thread knows you are a liar.",
    "What a pathetic answer. You are a fraud and a hack, and whoever upvoted this is a moron too.",
    "You are worthless at your job if you think that's how memory management works. Stop answering questions.",
    "Only an idiot would recommend that framework. You people ruin every discussion with your stupid opinions.",
    "This is the dumbest thing I have read all week. The author is a subhuman waste of space.",
]

AI_SLOP = [
    "As an AI language model, I don't have personal experiences, but I can provide a general overview. In conclusion, it depends on many factors. I hope this helps!",
    "It's important to note that this is a nuanced topic. Furthermore, one must consider various perspectives. Moreover, context matters greatly. In today's fast-paced world, this is crucial.",
    "Let's delve into this fascinating question. First and foremost, there are several key considerations. In conclusion, the answer depends on your specific needs. I hope this answer finds you well!",
    "In today's digital age, this question is more relevant than ever. It's important to note that many factors play a role. Furthermore, expert opinions vary. I hope this helps!",
    "Great question! Let's delve into the details. First and foremost, it is essential to understand the fundamentals. Moreover, every situation is unique. In conclusion, do your own research.",
    "As an AI, I can say that this is a multifaceted issue. It's important to note that different approaches exist. Furthermore, best practices evolve constantly. I hope this answer finds you well!",
]

def main():
    rows = []
    for t in CLEAN:
        rows.append({"text": t, "labels": {"spam": False, "harassment": False, "ai_slop": False}})
    for t in SPAM:
        rows.append({"text": t, "labels": {"spam": True, "harassment": False, "ai_slop": False}})
    for t in HARASS:
        rows.append({"text": t, "labels": {"spam": False, "harassment": True, "ai_slop": False}})
    for t in AI_SLOP:
        rows.append({"text": t, "labels": {"spam": False, "harassment": False, "ai_slop": True}})
    # hard negatives: clean answers containing tricky words
    rows.append({"text": "If you get spam like 'earn $500 fast' in your inbox, report it as phishing instead of clicking any link.", "labels": {"spam": False, "harassment": False, "ai_slop": False}})
    rows.append({"text": "Calling an idea 'stupid' in a code review is fine; calling a person stupid crosses the line into harassment. Focus comments on the work.", "labels": {"spam": False, "harassment": False, "ai_slop": False}})
    random.shuffle(rows)
    with open("data/labeled_answers.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows")

if __name__ == "__main__":
    main()
