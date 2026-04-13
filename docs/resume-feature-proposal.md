# Resume Feature Proposal

## What This Project Does

We built an AI chatbot that helps students find Coursera courses. The chatbot asks five questions — your background, what topic you want to learn, how much time you have, which provider you prefer, and what difficulty level — and then recommends courses based on your answers. The recommendations are powered by a combination of semantic search (finding courses that are conceptually similar to what you described) and an AI model (Google Gemini) that writes a short explanation for why each course fits you.

It works, but the information it gets from the student is limited. Five quick chat answers can only tell us so much.

---

## What We Want to Add

We want to let students share their resume with the chatbot before the conversation starts.

The chatbot would read the resume, pull out useful information (like the student's major, skills, and career interests), and use that to give better recommendations — without making the student answer a long list of questions.

The goal is simple: **the more the chatbot knows about you upfront, the better the recommendations it can give.**

Here is what the new flow would look like:

1. Student opens the chatbot
2. Chatbot asks: "Would you like to share your resume for more personalized recommendations?"
3. Student pastes their resume text (or skips this step)
4. The system reads the resume and extracts key information
5. Chatbot asks 1–2 short follow-up questions for anything the resume did not make clear
6. Chatbot recommends courses based on both the resume and the follow-up answers

---

## What Information We Would Extract from the Resume

We do not need everything on a resume. We only need the parts that help us recommend courses.

The fields we plan to extract:

| Field | Example |
|-------|---------|
| Background / Major | Computer Science, Electrical Engineering |
| Technical skills | Python, SQL, TensorFlow |
| Experience level | Student, 1–2 years of work experience |
| Interest areas | Machine learning, cloud infrastructure |
| Career goal | Data scientist, ML engineer |

We would use an LLM (probably the same Gemini model we already use) to read the resume text and return these fields in a structured format. We are not training a new model — we are writing a good prompt and letting an existing AI do the extraction.

---

## What Is Still Unclear — Things We Need to Decide as a Group

These are the questions that do not have a clear answer yet. We should decide them before writing code, otherwise we will need to rewrite things later.

---

### Question 1: Is the resume step required or optional?

If it is **optional**, students who do not have a resume (or do not want to share one) can still use the chatbot the same way as before. The resume just becomes an extra input that improves the experience.

If it is **required**, every student has to go through the resume step, which changes the whole conversation flow.

This matters because it changes how we design the opening of the conversation.

---

### Question 2: If the resume already answers some questions, do we still ask them?

Right now the chatbot asks: background → topic → skill → time → provider → difficulty.

If a student's resume already makes their background and skills obvious, it feels redundant to ask those questions again. But we also want to give students a chance to confirm or correct what we extracted.

The options are:
- **Skip questions the resume already answered** — faster, but riskier if the extraction was wrong
- **Show pre-filled answers and ask the student to confirm** — friendlier and more accurate, but takes slightly longer
- **Keep all questions as-is** — simplest to build, but misses the point of having the resume

---

### Question 3: How exactly does the resume information improve the recommendations?

This is a technical question but it has a real impact on whether the feature actually feels useful.

Right now, the chatbot builds a search query from the student's answers, then finds matching courses. If we add resume information, we need to decide:
- Does the resume data get added to the search query to find better matches?
- Does it change how results are ranked (for example, prioritize courses that match the student's career goal)?
- Or does it only influence the follow-up questions, not the search itself?

If the resume does not actually change the courses we find, the feature feels like a lot of effort for very little benefit.

---

### Question 4: What should the input look like on screen?

Two realistic options:

**Option A: A text box before the chat starts**
The student sees a box that says "Paste your resume here (optional)" before the conversation begins. Clean and obvious, but requires UI changes.

**Option B: The chatbot asks within the chat**
The chatbot's first message says "Hi! You can share your resume to get better recommendations — just paste the text here, or type Skip to continue without one." No separate UI needed, easier to build.

---

### Question 5: What if the resume is messy or hard to read?

Some resumes are short, poorly formatted, or written in unusual styles. The LLM might not extract useful information from them. We need a plan for this:

- If extraction fails completely, fall back to the original five-question flow
- If only some fields are extracted, ask follow-up questions for the missing ones
- If the student pastes something that is clearly not a resume, the chatbot should gently ask them to try again or skip

---

### Question 6: Does the professor expect us to train a model?

Based on our current plan, we would use an existing LLM (Gemini) to do the resume extraction — not train anything ourselves. This is a valid engineering approach, but we should confirm with the professor or TA that this satisfies the project requirements.

---

## Our Recommended Decisions

Based on what we know right now, here is what we suggest. These are not final — they are a starting point for the group discussion.

| Question | Our Recommendation |
|----------|-------------------|
| Required or optional? | **Optional.** Students who skip it get the same experience as before. |
| What to do with questions the resume answers? | **Show pre-filled answers and let the student confirm.** More work to build, but much better experience. |
| How does resume data improve recommendations? | **Add it to the search query and use it for ranking.** Otherwise the feature is not really useful. |
| What does the input look like? | **Option B — ask inside the chat.** Simpler to build, no UI redesign needed. |
| What if extraction fails? | **Fall back to the normal question flow silently.** The student does not need to see an error. |
| Do we train a model? | **No — use Gemini with a well-designed prompt.** Confirm this with the professor. |

---

## MVP Scope

For the first working version, we suggest:

- Student can paste resume text into the chat (optional)
- System extracts 5 fields: background, skills, experience level, interests, career goal
- Chatbot skips or pre-fills questions for fields that were found
- Chatbot asks follow-up questions for anything missing
- Extracted data is used in the search query and influences course ranking
- No file upload, no database storage — session only

This is enough to show the feature works and demonstrate real improvement in recommendation quality.
