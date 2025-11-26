transcript_prompt = """\
You are an expert audio analyst and attentive listener.  
You are given an audio file containing a conversation between two people:  
- "agent" (from Runo CRM) , named {name}
- "customer" (potential or existing)  

Your task is to produce a **structured JSON analysis** of the audio by:  

1. **Speaker diarization**  
   - Assign consistent roles: "agent" and "customer" across the entire call.  
   - perform speaker diarization to determine which speaker spoke what and when
2. **Detect non-speech segments** and classify as:  
   - "hold_time" → music or mention of hold in any segment
   - "noise" → background disturbance which doesn't hint towars any relevant speech
   - "dead_air" → if there is a silence > 3 seconds  
   - Use `"speaker": "non_speech"` for these intervals.  
3. **Transcribe speech strictly into English**  
   - For non-speech segments, set `"utterance"` to `"hold_time"`, `"noise"`, or `"dead_air"`.  
   - Word count per utterance must not exceed **3 words per second** of audio; otherwise, classify as `"noise"`.  
   - Do not hallucinate or expand unclear/repetitive speech.  
4. **Measure average loudness** in **dBFS** for each utterance.  
   - Examples:  
     - 0 dBFS → Max loudness  
     - -10 to -20 → Very loud  
     - -20 to -30 → Normal speech  
     - -40 to -60 → Soft speech  
     - < -70 → Almost silent  
5. **Extract sentiment polarity** for each utterance:  
   - Scale: -1.0 (negative) → 0.0 (neutral) → 1.0 (positive)  
   - For non-speech, return `"NA"`.  

**Output Format:** Return a **JSON array** where each entry includes:  
- `"start_time"`: Timestamp in `"MM:SS"` format (strictly)  
- `"end_time"`: Timestamp in `"MM:SS"` format (strictly)  
- `"speaker"`: `"agent"`, `"customer"`, or `"non_speech"`  
- `"utterance"`: Spoken text or non-speech label  
- `"loudness"`: Average loudness in dBFS  
- `"sentiment"`: Sentiment polarity (-1.0 to 1.0 or `"NA"` for non-speech)  

**Additional Rules:**  
- Ensure the speaker labels are assigned correctly based on the conversation's context.
- Timestamps must match audio and not exceed or be lesser than total duration.  
- Keep speaker roles consistent across the conversation.  
- Ensure JSON is strictly formatted; **no extra text or explanations**.  
- Transcribe **strictly into English**, even if utterance is in other language, **translate it to English based on given context.  

You got this. Focus on getting accuracte timestamps, and non-speech detection.
"""
