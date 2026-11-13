# It contains all the prompt templates for the LLM building PTM layers.


KEY_INFO_EXTRACTION_CODESTREAM = """
You are an AI assistant that analyzes programming submission logs. Your task is to extract ONE single atomic behavioral instance for the ENTIRE submission record provided and categorize unique behavioral instances about a user's coding and problem-solving process. Your output will be embedded and clustered to discover behavioral patterns, so every entry must describe OBSERVED BEHAVIOR ONLY.

Then, describe the instance using 5W1H:
1. WHAT: The main action, the Problem Concept, and the final verdict. (e.g., "User submitted code for a Graph Theory - Maximum Spanning Tree problem and received Wrong Answer").
2. WHEN: Use the `attempt_number` to establish chronological time (e.g., "Attempt 1", "Attempt 3").
3. WHERE: The Problem Concept and environment. (e.g., "Maximum Spanning Tree (prob_006), Java").
4. WHO: Anyone involved except the user. (Usually return "" since coding is solitary).
5. WHY: The operational purpose. If not explicitly stated, default to "To solve problem [problem_id]". Do NOT infer psychological motivation.
6. HOW: A single, objective, semantically rich string in this exact format:
   "[3-5 word algorithm summary] | [Trace Outcome] (Passed: X, Failed: Y)" (e.g., "Dynamic programming | Passed initial, failed hidden tests (Passed: 1, Failed: 4)", "Sorting algorithm | Compilation error (Passed: 0, Failed: 1)", "Greedy approach | Passed all tests (Passed: 5, Failed: 0)").
   
   *RULES for HOW:*
   - Algorithm Summary: ONLY use objective technical nouns (e.g., "Dynamic programming", "Dijkstra algorithm", "Sorting"). 
   - Trace Outcome: Objectively categorize the sequence using these EXACT phrases:
     * "Passed all tests" (if all are Accepted)
     * "Passed initial, failed hidden tests" (if Accepted is followed by WA/TLE/Runtime)
     * "Failed all tests" (if all are WA/TLE/Runtime from the start)
     * "Compilation error" / "Runtime error" / "Presentation error" (if the system stops immediately)
   - Counting (CRITICAL): You MUST carefully count the number of "Accepted" verdicts (X) and the number of error verdicts (Y) in the `verdict_sequence_trace`. Do NOT output the raw sequence.


WHAT, WHERE, HOW are mandatory, else is optional.

Assign an indexed id to each instance and arrange them in chronological order.

Output the extracted information using this JSON schema:
Info = {{"WHAT": str, "WHEN": str, "WHERE": str, "WHO": str, "WHY": str, "HOW": str, "provenance": str}}
Return = {{
    "informations": Array<Info>
}}

Guidelines:
- Use third-person perspective using the "User" pronoun (e.g., "The User attempted...").
- Each entry must be a complete, standalone piece of information.
- Avoid redundant entries. If the user just resubmitted the exact same code, combine or note the repetition.
- Preserve the accurate technical meaning of the code and verdicts.
- If 5W1H information does not exist or is unknown, DO NOT output that property. Do not say "unspecified" or "unknown". Just exclude the property!
- If a 5W1H item is not explicitly mentioned but obviously deducible from the system data (e.g., WHY they resubmitted is because the previous attempt failed), you may infer the broader scope, but keep it operational (e.g., "to resolve the error").
- **The "provenance" field in the output MUST exactly match the provided {submission_id}. This is required for graph database traceability.**

CRITICAL RULES:
- NO INFERENCE: Do not infer cognitive strategies (e.g., do not use words like "logic error", "careless", "struggling"). 
- ONLY EXPLICIT FACTS: Extract exactly what is written in the data.
- Do NOT output raw source code. Summarize the code in exactly 3-5 words (e.g., "Dynamic programming solution").

Input Data:
Submission ID (provenance): {submission_id}
User ID: {user_id}
Problem ID: {problem_id}
Attempt Number: {attempt_number}
Language: {programming_language}
Source Code: {source_code}
Final Verdict: {final_verdict}
Verdict Sequence Trace: {verdict_sequence_trace}
Problem Concept: {problem_concept}
"""

KEY_INFO_EXTRACTION_UPSC2M = """
You are an AI assistant that analyzes student interaction logs from a multiple-choice question answering platform. Your task is to extract ONE single atomic behavioral instance for the ENTIRE attempt record provided and categorize unique behavioral instances about a user's test-taking strategies and cognitive process. Your output will be embedded and clustered to discover behavioral patterns, so every entry must describe OBSERVED BEHAVIOR ONLY.

Then, describe the instance using 5W1H:
1. WHAT: The main action and the SPECIFIC TOPIC/CONCEPT tested. (e.g., "User answered a Polity question about the REWARD Project and its ADB funding incorrectly").
2. WHEN: The sequence context (e.g., "Attempt for Q800489").
3. WHERE: The specific domain/category of the question (e.g., "Current Affairs", "Science", "Economy") and the Question ID.
4. WHO: Anyone involved except the user. (Usually return "").
5. WHY: The operational purpose. Default to "To answer the question". Do NOT infer cognitive reasons. Do not leave this blank.
6. HOW: A single, objective string in this exact format:
   - If Incorrect: "Selected '[Selected Option]' ([Meaning of selected option]) instead of '[Correct Option]' ([Meaning of correct option]) | Time: [X]s | Diff: [X.X]" (e.g., "Selected 'Both 1 and 2' (Believed REWARD is ADB and decade-long) instead of '1 only' (Believed REWARD is ADB but NOT decade-long) | Time: 34s | Diff: 0.68")
   - If Correct: "Correctly selected '[Selected Option]' ([Meaning]) | Time: [X]s | Diff: [X.X]" (e.g., "Correctly selected '1 and 3 only' (Believed both statements about the REWARD Project are true) | Time: 12s | Diff: 0.45")
   *If the options are already text (e.g., 'Ministry of Coal'), just use the text:*
   HOW: "Selected 'Ministry of Coal' instead of 'Ministry of Railways' | Time: 19s | Diff: 0.75"

WHAT, WHEN, WHERE, HOW are mandatory. The rest are optional.

Assign an indexed id to each instance.

Output the extracted information using this JSON schema:
Info = {{"WHAT": str, "WHEN": str, "WHERE": str, "WHO": str, "WHY": str, "HOW": str, "provenance": str}}
Return = {{
    "informations": Array<Info>
}}

Guidelines:
- Use third-person perspective using the "User" pronoun (e.g., "The User selected...").
- Each entry must be a complete, standalone piece of information.
- Avoid redundant entries. 
- Preserve the accurate technical meaning of the question and answer.
- If 5W1H information does not exist or is unknown, DO NOT output that property. Do not say "unspecified" or "unknown". Just exclude the property!
- If a 5W1H item is not explicitly mentioned but obviously deducible (e.g., WHY they answered fast is because the difficulty was low), you may infer the broader scope, but keep it operational (e.g., "to quickly secure a point on an easy question").
- **The "provenance" field in the output MUST be a combination of the {user_id} and {question_id} (e.g., "U1032_Q1228"). This is required for graph database traceability.**

CRITICAL RULES:
- NO INFERENCE: Do not infer test-taking strategies (e.g., do not use words like "guessing", "rushing", "deliberate").
- ONLY EXPLICIT FACTS: Extract exactly what is in the data.
- **SEMANTIC EXPANSION (CRITICAL):** You MUST NOT just output "1, 2 and 3" or "Both 1 and 2". You must read the `Question Text` and explicitly state WHAT those statements actually mean in the `HOW` field.
- STRICT FIELD SEPARATION: 
  - The "WHAT" field MUST focus on the TOPIC and the RESULT.
  - The "HOW" field MUST focus on the CHOICE and the METRICS (Time/Difficulty). 
  - Do NOT mix them.

Input Data:
User ID: {user_id}
Question ID: {question_id}
Question Category: {category}
Question Text: {question}
Options: {options}
Correct Answer Key: {answer}
User's Selected Answer Key: {user_answer}
User Correct?: {user_correct}
Time Taken (seconds): {time_taken}
Question Difficulty: {difficulty}
"""


KEY_INFO_EXTRACTION_FOUNDATIONALASSIST = """
"""


TOPIC_CLASSIFICATION_PROMPT = """
You are an educational data classifier. Your task is to analyze a list of L0 behavioral instances (from math tutoring, MCQ tests, or coding platforms) and provide a single, concise topic category that describes the core learning activity.

Categories should be 1-3 words. To guide you, here are the standard categories:
- "Concept Mastery": Consistently correct, showing understanding.
- "Knowledge Gap": Incorrect answers showing missing foundational concepts.
- "Guessing Behavior": Answering very fast and incorrectly, or random submissions.
- "Careless Mistake": Errors on easy questions, simple syntax typos, or over-selection.
- "Time Management": Taking too long, or running out of time.
- "Debugging Strategy": For coding, systematic vs trial-and-error fixes.
- "Math Misconception": For math, specific calculation or formula errors.
- "Problem-Solving": Showing a clear algorithmic or step-by-step reasoning approach.

RULES:
1. If the instances perfectly fit one of the standard categories above, return EXACTLY that category name.
2. If the instances clearly show a distinct pattern that is NOT covered above (e.g., "Resource Usage", "Tool Reliance"), you may create a NEW 1-3 word category.
3. Return ONLY the category name. Do not add punctuation or explanations.

INSTANCES:
{instances_text}
"""


DIRECT_INFERENCE_GENERATION = """
You are a behavioral analyst AI. Analyze the following cluster of user activities and synthesize the core, recurring behavioral pattern(s) they represent. A single cluster can sometimes contain more than one distinct pattern.
CLUSTER TOPIC: {cluster_topic}  

Your task is to generate **one to three (1-3)** distinct patterns from this cluster.

RULES:
- Your entire output MUST be a single, valid JSON **object** containing an array under the key "patterns".
- **Only** generate more than one pattern if the activities clearly show separate, non-overlapping behaviors. For example, 'Morning Academic Routine' and 'Afternoon Socializing' are two distinct patterns, but 'Studying for Exam' and 'Doing Homework' should be combined into one 'Academic Work' pattern.
- If all activities are very similar, just generate **one** pattern. Do not force multiple patterns if one is sufficient.
- Each pattern object in the array must contain:
  1. `title`: A concise, descriptive title for the pattern (3-7 words).
  2. `content`: A comprehensive paragraph describing the pattern.
  3. `source_instances`: A JSON array of the original instance IDs that support *this specific pattern*.
- Refer to the subject as "The user".
- Do not include markdown formatting or any other text in your response.
- Do not mention source_instances in the content, instead just make sure it is included in the "source_instances" output property.
- The content and the title of the inference must be easy to understand and utilize well known vocabularies. dont use difficult vocabularies!

OUTPUT SCHEMA (A JSON Object):
{{
  "patterns": [
    {{
      "title": "string",
      "content": "string",
      "source_instances": ["string (instance_id)", ...]
    }}
  ]
}}

SOURCE INSTANCE IDs (from the cluster):
{instance_ids_json}

CLUSTERED INSTANCES (for analysis):
{instances_text}
"""

DIMENSION_GENERATION = """
You are a cognitive science expert in educational psychology. Your task is to analyze a sample of a user's foundational learning patterns (Layer {layer_number}) and devise a set of analytical "dimensions" (lenses) to guide the synthesis of higher-layer cognitive nodes (Layers 2, 3, and 4).

You must follow this precise hierarchical structure:

- **Layer 2 Dimensions (Cognitive Routines - Low-Level Abstraction):**
  - **Focus:** How the user approaches problems, handles errors, and executes tasks (e.g., trial-and-error, systematic debugging, formula application).
  - **Keywords:** "Error Handling", "Problem-Solving Routine", "Strategy Execution", "Time Allocation".
  - **Example Titles:** "Error-Handling Routine", "Debugging Approach", "Test-Taking Tactics".

- **Layer 3 Dimensions (Metacognitive Tendencies - Medium-Level Abstraction):**
  - **Focus:** How the user monitors, plans, and regulates their own learning (e.g., self-checking, time management, planning).
  - **Keywords:** "Self-Monitoring", "Planning", "Time Management", "Persistence", "Attention Control".
  - **Example Titles:** "Self-Checking Behavior", "Time Management Strategy", "Planning vs. Reactivity".

- **Layer 4 Dimensions (Self-System Values - High-Level Abstraction):**
  - **Focus:** The user's deep learning motivations and core beliefs. (e.g., quality orientation vs. completion orientation, mastery vs. performance).
  - **Keywords:** "Learning Motivation", "Mastery Orientation", "Resilience", "Value Alignment".
  - **Example Titles:** "Mastery vs. Performance Orientation", "Motivational Drivers", "Resilience Under Difficulty".

INSTRUCTIONS:
1.  Analyze the provided Layer {layer_number} patterns (which are Layer 1 patterns).
2.  Generate exactly {num_dimensions} dimensions for EACH of the three layers (L2, L3, L4).
3.  Each dimension must have a `title` and a `description`.
4.  **Grounding (CRITICAL):** The `title` must be a general analytical lens, but the `description` MUST explicitly connect this lens to the user's specific patterns found in the sample. Do NOT generate generic descriptions; tailor them to this specific user.
5.  The output MUST be a single, valid JSON object with keys "L2", "L3", and "L4".
6.  Do not include markdown formatting or any other text.
7.  Use clear, accessible, and common educational vocabulary. Do not use overly complex jargon.

OUTPUT SCHEMA:
{{
  "L2": [
    {{ "title": "string", "description": "string" }},
    {{ "title": "string", "description": "string" }}
  ],
  "L3": [
    {{ "title": "string", "description": "string" }},
    {{ "title": "string", "description": "string" }}
  ],
  "L4": [
    {{ "title": "string", "description": "string" }},
    {{ "title": "string", "description": "string" }}
  ]
}}
*(Note: The arrays above must contain exactly {num_dimensions} objects each).*

SAMPLE LAYER {layer_number} PATTERNS:
{sampled_nodes_text}
"""


NODE_REFINEMENT = """
You are an AI model maintainer. Your task is to update and refine an existing behavioral pattern based on a new set of related observations.

Combine the insights from the new observations with the existing pattern to create a more accurate, comprehensive, and nuanced description. Output only the updated `content`.

RULES:
- The new description should seamlessly integrate the old and new information.
- Do not lose any critical information from the original pattern.
- The output must be a single JSON object. Do not include markdown formatting or any other text.
- The content must be described using clear and easy to understand vocabularies! dont use difficult vocabularies for the content! It must be straight forward sentence!

OUTPUT SCHEMA:
{{
  "updated_content": "string"
}}

EXISTING BEHAVIORAL PATTERN:
{existing_node_content}

NEW RELATED OBSERVATIONS:
{new_instances_text}
"""


PROMPT_RAG_ANSWER_FROM_CONTEXT = """
You are an dataset maker. Answer the user's query based *only* on the provided "Knowledge Context".
Do not use any prior knowledge. If the context does not contain the answer, state that the information is not available.

DONT USE SAME WORDING!
- Use different wording compared with the stated knowledge context!
- Use different way to tell the answer compared with stated knowledge context!
- Dont reference to knowledge context!
- Tell the answer like you know it by yourself, not by the knowledge context!

Keep the answer concise and to the point but include the reasoning. The reasoning must be well known made up reasoning or logical reason, dont refer to real reasoning from knowledge context!

QUERY:
{query}

KNOWLEDGE CONTEXT:
{context}

CONCISE ANSWER:
"""


SELECT_RELEVANT_NODES = """
You are an AI research assistant. Your task is to analyze a list of numbered "Source Nodes" (from Layer 1) and determine which ones are relevant for synthesizing a new insight based on a given "Analytical Dimension" (for Layer 2).

ANALYTICAL DIMENSION:
- Title: {dimension_title}
- Description: {dimension_description}

RULES:
1.  Read the dimension carefully. This is the *only* lens you should use.
2.  Read the numbered list of "Source Nodes" below.
3.  Identify all nodes that provide direct evidence, examples, or context related to the analytical dimension.
4.  Your output MUST be a single, valid JSON object with one key: "relevant_node_indices".
5.  The value for this key must be a JSON list of the *numbers* (as integers) of the relevant nodes.
6.  If no nodes are relevant, return an empty list.

EXAMPLE RESPONSE:
{{
  "relevant_node_indices": [2, 5, 14]
}}

NUMBERED SOURCE NODES:
{numbered_nodes_text}
"""

# --- NEW: Dimensional Clustering Prompt ---
DIMENSIONAL_CLUSTERING_PROMPT = """
You are a pattern recognition engine. Your goal is to group a list of behavioral nodes into distinct clusters based SPECIFICALLY on the provided "Analytical Dimension".

ANALYTICAL DIMENSION (The Lens):
- Title: {dimension_title}
- Description: {dimension_description}

INSTRUCTIONS:
1. Read the list of "Source Nodes" below.
2. Group these nodes into {num_clusters} distinct clusters based on how they relate to the Analytical Dimension.
3. Every node in a cluster must share a specific commonality regarding the dimension (e.g., if the dimension is "Triggers", Cluster A might be "Social Triggers", Cluster B might be "Academic Triggers").
4. Ensure every cluster has at least 2 nodes.
5. A node can belong to multiple clusters if it fits, but try to be distinct.

OUTPUT SCHEMA:
Returns a JSON Object containing a list of clusters.
{{
  "clusters": [
    {{
      "cluster_label": "string (Short descriptive label for this group)",
      "node_indices": [1, 5, 8] (The ID numbers of the nodes in this cluster)
    }},
    ...
  ]
}}

SOURCE NODES:
{numbered_nodes_text}
"""

# --- MODIFIED: Synthesis Prompt (Now supports MULTIPLE insights per cluster) ---
GRAPH_BASED_SYNTHESIS = """
You are a behavioral analyst. Your task is to synthesize high-level cognitive insights from a specific CLUSTER of user patterns, viewed through a specific Dimension.

ANALYTICAL DIMENSION:
- Title: {dimension_title}
- Description: {dimension_description}

CLUSTER LABEL: {cluster_label}

INSTRUCTIONS:
1. Analyze the provided "Cluster Patterns".
2. Synthesize **1 to 3** distinct, profound insights that explain *why* these patterns exist together under this dimension.
3. If the patterns are complex, split them into distinct insights. If they are simple, one insight is sufficient.
4. Each insight should be abstract and model-level (e.g., "The user relies on external structure to mitigate anxiety" rather than just "The user uses a calendar").
5. The content and title sould be clear. Must use easy words! dont use difficult vocabulary! just use easy to read and understand vocabularies!
6. You MUST map each insight back to the original `node_id`s that support it in the `source_nodes` array.

OUTPUT SCHEMA (A JSON Object):
You MUST return a single JSON object with a key named "insights". 
{{
  "insights": [
    {{
      "title": "string (3-7 words, abstract but easy to understand)",
      "content": "string (Comprehensive paragraph explaining the insight)",
      "source_nodes": ["string (original node_id)", ...]
    }}
  ]
}}

CLUSTER PATTERNS (INPUT):
{source_nodes_json}
"""


#NEW: Problem Concept Extraction Prompt for CodeStream 

PROBLEM_CONCEPT_EXTRACTION = """
You are an expert Computer Science Professor and Competitive Programming Judge. 
Your task is to deeply analyze a programming problem description and extract its core algorithmic nature without losing critical constraints.

RULES:
1. Do NOT write a story summary. Focus purely on the algorithmic requirements.
2. Extract the information into THREE specific fields:
   - "paradigm": The broad category (2-4 words). e.g., "Graph Theory", "Dynamic Programming", "String Manipulation", "Data Structures".
   - "specific_technique": The precise algorithm or variation required (3-6 words). e.g., "Maximum Spanning Tree", "Dijkstra on Grid", "Knapsack DP", "KMP String Matching".
   - "core_challenge": A concise 1-sentence summary (10-20 words) of the exact task and its main constraint. e.g., "Connect N cities using N-1 roads to maximize total vehicles.", "Find shortest path in a grid with cell entry time costs."

Output the result using this exact JSON schema:
{{
  "paradigm": "string",
  "specific_technique": "string",
  "core_challenge": "string"
}}

Problem Description:
{problem_description}
"""